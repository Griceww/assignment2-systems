"""End-to-end forward / backward timing for BasicsTransformerLM (CS336 assignment 2)."""

from __future__ import annotations

import argparse
import contextlib
import math
import statistics
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from einops import einsum, rearrange

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW

PRESETS: dict[str, tuple[int, int, int, int]] = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (1600, 6400, 48, 25),
    "2.7B": (2560, 10240, 32, 32),
}

BenchmarkMode = Literal["forward", "forward_backward", "both", "train_step"]
MixedPrecisionMode = Literal["none", "bf16"]


@dataclass(frozen=True)
class BenchConfig:
    vocab_size: int
    batch_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    rope_theta: float
    device: torch.device
    use_cuda: bool


def start_memory_profiling(enabled: bool, use_cuda: bool, max_entries: int) -> None:
    if not enabled or not use_cuda:
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=max_entries)


def stop_memory_profiling(
    enabled: bool,
    use_cuda: bool,
    snapshot_path: Path,
) -> tuple[float | None, str | None]:
    if not enabled or not use_cuda:
        return None, None
    maybe_sync(use_cuda)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(snapshot_path))
    torch.cuda.memory._record_memory_history(enabled=None)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return peak_mb, str(snapshot_path)


def maybe_sync(use_cuda: bool) -> None:
    if use_cuda:
        torch.cuda.synchronize()


def nvtx_range(name: str, enabled: bool):
    if enabled:
        return nvtx.range(name)
    return contextlib.nullcontext()


def autocast_or_nullcontext(mixed_precision: MixedPrecisionMode, use_cuda: bool):
    if mixed_precision == "bf16" and use_cuda:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def maybe_patch_attention_with_nvtx(enabled: bool, use_cuda: bool) -> None:
    if not enabled:
        return

    def annotated_scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d_k = K.shape[-1]
        with nvtx_range("attention_scores_matmul", use_cuda):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with nvtx_range("attention_softmax", use_cuda):
            attention_weights = softmax(attention_scores, dim=-1)
        with nvtx_range("attention_output_matmul", use_cuda):
            return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def lm_cross_entropy_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token CE: logits[..., :-1] predicts input_ids[..., 1:]."""
    logits_flat = rearrange(logits[..., :-1, :], "b s v -> (b s) v")
    targets = rearrange(input_ids[..., 1:], "b s -> (b s)")
    return F.cross_entropy(logits_flat, targets)


def run_warmup(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    warmup: int,
    mode: BenchmarkMode,
    use_cuda: bool,
    nvtx_enabled: bool,
    mixed_precision: MixedPrecisionMode,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    with nvtx_range("warmup", nvtx_enabled):
        for _ in range(warmup):
            if mode == "forward":
                model.zero_grad(set_to_none=True)
                maybe_sync(use_cuda)
                with nvtx_range("forward", nvtx_enabled):
                    with autocast_or_nullcontext(mixed_precision, use_cuda):
                        logits = model(input_ids)
                with nvtx_range("loss", nvtx_enabled):
                    with autocast_or_nullcontext(mixed_precision, use_cuda):
                        loss = lm_cross_entropy_loss(logits, input_ids)
                # Keep forward-mode benchmarking in training semantics while
                # avoiding graph buildup across iterations.
                with nvtx_range("backward_cleanup", nvtx_enabled):
                    loss.backward()
                maybe_sync(use_cuda)
            elif mode == "forward_backward" or mode == "both":
                model.zero_grad(set_to_none=True)
                maybe_sync(use_cuda)
                with nvtx_range("forward", nvtx_enabled):
                    with autocast_or_nullcontext(mixed_precision, use_cuda):
                        logits = model(input_ids)
                with nvtx_range("loss", nvtx_enabled):
                    with autocast_or_nullcontext(mixed_precision, use_cuda):
                        loss = lm_cross_entropy_loss(logits, input_ids)
                with nvtx_range("backward", nvtx_enabled):
                    loss.backward()
                maybe_sync(use_cuda)
            else:  # train_step
                if optimizer is None:
                    raise ValueError("optimizer is required for train_step warmup")
                optimizer.zero_grad(set_to_none=True)
                maybe_sync(use_cuda)
                with nvtx_range("forward", nvtx_enabled):
                    with autocast_or_nullcontext(mixed_precision, use_cuda):
                        logits = model(input_ids)
                with nvtx_range("loss", nvtx_enabled):
                    with autocast_or_nullcontext(mixed_precision, use_cuda):
                        loss = lm_cross_entropy_loss(logits, input_ids)
                with nvtx_range("backward", nvtx_enabled):
                    loss.backward()
                with nvtx_range("optimizer_step", nvtx_enabled):
                    optimizer.step()
                maybe_sync(use_cuda)


def measure_forward_only(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    steps: int,
    use_cuda: bool,
    nvtx_enabled: bool,
    mixed_precision: MixedPrecisionMode,
    track_grad: bool = False,
    cleanup_backward: bool = False,
) -> list[float]:
    times: list[float] = []
    with nvtx_range("measurement", nvtx_enabled):
        for _ in range(steps):
            if track_grad:
                model.zero_grad(set_to_none=True)
            maybe_sync(use_cuda)
            t0 = timeit.default_timer()
            with nvtx_range("forward", nvtx_enabled):
                with autocast_or_nullcontext(mixed_precision, use_cuda):
                    context = contextlib.nullcontext() if track_grad else torch.inference_mode()
                    with context:
                        logits = model(input_ids)
            with nvtx_range("loss", nvtx_enabled):
                with autocast_or_nullcontext(mixed_precision, use_cuda):
                    loss = lm_cross_entropy_loss(logits, input_ids)
            maybe_sync(use_cuda)
            times.append(timeit.default_timer() - t0)
            if track_grad and cleanup_backward:
                with nvtx_range("backward_cleanup", nvtx_enabled):
                    loss.backward()
                maybe_sync(use_cuda)
    return times


def measure_forward_backward(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    steps: int,
    use_cuda: bool,
    nvtx_enabled: bool,
    mixed_precision: MixedPrecisionMode,
) -> list[float]:
    times: list[float] = []
    with nvtx_range("measurement", nvtx_enabled):
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            maybe_sync(use_cuda)
            t0 = timeit.default_timer()
            with nvtx_range("forward", nvtx_enabled):
                with autocast_or_nullcontext(mixed_precision, use_cuda):
                    logits = model(input_ids)
            with nvtx_range("loss", nvtx_enabled):
                with autocast_or_nullcontext(mixed_precision, use_cuda):
                    loss = lm_cross_entropy_loss(logits, input_ids)
            with nvtx_range("backward", nvtx_enabled):
                loss.backward()
            maybe_sync(use_cuda)
            times.append(timeit.default_timer() - t0)
    return times


def measure_forward_and_backward_split(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    steps: int,
    use_cuda: bool,
    nvtx_enabled: bool,
    mixed_precision: MixedPrecisionMode,
) -> tuple[list[float], list[float]]:
    """Measure training-style forward and backward in the same step.

    We time forward+loss and backward separately, but execute both every step so
    autograd graphs are consumed consistently.
    """
    forward_times: list[float] = []
    backward_times: list[float] = []
    with nvtx_range("measurement", nvtx_enabled):
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            maybe_sync(use_cuda)

            t0 = timeit.default_timer()
            with nvtx_range("forward", nvtx_enabled):
                with autocast_or_nullcontext(mixed_precision, use_cuda):
                    logits = model(input_ids)
            with nvtx_range("loss", nvtx_enabled):
                with autocast_or_nullcontext(mixed_precision, use_cuda):
                    loss = lm_cross_entropy_loss(logits, input_ids)
            maybe_sync(use_cuda)
            t1 = timeit.default_timer()

            with nvtx_range("backward", nvtx_enabled):
                loss.backward()
            maybe_sync(use_cuda)
            t2 = timeit.default_timer()

            forward_times.append(t1 - t0)
            backward_times.append(t2 - t1)
    return forward_times, backward_times


def measure_train_step(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    steps: int,
    use_cuda: bool,
    nvtx_enabled: bool,
    mixed_precision: MixedPrecisionMode,
) -> list[float]:
    times: list[float] = []
    with nvtx_range("measurement", nvtx_enabled):
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            maybe_sync(use_cuda)
            t0 = timeit.default_timer()
            with nvtx_range("forward", nvtx_enabled):
                with autocast_or_nullcontext(mixed_precision, use_cuda):
                    logits = model(input_ids)
            with nvtx_range("loss", nvtx_enabled):
                with autocast_or_nullcontext(mixed_precision, use_cuda):
                    loss = lm_cross_entropy_loss(logits, input_ids)
            with nvtx_range("backward", nvtx_enabled):
                loss.backward()
            with nvtx_range("optimizer_step", nvtx_enabled):
                optimizer.step()
            maybe_sync(use_cuda)
            times.append(timeit.default_timer() - t0)
    return times


def summarize_seconds(samples: list[float]) -> tuple[float, float]:
    mean_s = statistics.mean(samples)
    if len(samples) < 2:
        return mean_s, 0.0
    return mean_s, statistics.stdev(samples)


def parse_device(name: str) -> tuple[torch.device, bool]:
    if name == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA 不可用，回退到 CPU（计时含义与 GPU 不同）。", file=sys.stderr)
            return torch.device("cpu"), False
        return torch.device("cuda"), True
    return torch.device(name), False


def build_model(cfg: BenchConfig) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
    )
    return model.to(cfg.device)


def run_benchmark_for_config(
    cfg: BenchConfig,
    label: str,
    mode: BenchmarkMode,
    warmup: int,
    steps: int,
    use_cuda: bool,
    nvtx_enabled: bool,
    annotate_attention: bool,
    mixed_precision: MixedPrecisionMode,
    compile_mode: Literal["none", "compiled"],
    profile_memory: bool,
    memory_max_entries: int,
    memory_snapshot_dir: str,
) -> tuple[dict[str, float | str], str]:
    """Returns (table row dict, human-readable line)."""
    def finalize_with_memory(
        line: str,
        row: dict[str, float | str],
        peak_mb: float | None,
        dumped_snapshot: str | None,
    ) -> tuple[dict[str, float | str], str]:
        if peak_mb is not None:
            line += f"  peak_mem={peak_mb:.2f} MiB"
            row["peak_memory_mib"] = peak_mb
        if dumped_snapshot is not None:
            line += f"  snapshot={dumped_snapshot}"
            row["memory_snapshot"] = dumped_snapshot
        return row, line

    maybe_patch_attention_with_nvtx(annotate_attention, nvtx_enabled)
    model = build_model(cfg)
    compile_model = compile_mode == "compiled"
    if compile_model:
        model = cast(BasicsTransformerLM, torch.compile(model))
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, cfg.context_length),
        device=cfg.device,
        dtype=torch.long,
    )

    snapshot_path = (
        Path(memory_snapshot_dir)
        / f"{label}_ctx{cfg.context_length}_{mode}_{mixed_precision}_memory_snapshot.pickle"
    )

    if mode == "forward":
        run_warmup(model, input_ids, warmup, "forward", use_cuda, nvtx_enabled, mixed_precision)
        start_memory_profiling(profile_memory, use_cuda, memory_max_entries)
        samples = measure_forward_only(
            model,
            input_ids,
            steps,
            use_cuda,
            nvtx_enabled,
            mixed_precision,
            track_grad=True,
            cleanup_backward=True,
        )
        mean_s, std_s = summarize_seconds(samples)
        peak_mb, dumped_snapshot = stop_memory_profiling(profile_memory, use_cuda, snapshot_path)
        line = (
            f"[{label}] mode=forward mp={mixed_precision} compile_mode={compile_mode} warmup={warmup} steps={steps}  "
            f"mean={mean_s * 1e3:.4f} ms  std={std_s * 1e3:.4f} ms"
        )
        if peak_mb is not None:
            line += f"  peak_mem={peak_mb:.2f} MiB"
        if dumped_snapshot is not None:
            line += f"  snapshot={dumped_snapshot}"
        row = {
            "preset": label,
            "mixed_precision": mixed_precision,
            "compile_mode": compile_mode,
            "forward_ms_mean": mean_s * 1e3,
            "forward_ms_std": std_s * 1e3,
        }
        return finalize_with_memory(line, row, peak_mb, dumped_snapshot)

    if mode == "forward_backward":
        run_warmup(model, input_ids, warmup, "forward_backward", use_cuda, nvtx_enabled, mixed_precision)
        start_memory_profiling(profile_memory, use_cuda, memory_max_entries)
        samples = measure_forward_backward(model, input_ids, steps, use_cuda, nvtx_enabled, mixed_precision)
        mean_s, std_s = summarize_seconds(samples)
        peak_mb, dumped_snapshot = stop_memory_profiling(profile_memory, use_cuda, snapshot_path)
        line = (
            f"[{label}] mode=forward_backward mp={mixed_precision} compile_mode={compile_mode} warmup={warmup} steps={steps}  "
            f"mean={mean_s * 1e3:.4f} ms  std={std_s * 1e3:.4f} ms"
        )
        if peak_mb is not None:
            line += f"  peak_mem={peak_mb:.2f} MiB"
        if dumped_snapshot is not None:
            line += f"  snapshot={dumped_snapshot}"
        row = {
            "preset": label,
            "mixed_precision": mixed_precision,
            "compile_mode": compile_mode,
            "fwd_bwd_ms_mean": mean_s * 1e3,
            "fwd_bwd_ms_std": std_s * 1e3,
        }
        return finalize_with_memory(line, row, peak_mb, dumped_snapshot)

    if mode == "both":
        run_warmup(model, input_ids, warmup, "both", use_cuda, nvtx_enabled, mixed_precision)
        start_memory_profiling(profile_memory, use_cuda, memory_max_entries)
        fwd, bwd = measure_forward_and_backward_split(
            model, input_ids, steps, use_cuda, nvtx_enabled, mixed_precision
        )
        fm, fs = summarize_seconds(fwd)
        bm, bs = summarize_seconds(bwd)
        peak_mb, dumped_snapshot = stop_memory_profiling(profile_memory, use_cuda, snapshot_path)
        line = (
            f"[{label}] mode=both mp={mixed_precision} compile_mode={compile_mode} warmup={warmup} steps={steps}  "
            f"forward_mean={fm * 1e3:.4f} ms  forward_std={fs * 1e3:.4f} ms  "
            f"backward_mean={bm * 1e3:.4f} ms  backward_std={bs * 1e3:.4f} ms"
        )
        if peak_mb is not None:
            line += f"  peak_mem={peak_mb:.2f} MiB"
        if dumped_snapshot is not None:
            line += f"  snapshot={dumped_snapshot}"
        row = {
            "preset": label,
            "mixed_precision": mixed_precision,
            "compile_mode": compile_mode,
            "forward_ms_mean": fm * 1e3,
            "forward_ms_std": fs * 1e3,
            "backward_ms_mean": bm * 1e3,
            "backward_ms_std": bs * 1e3,
        }
        return finalize_with_memory(line, row, peak_mb, dumped_snapshot)

    run_warmup(model, input_ids, warmup, "train_step", use_cuda, nvtx_enabled, mixed_precision, optimizer)
    start_memory_profiling(profile_memory, use_cuda, memory_max_entries)
    samples = measure_train_step(model, input_ids, optimizer, steps, use_cuda, nvtx_enabled, mixed_precision)
    mean_s, std_s = summarize_seconds(samples)
    peak_mb, dumped_snapshot = stop_memory_profiling(profile_memory, use_cuda, snapshot_path)
    line = (
        f"[{label}] mode=train_step mp={mixed_precision} compile_mode={compile_mode} warmup={warmup} steps={steps}  "
        f"mean={mean_s * 1e3:.4f} ms  std={std_s * 1e3:.4f} ms"
    )
    if peak_mb is not None:
        line += f"  peak_mem={peak_mb:.2f} MiB"
    if dumped_snapshot is not None:
        line += f"  snapshot={dumped_snapshot}"
    row = {
        "preset": label,
        "mixed_precision": mixed_precision,
        "compile_mode": compile_mode,
        "train_step_ms_mean": mean_s * 1e3,
        "train_step_ms_std": std_s * 1e3,
    }
    return finalize_with_memory(line, row, peak_mb, dumped_snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM forward/backward.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Table 1 model size (sets d_model, d_ff, num_layers, num_heads).",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=sorted(PRESETS.keys()),
        default=None,
        metavar="NAME",
        help="Run several presets in one process and print a combined pandas table (e.g. --presets small medium).",
    )
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=["forward", "forward_backward", "both", "train_step"],
        default="both",
        help="forward: training-style forward+loss (builds grad graph, no backward); forward_backward: fwd+loss+bwd; both: training-style split Fwd/Bwd; train_step: fwd+bwd+optimizer.step.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mixed-precision",
        choices=["none", "bf16"],
        default="none",
        help="Enable mixed precision autocast (bf16 on CUDA) or disable it.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["none", "compiled"],
        default="none",
        help="Run vanilla model or compile the whole model with torch.compile.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable CUDA memory history profiling and dump a memory_snapshot pickle.",
    )
    parser.add_argument(
        "--memory-snapshot-dir",
        type=str,
        default="memory_snapshots",
        help="Directory where memory snapshot pickle files are saved.",
    )
    parser.add_argument(
        "--memory-max-entries",
        type=int,
        default=1_000_000,
        help="Max entries used by torch.cuda.memory._record_memory_history.",
    )
    parser.add_argument("--nvtx", action="store_true", help="Enable NVTX ranges for Nsight Systems profiling.")
    parser.add_argument(
        "--annotate-attention",
        action="store_true",
        help="Monkey-patch scaled_dot_product_attention with fine-grained NVTX ranges.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="After a single --preset run, also print one pandas markdown row.",
    )
    parser.add_argument(
        "--latex-table",
        action="store_true",
        help="With --presets, also print pandas DataFrame.to_latex() for writeup.",
    )
    args = parser.parse_args()

    has_explicit = (
        args.d_model is not None
        or args.d_ff is not None
        or args.num_layers is not None
        or args.num_heads is not None
    )
    multi = args.presets is not None
    if multi and (args.preset is not None or has_explicit):
        parser.error("Use --presets alone, or --preset / explicit hyperparameters — not both.")
    if not multi:
        if args.preset is None and not has_explicit:
            parser.error("Specify --preset, --presets, or all of --d-model --d-ff --num-layers --num-heads")
        if args.preset is not None and has_explicit:
            parser.error("Use either --preset or explicit hyperparameters, not both")

    torch.manual_seed(args.seed)
    device, use_cuda = parse_device(args.device)
    if args.mixed_precision == "bf16" and not use_cuda:
        print("Warning: BF16 mixed precision 仅在 CUDA 生效，当前将退化为 full precision。", file=sys.stderr)
    if args.profile_memory and not use_cuda:
        print("Warning: --profile-memory 仅在 CUDA 生效，当前不会记录 memory snapshot。", file=sys.stderr)

    if multi:
        import pandas as pd

        rows: list[dict[str, float | str]] = []
        for pname in args.presets:
            d_model, d_ff, num_layers, num_heads = PRESETS[pname]
            cfg = BenchConfig(
                vocab_size=args.vocab_size,
                batch_size=args.batch_size,
                context_length=args.context_length,
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_layers,
                num_heads=num_heads,
                rope_theta=args.rope_theta,
                device=device,
                use_cuda=use_cuda,
            )
            row, line = run_benchmark_for_config(
                cfg,
                pname,
                args.mode,
                args.warmup,
                args.steps,
                use_cuda,
                args.nvtx and use_cuda,
                args.annotate_attention,
                args.mixed_precision,
                args.compile_mode,
                args.profile_memory,
                args.memory_max_entries,
                args.memory_snapshot_dir,
            )
            rows.append(row)
            print(line)
        df = pd.DataFrame(rows)
        print("\n### Markdown（可粘贴 writeup）\n")
        print(df.to_markdown(index=False))
        if args.latex_table:
            print("\n### LaTeX\n")
            print(df.to_latex(index=False))
        return

    if args.preset is not None:
        d_model, d_ff, num_layers, num_heads = PRESETS[args.preset]
        label = args.preset
    else:
        assert args.d_model is not None and args.d_ff is not None
        assert args.num_layers is not None and args.num_heads is not None
        d_model, d_ff, num_layers, num_heads = args.d_model, args.d_ff, args.num_layers, args.num_heads
        label = f"d{d_model}_L{num_layers}"

    cfg = BenchConfig(
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        rope_theta=args.rope_theta,
        device=device,
        use_cuda=use_cuda,
    )
    row, line = run_benchmark_for_config(
        cfg,
        label,
        args.mode,
        args.warmup,
        args.steps,
        use_cuda,
        args.nvtx and use_cuda,
        args.annotate_attention,
        args.mixed_precision,
        args.compile_mode,
        args.profile_memory,
        args.memory_max_entries,
        args.memory_snapshot_dir,
    )
    print(line)

    if args.markdown:
        import pandas as pd

        print(pd.DataFrame([row]).to_markdown(index=False))


if __name__ == "__main__":
    main()
