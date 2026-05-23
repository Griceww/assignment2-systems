"""Benchmark PyTorch scaled-dot-product attention for assignment 1.2.1."""

from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from cs336_basics.model import scaled_dot_product_attention


@dataclass
class BenchResult:
    compile_mode: str
    d_model: int
    seq_len: int
    status: str
    forward_ms_mean: float | None = None
    backward_ms_mean: float | None = None
    memory_before_backward_mib: float | None = None
    peak_memory_mib: float | None = None
    oom_message: str | None = None


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _maybe_make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    return idx[:, None] >= idx[None, :]


def _mean_ms(samples: list[float]) -> float:
    return (sum(samples) / len(samples)) * 1000.0


def _benchmark_one_config(
    attention_fn,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    seq_len: int,
    d_model: int,
    warmup_steps: int,
    measure_steps: int,
    use_causal_mask: bool,
) -> BenchResult:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    mask = _maybe_make_causal_mask(seq_len, device) if use_causal_mask else None

    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

    # Warmup forward
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = attention_fn(q, k, v, mask=mask)
        _cuda_sync(device)

    forward_times: list[float] = []
    for _ in range(measure_steps):
        _cuda_sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = attention_fn(q, k, v, mask=mask)
        _cuda_sync(device)
        forward_times.append(time.perf_counter() - start)

    # Warmup backward
    for _ in range(warmup_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        out = attention_fn(q, k, v, mask=mask)
        loss = out.sum()
        loss.backward()
        _cuda_sync(device)

    backward_times: list[float] = []
    mem_before_backward_samples: list[float] = []
    for _ in range(measure_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        out = attention_fn(q, k, v, mask=mask)
        loss = out.sum()

        if device.type == "cuda":
            _cuda_sync(device)
            mem_before_backward_samples.append(torch.cuda.memory_allocated(device=device) / (1024**2))
        else:
            mem_before_backward_samples.append(float("nan"))

        _cuda_sync(device)
        start = time.perf_counter()
        loss.backward()
        _cuda_sync(device)
        backward_times.append(time.perf_counter() - start)

    peak_memory_mib = None
    if device.type == "cuda":
        peak_memory_mib = torch.cuda.max_memory_allocated(device=device) / (1024**2)

    return BenchResult(
        compile_mode="",
        d_model=d_model,
        seq_len=seq_len,
        status="ok",
        forward_ms_mean=_mean_ms(forward_times),
        backward_ms_mean=_mean_ms(backward_times),
        memory_before_backward_mib=sum(mem_before_backward_samples) / len(mem_before_backward_samples),
        peak_memory_mib=peak_memory_mib,
    )


def _is_oom_error(err: RuntimeError) -> bool:
    return "out of memory" in str(err).lower()


def _format_float(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "-"
    return f"{v:.{digits}f}"


def _print_markdown_table(results: list[BenchResult]) -> None:
    print("| compile_mode | d_model | seq_len | status | forward_ms_mean | backward_ms_mean | mem_before_bwd_MiB | peak_mem_MiB |")
    print("|:---|---:|---:|:---|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r.compile_mode} | {r.d_model} | {r.seq_len} | {r.status} | "
            f"{_format_float(r.forward_ms_mean)} | {_format_float(r.backward_ms_mean)} | "
            f"{_format_float(r.memory_before_backward_mib)} | {_format_float(r.peak_memory_mib)} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark plain PyTorch attention across (d_model, seq_len).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run benchmark on (default: cuda).")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument(
        "--compile-mode",
        choices=["none", "compiled"],
        default="none",
        help="Run vanilla attention or compiled attention.",
    )
    parser.add_argument(
        "--no-causal-mask",
        action="store_true",
        help="Disable causal mask. By default a causal mask is applied.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请改用 --device cpu 或在有 GPU 的环境运行。")

    device = torch.device(args.device)
    dtype_map: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    results: list[BenchResult] = []
    use_causal_mask = not args.no_causal_mask
    compile_modes = [args.compile_mode]

    for compile_mode in compile_modes:
        attention_fn = scaled_dot_product_attention
        if compile_mode == "compiled":
            attention_fn = torch.compile(scaled_dot_product_attention)

        for d_model in args.d_models:
            for seq_len in args.seq_lens:
                print(f"[run] compile_mode={compile_mode}, d_model={d_model}, seq_len={seq_len}")
                try:
                    result = _benchmark_one_config(
                        attention_fn=attention_fn,
                        device=device,
                        dtype=dtype,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        d_model=d_model,
                        warmup_steps=args.warmup_steps,
                        measure_steps=args.measure_steps,
                        use_causal_mask=use_causal_mask,
                    )
                    result.compile_mode = compile_mode
                    results.append(result)
                    print(
                        f"  -> ok, fwd={result.forward_ms_mean:.3f} ms, "
                        f"bwd={result.backward_ms_mean:.3f} ms, "
                        f"mem_before_bwd={_format_float(result.memory_before_backward_mib)} MiB"
                    )
                except RuntimeError as err:
                    if _is_oom_error(err):
                        results.append(
                            BenchResult(
                                compile_mode=compile_mode,
                                d_model=d_model,
                                seq_len=seq_len,
                                status="oom",
                                oom_message=str(err),
                            )
                        )
                        print("  -> OOM")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise
                finally:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    print("\n## Results (Markdown)")
    _print_markdown_table(results)

    for compile_mode in compile_modes:
        oom_results = [r for r in results if r.status == "oom" and r.compile_mode == compile_mode]
        if oom_results:
            first = oom_results[0]
            print(f"\n首次出现 OOM 的配置（compile_mode={compile_mode}）：")
            print(f"- d_model={first.d_model}, seq_len={first.seq_len}")
            if first.oom_message is not None:
                print(f"- message: {first.oom_message}")
        else:
            print(f"\n在当前扫描范围内未出现 OOM（compile_mode={compile_mode}）。")


if __name__ == "__main__":
    main()
