from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import triton
from einops import einsum

from cs336_systems.flash_attention_triton import FlashAttention2Triton


@dataclass
class BenchRow:
    impl: str
    dtype: str
    seq_len: int
    d_model: int
    status: str
    forward_ms: float | None = None
    backward_ms: float | None = None
    end_to_end_ms: float | None = None
    note: str | None = None


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype)


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}"


def _print_markdown_table(rows: list[BenchRow]) -> None:
    print("| impl | dtype | seq_len | d_model | status | forward_ms | backward_ms | end_to_end_ms | note |")
    print("|:--|:--|--:|--:|:--|--:|--:|--:|:--|")
    for r in rows:
        print(
            f"| {r.impl} | {r.dtype} | {r.seq_len} | {r.d_model} | {r.status} | "
            f"{_fmt(r.forward_ms)} | {_fmt(r.backward_ms)} | {_fmt(r.end_to_end_ms)} | {r.note or ''} |"
        )


def _make_inputs(
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def _plain_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    d = q.shape[-1]
    scores = einsum(q, k, "... q d, ... k d -> ... q k") * (1.0 / math.sqrt(d))
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        q_idx = torch.arange(n_queries, device=q.device)
        k_idx = torch.arange(n_keys, device=q.device)
        mask = q_idx[:, None] >= k_idx[None, :]
        scores = torch.where(mask, scores, -1e6)
    p = torch.softmax(scores, dim=-1)
    return einsum(p, v, "... q k, ... k d -> ... q d")


def _run_forward(fn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    with torch.no_grad():
        return fn(q, k, v, is_causal)


def _run_backward(fn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> None:
    q.grad = None
    k.grad = None
    v.grad = None
    out = fn(q, k, v, is_causal)
    do = torch.randn_like(out)
    out.backward(do)


def _run_end_to_end(fn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> None:
    _run_backward(fn, q, k, v, is_causal)


def _measure_one_impl(
    impl_name: str,
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    warmup_ms: int,
    rep_ms: int,
) -> BenchRow:
    dtype_name = _dtype_name(q.dtype)
    seq_len = q.shape[-2]
    d_model = q.shape[-1]
    try:
        # Smoke call to surface shape/dtype errors before benchmarking.
        _run_forward(fn, q, k, v, is_causal)

        forward_ms = float(
            triton.testing.do_bench(
                lambda: _run_forward(fn, q, k, v, is_causal),
                warmup=warmup_ms,
                rep=rep_ms,
            )
        )
        backward_ms = float(
            triton.testing.do_bench(
                lambda: _run_backward(fn, q, k, v, is_causal),
                warmup=warmup_ms,
                rep=rep_ms,
            )
        )
        end_to_end_ms = float(
            triton.testing.do_bench(
                lambda: _run_end_to_end(fn, q, k, v, is_causal),
                warmup=warmup_ms,
                rep=rep_ms,
            )
        )
        return BenchRow(
            impl=impl_name,
            dtype=dtype_name,
            seq_len=seq_len,
            d_model=d_model,
            status="ok",
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            end_to_end_ms=end_to_end_ms,
        )
    except Exception as err:
        message = str(err)
        lowered = message.lower()
        status = "oom" if "out of memory" in lowered else "error"
        if q.device.type == "cuda":
            torch.cuda.empty_cache()
        return BenchRow(
            impl=impl_name,
            dtype=dtype_name,
            seq_len=seq_len,
            d_model=d_model,
            status=status,
            note=message.split("\n")[0][:160],
        )


def _run_grid_benchmark(
    device: torch.device,
    batch_size: int,
    seq_lens: list[int],
    d_models: list[int],
    dtypes: list[torch.dtype],
    warmup_ms: int,
    rep_ms: int,
) -> list[BenchRow]:
    rows: list[BenchRow] = []
    is_causal = True

    flash_fn = FlashAttention2Triton.apply
    torch_fn = _plain_attention

    for dtype in dtypes:
        for seq_len in seq_lens:
            for d_model in d_models:
                print(f"[run] dtype={_dtype_name(dtype)} seq_len={seq_len} d_model={d_model}")
                q, k, v = _make_inputs(batch_size, seq_len, d_model, dtype, device)
                rows.append(
                    _measure_one_impl(
                        "flash_triton",
                        flash_fn,
                        q,
                        k,
                        v,
                        is_causal=is_causal,
                        warmup_ms=warmup_ms,
                        rep_ms=rep_ms,
                    )
                )
                rows.append(
                    _measure_one_impl(
                        "torch_attention",
                        torch_fn,
                        q,
                        k,
                        v,
                        is_causal=is_causal,
                        warmup_ms=warmup_ms,
                        rep_ms=rep_ms,
                    )
                )
    return rows


def _run_leaderboard_baseline(
    warmup_ms: int,
    rep_ms: int,
    device: torch.device,
) -> BenchRow:
    n_heads = 16
    d_head = 64
    seq_len = 16384
    q, k, v = torch.randn(
        3,
        n_heads,
        seq_len,
        d_head,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    flash = torch.compile(FlashAttention2Triton.apply)

    def flash_forward_backward() -> None:
        q.grad = None
        k.grad = None
        v.grad = None
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    try:
        latency_ms = float(triton.testing.do_bench(flash_forward_backward, warmup=warmup_ms, rep=rep_ms))
        return BenchRow(
            impl="flash_triton_compiled_leaderboard",
            dtype="bfloat16",
            seq_len=seq_len,
            d_model=n_heads * d_head,
            status="ok",
            end_to_end_ms=latency_ms,
        )
    except Exception as err:
        message = str(err)
        lowered = message.lower()
        status = "oom" if "out of memory" in lowered else "error"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return BenchRow(
            impl="flash_triton_compiled_leaderboard",
            dtype="bfloat16",
            seq_len=seq_len,
            d_model=n_heads * d_head,
            status=status,
            note=message.split("\n")[0][:160],
        )


def _parse_int_powers(spec: str) -> list[int]:
    # Example: "7:16" => [2^7, 2^8, ..., 2^16]
    lo_str, hi_str = spec.split(":")
    lo = int(lo_str)
    hi = int(hi_str)
    if lo > hi:
        raise ValueError(f"Invalid exponent range: {spec}")
    return [2**p for p in range(lo, hi + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention2 Triton vs regular PyTorch attention.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-exp-range", type=str, default="7:16", help="Power-of-two seq range, e.g. 7:16")
    parser.add_argument("--d-exp-range", type=str, default="4:7", help="Power-of-two d range, e.g. 4:7")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["bfloat16", "float32"],
        choices=["bfloat16", "float32"],
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument(
        "--leaderboard-mode",
        action="store_true",
        help="Also run one leaderboard-style compiled flash fwd+bwd benchmark.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark script.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    seq_lens = _parse_int_powers(args.seq_exp_range)
    d_models = _parse_int_powers(args.d_exp_range)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    dtypes = [dtype_map[d] for d in args.dtypes]

    rows = _run_grid_benchmark(
        device=device,
        batch_size=args.batch_size,
        seq_lens=seq_lens,
        d_models=d_models,
        dtypes=dtypes,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )

    print("\n## Flash Benchmark Results (Markdown)")
    _print_markdown_table(rows)

    if args.leaderboard_mode:
        print("\n## Leaderboard Baseline")
        row = _run_leaderboard_baseline(
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            device=device,
        )
        _print_markdown_table([row])


if __name__ == "__main__":
    main()
