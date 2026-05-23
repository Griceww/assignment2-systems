from __future__ import annotations

import math
from typing import cast

import torch
from einops import einsum, rearrange


def _flash_backward_recompute(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    q = Q.to(torch.float32)
    k = K.to(torch.float32)
    v = V.to(torch.float32)
    o = O.to(torch.float32)
    do = dO.to(torch.float32)
    l = L.to(torch.float32)

    scores = einsum(q, k, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        q_idx = torch.arange(n_queries, device=Q.device)
        k_idx = torch.arange(n_keys, device=Q.device)
        mask = q_idx[:, None] >= k_idx[None, :]
        scores = torch.where(mask, scores, -1e6)
    else:
        mask = None

    p = torch.exp(scores - l[..., :, None])
    if mask is not None:
        p = torch.where(mask, p, torch.zeros((), device=p.device, dtype=p.dtype))

    dv = einsum(p, do, "... q k, ... q d -> ... k d")
    dp = einsum(do, v, "... q d, ... k d -> ... q k")
    D = torch.sum(do * o, dim=-1)
    ds = p * (dp - D[..., :, None])
    if mask is not None:
        ds = torch.where(mask, ds, torch.zeros((), device=ds.device, dtype=ds.dtype))

    dQ = einsum(ds, k, "... q k, ... k d -> ... q d") * scale
    dK = einsum(ds, q, "... q k, ... q d -> ... k d") * scale
    dV = dv
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


class FlashAttention2PyTorch(torch.autograd.Function):
    """
    Pure PyTorch (no Triton) FlashAttention-2 forward pass.

    This implementation follows the online-softmax tiled algorithm and stores
    (L, Q, K, V, O) for backward compatibility with later tasks.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if Q.shape[:-2] != K.shape[:-2] or Q.shape[:-2] != V.shape[:-2]:
            raise ValueError("Q, K, V must share leading dimensions.")
        if Q.shape[-1] != K.shape[-1]:
            raise ValueError("Q and K must have the same head dimension.")
        if K.shape[-2] != V.shape[-2]:
            raise ValueError("K and V must have the same sequence length.")

        # Part (a) explicitly allows ignoring causal masking.
        _ = is_causal

        leading_shape = Q.shape[:-2]
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d_qk = Q.shape[-1]
        d_v = V.shape[-1]

        batch = math.prod(leading_shape) if len(leading_shape) > 0 else 1
        q_flat = Q.reshape(batch, n_queries, d_qk)
        k_flat = K.reshape(batch, n_keys, d_qk)
        v_flat = V.reshape(batch, n_keys, d_v)

        # Tile sizes must be >= 16x16 per the assignment.
        b_q = 64
        b_k = 64
        scale = 1.0 / math.sqrt(d_qk)

        out_acc = torch.empty((batch, n_queries, d_v), device=Q.device, dtype=torch.float32)
        lse = torch.empty((batch, n_queries), device=Q.device, dtype=torch.float32)

        for q_start in range(0, n_queries, b_q):
            q_end = min(q_start + b_q, n_queries)
            q_block = q_flat[:, q_start:q_end, :].to(torch.float32)
            q_block_size = q_end - q_start

            m_i = torch.full((batch, q_block_size), -float("inf"), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((batch, q_block_size), device=Q.device, dtype=torch.float32)
            o_i = torch.zeros((batch, q_block_size, d_v), device=Q.device, dtype=torch.float32)

            for k_start in range(0, n_keys, b_k):
                k_end = min(k_start + b_k, n_keys)
                k_block = k_flat[:, k_start:k_end, :].to(torch.float32)
                v_block = v_flat[:, k_start:k_end, :].to(torch.float32)

                # S_i^(j) = Q_i K_j^T / sqrt(d)
                scores = einsum(q_block, k_block, "b nq d, b nk d -> b nq nk") * scale
                # m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
                m_i_prev = m_i
                m_i = torch.maximum(m_i, torch.max(scores, dim=-1).values)
                # P_tilde_i^(j) = exp(S_i^(j) - m_i^(j))
                p_tilde = torch.exp(scores - m_i.unsqueeze(-1))
                # l_i^(j) = exp(m_i^(j-1)-m_i^(j)) * l_i^(j-1) + rowsum(P_tilde_i^(j))
                exp_m_delta = torch.exp(m_i_prev - m_i)
                l_i = exp_m_delta * l_i + torch.sum(p_tilde, dim=-1)
                # O_i^(j) = diag(exp(m_i^(j-1)-m_i^(j))) O_i^(j-1) + P_tilde_i^(j) V^(j)
                o_i = rearrange(exp_m_delta, "b nq -> b nq 1") * o_i + einsum(
                    p_tilde,
                    v_block,
                    "b nq nk, b nk dv -> b nq dv",
                )

            out_acc[:, q_start:q_end, :] = o_i / l_i.unsqueeze(-1)
            lse[:, q_start:q_end] = m_i + torch.log(l_i)

        O = out_acc.reshape(*leading_shape, n_queries, d_v).to(Q.dtype)
        L = lse.reshape(*leading_shape, n_queries)
        ctx.save_for_backward(L, Q, K, V, O)
        setattr(ctx, "is_causal", is_causal)
        return O

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor):
        (dO,) = grad_outputs
        L, Q, K, V, O = cast(tuple[torch.Tensor, ...], getattr(ctx, "saved_tensors"))
        is_causal = bool(getattr(ctx, "is_causal", False))
        dQ, dK, dV = _flash_backward_recompute(Q, K, V, O, dO, L, is_causal)
        return dQ, dK, dV, None
