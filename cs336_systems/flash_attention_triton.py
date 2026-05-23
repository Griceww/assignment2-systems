from __future__ import annotations

import math
from typing import Any, cast

import torch
import triton
import triton.language as tl
from einops import rearrange


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    query_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    valid_query = query_idx < N_QUERIES

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        scores = tl.dot(q, tl.trans(k)) * scale
        key_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        valid_key = key_idx < N_KEYS

        valid = valid_query[:, None] & valid_key[None, :]
        if is_causal:
            valid = valid & (query_idx[:, None] >= key_idx[None, :])
        scores = tl.where(valid, scores, -1e6)

        m_prev = m_i
        m_i = tl.maximum(m_i, tl.max(scores, axis=1))
        p_tilde = tl.exp(scores - m_i[:, None])
        exp_m_delta = tl.exp(m_prev - m_i)
        l_i = exp_m_delta * l_i + tl.sum(p_tilde, axis=1)
        o_i = tl.dot(
            p_tilde.to(V_block_ptr.type.element_ty),
            v,
            acc=exp_m_delta[:, None] * o_i,
        )

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o_i / l_i[:, None]
    lse = m_i + tl.log(l_i)
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, lse.to(L_block_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dQ_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dqv,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D_HEAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    q_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_doq, stride_dod),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    do = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")

    query_idx = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    valid_query = query_idx < N_QUERIES
    l_i = tl.load(L_ptr + batch_index * stride_lb + query_idx * stride_lq, mask=valid_query, other=0.0)
    d_i = tl.load(D_ptr + batch_index * stride_db + query_idx * stride_dqv, mask=valid_query, other=0.0)

    dq_acc = tl.zeros((Q_TILE_SIZE, D_HEAD), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        key_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        valid_key = key_idx < N_KEYS

        scores = tl.dot(q, tl.trans(k)) * scale
        valid = valid_query[:, None] & valid_key[None, :]
        if is_causal:
            valid = valid & (query_idx[:, None] >= key_idx[None, :])
        scores = tl.where(valid, scores, -1e6)

        p = tl.exp(scores - l_i[:, None])
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d_i[:, None]) * scale
        ds = tl.where(valid, ds, 0.0)
        dq_acc = tl.dot(ds.to(K_block_ptr.type.element_ty), k, acc=dq_acc)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dq_acc.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    D_ptr,
    dK_ptr,
    dV_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dqv,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D_HEAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    k_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )

    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    key_idx = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    valid_key = key_idx < N_KEYS

    dk_acc = tl.zeros((K_TILE_SIZE, D_HEAD), dtype=tl.float32)
    dv_acc = tl.zeros((K_TILE_SIZE, D_HEAD), dtype=tl.float32)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D_HEAD),
            strides=(stride_qq, stride_qd),
            offsets=(i * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D_HEAD),
            order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D_HEAD),
            strides=(stride_doq, stride_dod),
            offsets=(i * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D_HEAD),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")

        query_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        valid_query = query_idx < N_QUERIES
        l_i = tl.load(L_ptr + batch_index * stride_lb + query_idx * stride_lq, mask=valid_query, other=0.0)
        d_i = tl.load(D_ptr + batch_index * stride_db + query_idx * stride_dqv, mask=valid_query, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        valid = valid_query[:, None] & valid_key[None, :]
        if is_causal:
            valid = valid & (query_idx[:, None] >= key_idx[None, :])
        scores = tl.where(valid, scores, -1e6)

        p = tl.exp(scores - l_i[:, None])
        p = tl.where(valid, p, 0.0)

        dv_acc = tl.dot(tl.trans(p.to(dO_block_ptr.type.element_ty)), do, acc=dv_acc)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d_i[:, None]) * scale
        ds = tl.where(valid, ds, 0.0)
        dk_acc = tl.dot(tl.trans(ds.to(Q_block_ptr.type.element_ty)), q, acc=dk_acc)

    tl.store(dK_block_ptr, dk_acc.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv_acc.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not (Q.is_cuda and K.is_cuda and V.is_cuda):
            raise ValueError("FlashAttention2Triton expects CUDA tensors.")
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError("Q, K, V must have the same shape for this implementation.")
        if not (Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()):
            raise ValueError("Q, K, V must be contiguous.")

        leading_shape = Q.shape[:-2]
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]

        q_flat = rearrange(Q, "... n d -> (...) n d")
        k_flat = rearrange(K, "... n d -> (...) n d")
        v_flat = rearrange(V, "... n d -> (...) n d")
        batch = q_flat.shape[0]

        q_tile_size = 64
        k_tile_size = 64
        scale = 1.0 / math.sqrt(d)

        o_flat = torch.empty_like(q_flat)
        l_flat = torch.empty((batch, n_queries), device=Q.device, dtype=torch.float32)

        launch_grid = (triton.cdiv(n_queries, q_tile_size), batch)
        flash_fwd_kernel[launch_grid](
            q_flat,
            k_flat,
            v_flat,
            o_flat,
            l_flat,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            o_flat.stride(0),
            o_flat.stride(1),
            o_flat.stride(2),
            l_flat.stride(0),
            l_flat.stride(1),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,
            is_causal=cast(Any, is_causal),
            D=cast(Any, d),
            Q_TILE_SIZE=cast(Any, q_tile_size),
            K_TILE_SIZE=cast(Any, k_tile_size),
        )

        O = o_flat.reshape(*leading_shape, n_queries, d)
        L = l_flat.reshape(*leading_shape, n_queries)
        ctx.save_for_backward(L, Q, K, V, O)
        setattr(ctx, "is_causal", is_causal)
        return O

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor):
        (dO,) = grad_outputs
        L, Q, K, V, O = cast(tuple[torch.Tensor, ...], getattr(ctx, "saved_tensors"))
        is_causal = bool(getattr(ctx, "is_causal", False))
        d = Q.shape[-1]
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        leading_shape = Q.shape[:-2]

        q_flat = rearrange(Q, "... n d -> (...) n d")
        k_flat = rearrange(K, "... n d -> (...) n d")
        v_flat = rearrange(V, "... n d -> (...) n d")
        o_flat = rearrange(O, "... n d -> (...) n d")
        do_flat = rearrange(dO.contiguous(), "... n d -> (...) n d")
        l_flat = rearrange(L, "... n -> (...) n")

        batch = q_flat.shape[0]
        q_tile_size = 64
        k_tile_size = 64
        scale = 1.0 / math.sqrt(d)

        d_vec = torch.sum(do_flat.to(torch.float32) * o_flat.to(torch.float32), dim=-1)
        dq_flat = torch.empty_like(q_flat)
        dk_flat = torch.empty_like(k_flat)
        dv_flat = torch.empty_like(v_flat)

        dq_grid = (triton.cdiv(n_queries, q_tile_size), batch)
        flash_bwd_dq_kernel[dq_grid](
            q_flat,
            k_flat,
            v_flat,
            do_flat,
            l_flat,
            d_vec,
            dq_flat,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            do_flat.stride(0),
            do_flat.stride(1),
            do_flat.stride(2),
            l_flat.stride(0),
            l_flat.stride(1),
            d_vec.stride(0),
            d_vec.stride(1),
            dq_flat.stride(0),
            dq_flat.stride(1),
            dq_flat.stride(2),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,
            is_causal=cast(Any, is_causal),
            D_HEAD=cast(Any, d),
            Q_TILE_SIZE=cast(Any, q_tile_size),
            K_TILE_SIZE=cast(Any, k_tile_size),
        )

        dkdv_grid = (triton.cdiv(n_keys, k_tile_size), batch)
        flash_bwd_dkdv_kernel[dkdv_grid](
            q_flat,
            k_flat,
            v_flat,
            do_flat,
            l_flat,
            d_vec,
            dk_flat,
            dv_flat,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            do_flat.stride(0),
            do_flat.stride(1),
            do_flat.stride(2),
            l_flat.stride(0),
            l_flat.stride(1),
            d_vec.stride(0),
            d_vec.stride(1),
            dk_flat.stride(0),
            dk_flat.stride(1),
            dk_flat.stride(2),
            dv_flat.stride(0),
            dv_flat.stride(1),
            dv_flat.stride(2),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,
            is_causal=cast(Any, is_causal),
            D_HEAD=cast(Any, d),
            Q_TILE_SIZE=cast(Any, q_tile_size),
            K_TILE_SIZE=cast(Any, k_tile_size),
        )

        dQ = dq_flat.reshape(*leading_shape, n_queries, d)
        dK = dk_flat.reshape(*leading_shape, n_keys, d)
        dV = dv_flat.reshape(*leading_shape, n_keys, d)
        return dQ, dK, dV, None
