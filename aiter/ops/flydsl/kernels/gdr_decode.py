# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import (
    gpu as mlir_gpu,
)
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from .tensor_shim import (
    GTensor,
    _to_raw,
    get_dtype_bytes,
    get_dtype_in_kernel,
)


@functools.lru_cache(maxsize=1024)
def create_vk_gdr_decode_kernel(
    dtype: str,
    A_log_dtype: str,
    state_dtype: str,
    seq_length: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    q_strides: tuple,
    k_strides: tuple,
    v_strides: tuple,
    state_strides: tuple,
    a_strides: tuple,
    b_strides: tuple,
    use_qk_l2norm: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    NUM_BLOCKS_PER_V_DIM: int = 1,
    NUM_WARPS: int = 4,
    WARP_THREADS_K: int = 8,
):
    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K

    if "f32" in state_dtype:
        VALUES_PER_THREAD_K = 4  # 16B
    else:
        VALUES_PER_THREAD_K = 8  # 16B

    WARP_SIZE = WARP_THREADS_V * WARP_THREADS_K
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert WARP_SIZE == 64

    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    WARP_TILE_K_ITERS = head_k_dim // WARP_TILE_K
    assert WARP_TILE_K_ITERS >= 1
    assert head_k_dim % WARP_TILE_K == 0

    WARP_TILE_V = WARP_THREADS_V
    WARP_GROUP_TILE_V = NUM_WARPS * WARP_TILE_V
    TILE_V = head_v_dim // NUM_BLOCKS_PER_V_DIM
    WARP_TILE_V_ITERS = TILE_V // WARP_GROUP_TILE_V
    assert TILE_V >= 1 and head_v_dim % NUM_BLOCKS_PER_V_DIM == 0
    assert WARP_TILE_V_ITERS >= 1 and TILE_V % WARP_GROUP_TILE_V == 0

    WARP_THREADS_K_SHFL_OFFSETS = []
    offsets_ = WARP_THREADS_K // 2
    while offsets_ >= 1:
        WARP_THREADS_K_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    WARP_THREADS_K_SHFL_OFFSETS = WARP_THREADS_K_SHFL_OFFSETS[::-1]

    WARP_SIZE_SHFL_OFFSETS = []
    offsets_ = WARP_SIZE // 2
    while offsets_ >= 1:
        WARP_SIZE_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2

    KERNEL_NAME = f"gdr_decode_{dtype}_kh{num_k_heads}x{head_k_dim}_vh{num_v_heads}x{head_v_dim}_q{seq_length}"
    KERNEL_NAME += f"_{NUM_WARPS}w{WARP_THREADS_V}x{WARP_THREADS_K}"
    KERNEL_NAME += f"_vs{NUM_BLOCKS_PER_V_DIM}"

    @flyc.kernel
    def gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        read_indices: fx.Tensor,
        write_indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
    ):
        scale = fx.Float32(SCALE_VALUE)
        softplus_beta_ = fx.Float32(softplus_beta)
        softplus_threshold_ = fx.Float32(softplus_threshold)

        dtype_ = get_dtype_in_kernel(dtype)
        fx_dtype_ = fx.BFloat16 if dtype == "bf16" else fx.Float16
        A_log_dtype_ = get_dtype_in_kernel(A_log_dtype)
        state_dtype_ = get_dtype_in_kernel(state_dtype)
        f32_0 = fx.Float32(0.0)
        f32_1 = fx.Float32(1.0)
        width_i32 = _to_raw(fx.Int32(WARP_SIZE))
        vec_t = T.vec(VALUES_PER_THREAD_K, dtype_)
        acc_vec_t = T.vec(VALUES_PER_THREAD_K, T.f32)

        tidx = fx.thread_idx.x
        bidx = fx.block_idx.x
        w_tid = tidx % WARP_SIZE
        wid = tidx // WARP_SIZE

        b_hv_i = bidx // NUM_BLOCKS_PER_V_DIM
        tile_v_start = bidx % NUM_BLOCKS_PER_V_DIM * TILE_V

        b_i = b_hv_i // num_v_heads
        hv_i = b_hv_i % num_v_heads
        hk_i = hv_i // (num_v_heads // num_k_heads)

        warp_k_vec_start = w_tid % WARP_THREADS_K * VALUES_PER_THREAD_K
        global_v_start = tile_v_start + wid * WARP_TILE_V + w_tid // WARP_THREADS_K

        read_indices_tensor = GTensor(read_indices, dtype=T.i32, shape=(-1,))
        write_indices_tensor = GTensor(write_indices, dtype=T.i32, shape=(-1,))
        read_pool_idx = fx.Int32(read_indices_tensor[b_i])
        write_pool_idx = fx.Int32(write_indices_tensor[b_i])

        q_tensor = GTensor(
            query,
            dtype=dtype_,
            shape=(-1, seq_length, num_k_heads, head_k_dim),
            stride=q_strides,
        )
        k_tensor = GTensor(
            key,
            dtype=dtype_,
            shape=(-1, seq_length, num_k_heads, head_k_dim),
            stride=k_strides,
        )
        v_tensor = GTensor(
            value,
            dtype=dtype_,
            shape=(-1, seq_length, num_v_heads, head_v_dim),
            stride=v_strides,
        )
        a_tensor = GTensor(
            a,
            dtype=dtype_,
            stride=(a_strides[0], a_strides[1], a_strides[2]),
            shape=(-1, seq_length, num_v_heads),
        )
        b_tensor = GTensor(
            b,
            dtype=dtype_,
            stride=(b_strides[0], b_strides[1], b_strides[2]),
            shape=(-1, seq_length, num_v_heads),
        )
        dt_bias_tensor = GTensor(dt_bias, dtype=dtype_, shape=(num_v_heads,))
        A_log_tensor = GTensor(A_log, dtype=A_log_dtype_, shape=(num_v_heads,))
        out_tensor = GTensor(
            out, dtype=dtype_, shape=(-1, seq_length, num_v_heads, head_v_dim)
        )
        read_state_tensor = GTensor(
            state,
            dtype=state_dtype_,
            shape=(num_v_heads, head_v_dim, head_k_dim),
            stride=(state_strides[1], state_strides[2], state_strides[3]),
            static_bytes_offset_i64=fx.Int64(read_pool_idx)
            * fx.Int64(state_strides[0])
            * get_dtype_bytes(state_dtype),
        )
        write_state_tensor = GTensor(
            state,
            dtype=state_dtype_,
            shape=(num_v_heads, head_v_dim, head_k_dim),
            stride=(state_strides[1], state_strides[2], state_strides[3]),
            static_bytes_offset_i64=fx.Int64(write_pool_idx)
            * fx.Int64(state_strides[0])
            * get_dtype_bytes(state_dtype),
        )

        def fast_exp(x, use_exp2=True):
            if const_expr(use_exp2):
                log2e = 1.4426950408889634
                return rocdl.exp2(T.f32, _to_raw(fx.Float32(x) * log2e))
            return fx.math.exp(x, fastmath=fx.FastMathFlags.fast)

        def fast_log1p(x):
            return fx.math.log1p(x, fastmath=fx.FastMathFlags.fast)

        # Skip CG-pad slots (indices sentinel < 0). The guarded body is a
        # closure so the runtime `if` sees an opaque call (no GTensor "state"
        # to thread through an scf.if yield) -- lowers to scf.if, no raw region.
        def _do_decode():
            if const_expr("f32" in A_log_dtype):
                r_A_log = A_log_tensor[hv_i]
            else:
                r_A_log = A_log_tensor[hv_i].extf(T.f32)
            r_dt_bias = dt_bias_tensor[hv_i].extf(T.f32)

            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = (
                        read_state_tensor.vec_load(
                            (hv_i, global_v_i, warp_k_vec_i), VALUES_PER_THREAD_K
                        )
                    )
                    if const_expr("f32" in state_dtype):
                        pass
                    else:
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_vecs[
                            vi * WARP_TILE_K_ITERS + ki
                        ].extf(acc_vec_t)

            for sq_i in range_constexpr(seq_length):
                r_a = a_tensor[b_i, sq_i, hv_i].extf(T.f32)
                r_b = b_tensor[b_i, sq_i, hv_i].extf(T.f32)
                x = r_a + r_dt_bias
                beta_x = softplus_beta_ * x

                # softplus with the large-x identity: for beta_x > threshold,
                # softplus(x) == x. select computes both arms (the overflow arm
                # is discarded) -> bit-identical to the old branch.
                softplus_big = (f32_1 / softplus_beta_) * fast_log1p(fast_exp(beta_x))
                softplus_x = (
                    fx.Float32(beta_x) <= fx.Float32(softplus_threshold_)
                ).select(softplus_big, x)

                r_g_value = -fast_exp(r_A_log) * softplus_x
                r_beta = f32_1 / (f32_1 + fast_exp(-r_b))
                r_g = fast_exp(r_g_value)

                r_g_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(r_g), fx.Float32
                )

                sq_vecs = [0] * WARP_TILE_K_ITERS
                sk_vecs = [0] * WARP_TILE_K_ITERS

                scale_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(scale), fx.Float32
                )

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    q_vec = q_tensor.vec_load(
                        (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                    )
                    k_vec = k_tensor.vec_load(
                        (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                    )
                    sq_vecs[ki] = q_vec.extf(acc_vec_t)
                    sk_vecs[ki] = k_vec.extf(acc_vec_t)

                if const_expr(use_qk_l2norm):
                    sum_q_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_k_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sum_q_partial_vec = (
                            sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                        )
                        sum_k_partial_vec = (
                            sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
                        )
                    sum_q_partial = fx.Vector(sum_q_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    sum_k_partial = fx.Vector(sum_k_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_q_partial = sum_q_partial + sum_q_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                        sum_k_partial = sum_k_partial + sum_k_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                    local_sum_q = mlir_gpu.ShuffleOp(
                        _to_raw(sum_q_partial),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    local_sum_k = mlir_gpu.ShuffleOp(
                        _to_raw(sum_k_partial),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    inv_norm_q = fx.math.rsqrt(local_sum_q + 1e-6)
                    inv_norm_k = fx.math.rsqrt(local_sum_k + 1e-6)
                    inv_norm_q_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_q), fx.Float32
                    )
                    inv_norm_k_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_k), fx.Float32
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * inv_norm_q_vec * scale_vec
                        sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec
                else:
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec

                dot_kq_vec = fx.Vector.from_elements(
                    [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)], fx.Float32
                )
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    dot_kq_vec = fx.math.fma(sk_vecs[ki], sq_vecs[ki], dot_kq_vec)
                dot_kq = dot_kq_vec.reduce(fx.ReductionOp.ADD)
                for offset in WARP_THREADS_K_SHFL_OFFSETS:
                    dot_kq = dot_kq + dot_kq.shuffle_xor(offset, WARP_SIZE)

                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    r_v = v_tensor[b_i, sq_i, hv_i, global_v_i].extf(T.f32)

                    sum_hk = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_hq_old = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                        h_cur = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        sum_hk = fx.math.fma(h_cur, sk_vecs[ki], sum_hk)
                        sum_hq_old = fx.math.fma(h_cur, sq_vecs[ki], sum_hq_old)

                    sum_hk = sum_hk.reduce(fx.ReductionOp.ADD)
                    sum_hq_old = sum_hq_old.reduce(fx.ReductionOp.ADD)

                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_hk = sum_hk + sum_hk.shuffle_xor(offset, WARP_SIZE)
                        sum_hq_old = sum_hq_old + sum_hq_old.shuffle_xor(
                            offset, WARP_SIZE
                        )

                    v_new = (r_v - sum_hk) * r_beta
                    v_new = mlir_gpu.ShuffleOp(
                        _to_raw(v_new),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    sum_hq = sum_hq_old + v_new * dot_kq
                    v_new_bcast = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(v_new), fx.Float32
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        h_new = fx.math.fma(
                            sk_vecs[ki],
                            v_new_bcast,
                            state_vecs[vi * WARP_TILE_K_ITERS + ki],
                        )
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new

                    sum_hq = sum_hq.to(fx_dtype_)

                    # Only k-vec lane 0 writes the q output; closure keeps the
                    # GTensor store opaque to the runtime-if state analysis.
                    def _write_q(_sum_hq=sum_hq, _gv=global_v_i, _sq=sq_i):
                        out_tensor[b_i, _sq, hv_i, _gv] = _sum_hq

                    if warp_k_vec_start == 0:
                        _write_q()

            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    if const_expr("f32" in state_dtype):
                        out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                    else:
                        out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki].truncf(vec_t)
                    write_state_tensor.vec_store(
                        (hv_i, global_v_i, warp_k_vec_i), out_vec, VALUES_PER_THREAD_K
                    )

        if (read_pool_idx >= 0) & (write_pool_idx >= 0):
            _do_decode()

    @flyc.jit
    def launch_gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        read_indices: fx.Tensor,
        write_indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        gx = batch_size * num_v_heads * NUM_BLOCKS_PER_V_DIM
        gdr_decode_kernel._func.__name__ = KERNEL_NAME
        gdr_decode_kernel(
            query,
            key,
            value,
            a,
            b,
            dt_bias,
            A_log,
            read_indices,
            write_indices,
            state,
            out,
            batch_size,
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_gdr_decode_kernel


MTP_MODE_CHAIN = "chain"
MTP_MODE_SNAPSHOT = "snapshot"


@functools.lru_cache(maxsize=1024)
def create_vk_gdr_mtp_kernel(
    dtype: str,
    A_log_dtype: str,
    state_dtype: str,
    inter_dtype: str,
    seq_length: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    q_strides: tuple,
    k_strides: tuple,
    v_strides: tuple,
    state_strides: tuple,
    a_strides: tuple,
    b_strides: tuple,
    si_strides: tuple,
    inter_strides: tuple,
    parent_strides: tuple,
    use_qk_l2norm: bool,
    mode: str,
    has_tree: bool = False,
    disable_state_update: bool = False,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    NUM_BLOCKS_PER_V_DIM: int = 1,
    NUM_WARPS: int = 4,
    WARP_THREADS_K: int = 8,
):
    """Gated delta rule over a speculative draft window.

    The decode kernel above runs ``seq_length`` tokens straight through and
    stores the state once at the end. That is not verify: verify has to be able
    to *undo* tokens the target model rejects, which means both a rollback point
    to start from and a per-token record to roll back to. The two upstreams
    disagree on how to keep that record, so ``mode`` picks between them.

    ``MTP_MODE_CHAIN`` is vLLM's. The draft is a linear chain, the rollback
    point is the slot at ``state_indices[n, num_accepted - 1]``, and the record
    is the state pool itself: every token checkpoints into its own slot
    ``state_indices[n, t]``. There is no separate final store, because the last
    token's checkpoint already is it.

    ``MTP_MODE_SNAPSHOT`` is SGLang's. The rollback point is the sequence's one
    slot ``state_indices[n]``, and the record lives outside the pool, in
    ``intermediate_states_buffer``. With ``has_tree`` the draft is an EAGLE tree
    rather than a chain, so each token restarts from its parent's snapshot
    instead of from the previous token's state; ``disable_state_update``
    suppresses the write-back to the pool, which is how a verify pass leaves the
    committed state alone.

    Both modes keep the same 64/32 address split the state pool already forced
    on the decode kernel: the MUBUF offset operand is 32 bits, so anything
    scaling with the pool -- or, worse, with the snapshot buffer, whose slot
    pitch is ``cache_steps`` times larger -- has to be folded into the 64-bit
    descriptor base instead. Every base below is built that way, and both
    factors are widened before they are multiplied, since it is the product that
    overflows.
    """
    assert mode in (MTP_MODE_CHAIN, MTP_MODE_SNAPSHOT), f"unknown MTP mode {mode!r}"
    CHAIN = mode == MTP_MODE_CHAIN
    SNAPSHOT = not CHAIN
    TREE = bool(has_tree)
    NO_STATE_WRITE = bool(disable_state_update)
    # The snapshot buffer is what the tree reloads parents from, so a tree
    # without one has nothing to roll back to.
    assert not TREE or SNAPSHOT, "the EAGLE tree is the snapshot mode's"
    assert not TREE or len(inter_strides) == 5, "tree needs a snapshot buffer"
    SAVE_INTER = SNAPSHOT and len(inter_strides) == 5

    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K

    if "f32" in state_dtype:
        VALUES_PER_THREAD_K = 4  # 16B
    else:
        VALUES_PER_THREAD_K = 8  # 16B

    WARP_SIZE = WARP_THREADS_V * WARP_THREADS_K
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert WARP_SIZE == 64

    # The tiling arithmetic repeats the decode builder's rather than sharing a
    # helper with it: that kernel is tuned and shipping, and a shared helper is a
    # way for a change made for MTP to reach it.
    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    WARP_TILE_K_ITERS = head_k_dim // WARP_TILE_K
    assert WARP_TILE_K_ITERS >= 1
    assert head_k_dim % WARP_TILE_K == 0

    WARP_TILE_V = WARP_THREADS_V
    WARP_GROUP_TILE_V = NUM_WARPS * WARP_TILE_V
    TILE_V = head_v_dim // NUM_BLOCKS_PER_V_DIM
    WARP_TILE_V_ITERS = TILE_V // WARP_GROUP_TILE_V
    assert TILE_V >= 1 and head_v_dim % NUM_BLOCKS_PER_V_DIM == 0
    assert WARP_TILE_V_ITERS >= 1 and TILE_V % WARP_GROUP_TILE_V == 0

    WARP_THREADS_K_SHFL_OFFSETS = []
    offsets_ = WARP_THREADS_K // 2
    while offsets_ >= 1:
        WARP_THREADS_K_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    WARP_THREADS_K_SHFL_OFFSETS = WARP_THREADS_K_SHFL_OFFSETS[::-1]

    STATE_BYTES = get_dtype_bytes(state_dtype)
    INTER_BYTES = get_dtype_bytes(inter_dtype) if SAVE_INTER else 0

    # The snapshot borrows the state's lane count -- VALUES_PER_THREAD_K is
    # picked so the *state* vector is 16 bytes, and the snapshot reuses it with
    # its own element -- so a snapshot wider than the state overruns what a
    # buffer op carries. Caught here as well as at the API, because this is the
    # layer that cannot express the store: without it the combination reaches
    # the backend as a 32-byte store and dies there as `Cannot select`.
    assert not SAVE_INTER or VALUES_PER_THREAD_K * INTER_BYTES <= 16, (
        f"a {state_dtype} state splits K {VALUES_PER_THREAD_K} ways, so a "
        f"{inter_dtype} snapshot needs a "
        f"{VALUES_PER_THREAD_K * INTER_BYTES}-byte store; the snapshot dtype "
        f"cannot be wider than the state's"
    )

    KERNEL_NAME = f"gdr_mtp_{mode}_{dtype}_kh{num_k_heads}x{head_k_dim}_vh{num_v_heads}x{head_v_dim}_q{seq_length}"
    if TREE:
        KERNEL_NAME += "_tree"
    if SAVE_INTER:
        KERNEL_NAME += "_snap"
    if NO_STATE_WRITE:
        KERNEL_NAME += "_nowrite"
    KERNEL_NAME += f"_{NUM_WARPS}w{WARP_THREADS_V}x{WARP_THREADS_K}"
    KERNEL_NAME += f"_vs{NUM_BLOCKS_PER_V_DIM}"

    @flyc.kernel
    def gdr_mtp_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        state_indices: fx.Tensor,
        num_accepted: fx.Tensor,
        inter_indices: fx.Tensor,
        parent_tokens: fx.Tensor,
        state: fx.Tensor,
        inter_buffer: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
    ):
        scale = fx.Float32(SCALE_VALUE)
        softplus_beta_ = fx.Float32(softplus_beta)
        softplus_threshold_ = fx.Float32(softplus_threshold)

        dtype_ = get_dtype_in_kernel(dtype)
        fx_dtype_ = fx.BFloat16 if dtype == "bf16" else fx.Float16
        A_log_dtype_ = get_dtype_in_kernel(A_log_dtype)
        state_dtype_ = get_dtype_in_kernel(state_dtype)
        f32_0 = fx.Float32(0.0)
        f32_1 = fx.Float32(1.0)
        width_i32 = _to_raw(fx.Int32(WARP_SIZE))
        state_vec_t = T.vec(VALUES_PER_THREAD_K, state_dtype_)
        acc_vec_t = T.vec(VALUES_PER_THREAD_K, T.f32)

        tidx = fx.thread_idx.x
        bidx = fx.block_idx.x
        w_tid = tidx % WARP_SIZE
        wid = tidx // WARP_SIZE

        b_hv_i = bidx // NUM_BLOCKS_PER_V_DIM
        tile_v_start = bidx % NUM_BLOCKS_PER_V_DIM * TILE_V

        b_i = b_hv_i // num_v_heads
        hv_i = b_hv_i % num_v_heads
        hk_i = hv_i // (num_v_heads // num_k_heads)

        warp_k_vec_start = w_tid % WARP_THREADS_K * VALUES_PER_THREAD_K
        global_v_start = tile_v_start + wid * WARP_TILE_V + w_tid // WARP_THREADS_K

        # Flat views: the index tensors are addressed with the caller's strides
        # rather than a shape, so a 1-D [B] and a 2-D [B, T] map the same way.
        si_tensor = GTensor(state_indices, dtype=T.i32, shape=(-1,))

        # Read every token's checkpoint slot here rather than at the token that
        # writes it.
        #
        # The chain's slots are `seq_length` independent int32s, but read one
        # per token they turned into one dependent round trip per token: the
        # load is issued inside the token's own body and the compare that gates
        # its checkpoint waits at vmcnt(0), which drains every other load in
        # flight along with it. That was 20% of stall cycles at batch 1, spread
        # evenly across the loop, and it is the one part of the token's inputs
        # the lookahead below does not already cover.
        #
        # Reading them together costs `seq_length` registers and lets one wait
        # cover all of them, inside the prologue that is already waiting on the
        # rollback lookup. The snapshot contract has a single slot for the whole
        # sequence, so there is nothing to spread there.
        if const_expr(CHAIN):
            token_slots = [
                fx.Int32(si_tensor[b_i * si_strides[0] + t * si_strides[1]])
                for t in range_constexpr(seq_length)
            ]

        def _slot_at(token):
            if const_expr(CHAIN):
                return token_slots[token]
            return fx.Int32(si_tensor[b_i * si_strides[0]])

        # What counts as a dead slot, which the two contracts spell differently.
        # SGLang pads with a negative sentinel and slot 0 is an ordinary slot;
        # vLLM reserves slot 0 as its null block (`state_idx <= 0` skips the
        # sequence, `final_state_idx > 0` gates the checkpoint) and aiter's
        # Triton passes a negative sentinel through the same entry point. So the
        # chain has to reject both and the snapshot mode must not reject slot 0.
        MIN_LIVE_SLOT = 1 if CHAIN else 0

        # Rollback point. The chain rolls back to the last accepted token's
        # checkpoint; the snapshot mode's sequence has a single slot and rolls
        # back through the snapshot buffer instead.
        if const_expr(CHAIN):
            nacc_tensor = GTensor(num_accepted, dtype=T.i32, shape=(-1,))
            read_token = fx.Int32(nacc_tensor[b_i]) - fx.Int32(1)
            read_slot = fx.Int32(
                si_tensor[b_i * si_strides[0] + read_token * si_strides[1]]
            )
        else:
            read_slot = fx.Int32(si_tensor[b_i * si_strides[0]])

        if const_expr(SAVE_INTER):
            isi_tensor = GTensor(inter_indices, dtype=T.i32, shape=(-1,))
            cache_idx = fx.Int32(isi_tensor[b_i])
        if const_expr(TREE):
            parent_tensor = GTensor(parent_tokens, dtype=T.i32, shape=(-1,))

        q_tensor = GTensor(
            query,
            dtype=dtype_,
            shape=(-1, seq_length, num_k_heads, head_k_dim),
            stride=q_strides,
        )
        k_tensor = GTensor(
            key,
            dtype=dtype_,
            shape=(-1, seq_length, num_k_heads, head_k_dim),
            stride=k_strides,
        )
        v_tensor = GTensor(
            value,
            dtype=dtype_,
            shape=(-1, seq_length, num_v_heads, head_v_dim),
            stride=v_strides,
        )
        a_tensor = GTensor(
            a,
            dtype=dtype_,
            stride=(a_strides[0], a_strides[1], a_strides[2]),
            shape=(-1, seq_length, num_v_heads),
        )
        b_tensor = GTensor(
            b,
            dtype=dtype_,
            stride=(b_strides[0], b_strides[1], b_strides[2]),
            shape=(-1, seq_length, num_v_heads),
        )
        dt_bias_tensor = GTensor(dt_bias, dtype=dtype_, shape=(num_v_heads,))
        A_log_tensor = GTensor(A_log, dtype=A_log_dtype_, shape=(num_v_heads,))
        out_tensor = GTensor(
            out, dtype=dtype_, shape=(-1, seq_length, num_v_heads, head_v_dim)
        )

        def _state_at(slot):
            """State-pool view whose descriptor base already carries the slot.

            ``slot * state_strides[0]`` is the term that scales with the pool, so
            it is the one that cannot sit in the 32-bit buffer offset. Both
            factors are widened before multiplying.
            """
            return GTensor(
                state,
                dtype=state_dtype_,
                shape=(num_v_heads, head_v_dim, head_k_dim),
                stride=(state_strides[1], state_strides[2], state_strides[3]),
                static_bytes_offset_i64=fx.Int64(slot)
                * fx.Int64(state_strides[0])
                * STATE_BYTES,
            )

        def _inter_at(slot, step):
            """Snapshot view for one (sequence slot, draft step).

            This is the address that overflows first: a slot here spans
            ``cache_steps`` states where the pool spans one, so the same pool
            size crosses 2^31 elements ``cache_steps`` times sooner. Slot and
            step are folded into the 64-bit base separately, for the same reason
            the pool's is.
            """
            inter_dtype_ = get_dtype_in_kernel(inter_dtype)
            return GTensor(
                inter_buffer,
                dtype=inter_dtype_,
                shape=(num_v_heads, head_v_dim, head_k_dim),
                stride=(inter_strides[2], inter_strides[3], inter_strides[4]),
                static_bytes_offset_i64=(
                    fx.Int64(slot) * fx.Int64(inter_strides[0])
                    + fx.Int64(step) * fx.Int64(inter_strides[1])
                )
                * INTER_BYTES,
            )

        def fast_exp(x, use_exp2=True):
            if const_expr(use_exp2):
                log2e = 1.4426950408889634
                return rocdl.exp2(T.f32, _to_raw(fx.Float32(x) * log2e))
            return fx.math.exp(x, fastmath=fx.FastMathFlags.fast)

        def fast_log1p(x):
            return fx.math.log1p(x, fastmath=fx.FastMathFlags.fast)

        # Skip CG-pad slots (indices sentinel < 0). The guarded body is a
        # closure so the runtime `if` sees an opaque call (no GTensor "state"
        # to thread through an scf.if yield) -- lowers to scf.if, no raw region.
        #
        # ``reload_parents`` and ``snapshot`` are traced, not runtime, flags.
        # They have to be: the parent reload *produces* the running state, and a
        # value defined inside an scf.if does not dominate its use after it, so
        # the reload cannot sit under a runtime test. What makes hoisting it
        # sound is that ``cache_idx`` is fixed for the whole kernel -- one
        # sequence either has a snapshot slot or it does not -- so the test
        # belongs at the entry, once, rather than at every token. Emitting the
        # body twice buys back the per-token snapshot guard as well, since each
        # copy knows statically whether the slot is live.
        def _do_mtp(reload_parents=False, snapshot="no"):
            if const_expr("f32" in A_log_dtype):
                r_A_log = A_log_tensor[hv_i]
            else:
                r_A_log = A_log_tensor[hv_i].extf(T.f32)
            r_dt_bias = dt_bias_tensor[hv_i].extf(T.f32)

            read_state_tensor = _state_at(read_slot)
            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = (
                        read_state_tensor.vec_load(
                            (hv_i, global_v_i, warp_k_vec_i), VALUES_PER_THREAD_K
                        )
                    )
                    if const_expr("f32" in state_dtype):
                        pass
                    else:
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_vecs[
                            vi * WARP_TILE_K_ITERS + ki
                        ].extf(acc_vec_t)

            def _taps(sq):
                """The gate and value scalars one token reads, issued together."""
                return (
                    a_tensor[b_i, sq, hv_i],
                    b_tensor[b_i, sq, hv_i],
                    [
                        v_tensor[b_i, sq, hv_i, global_v_start + vi * WARP_GROUP_TILE_V]
                        for vi in range_constexpr(WARP_TILE_V_ITERS)
                    ],
                )

            # Read those a token ahead of where they are used.
            #
            # A token ends by storing its state, and nothing tells the compiler
            # that the store cannot land on the gate or value inputs, so a read
            # placed where it is used may not be hoisted over the store in front
            # of it. Every token then opens with a round trip that has only its
            # own gating arithmetic to hide it. Issuing the next token's reads
            # before this one's stores puts a whole token of work in front of
            # that round trip instead, and taking a token's value reads as a
            # group lets one wait cover the whole v loop rather than one wait per
            # iteration.
            #
            # Depth one, not the whole window: the taps stay unwidened, one
            # register each, so depth d costs `d * (2 + WARP_TILE_V_ITERS)`
            # registers, and hoisting all four tokens measured slower
            # everywhere. A token of work is already enough to cover the round
            # trip, so further depth buys nothing and the registers it sits in
            # cost occupancy.
            taps = _taps(0)

            for sq_i in range_constexpr(seq_length):
                # EAGLE tree: restart from the parent token's snapshot. Token 0
                # has no parent and keeps the rollback state loaded above. The
                # reload happens even when the parent is the previous token,
                # rather than reusing what is already in registers, because the
                # snapshot may be stored at a narrower dtype than the
                # accumulator -- re-reading it is a rounding the chain would not
                # otherwise take, and upstream takes it.
                if const_expr(reload_parents and sq_i != 0):
                    parent_step = fx.Int32(
                        parent_tensor[
                            b_i * parent_strides[0] + sq_i * parent_strides[1]
                        ]
                    )
                    parent_view = _inter_at(cache_idx, parent_step)
                    for vi in range_constexpr(WARP_TILE_V_ITERS):
                        gv = global_v_start + vi * WARP_GROUP_TILE_V
                        for ki in range_constexpr(WARP_TILE_K_ITERS):
                            kv = warp_k_vec_start + ki * WARP_TILE_K
                            loaded = parent_view.vec_load(
                                (hv_i, gv, kv), VALUES_PER_THREAD_K
                            )
                            if const_expr("f32" in inter_dtype):
                                state_vecs[vi * WARP_TILE_K_ITERS + ki] = loaded
                            else:
                                state_vecs[vi * WARP_TILE_K_ITERS + ki] = loaded.extf(
                                    acc_vec_t
                                )

                tap_a, tap_b, r_v_raw = taps
                if const_expr(sq_i + 1 < seq_length):
                    taps = _taps(sq_i + 1)

                r_a = tap_a.extf(T.f32)
                r_b = tap_b.extf(T.f32)
                x = r_a + r_dt_bias
                beta_x = softplus_beta_ * x

                # softplus with the large-x identity: for beta_x > threshold,
                # softplus(x) == x. select computes both arms (the overflow arm
                # is discarded) -> bit-identical to the old branch.
                softplus_big = (f32_1 / softplus_beta_) * fast_log1p(fast_exp(beta_x))
                softplus_x = (
                    fx.Float32(beta_x) <= fx.Float32(softplus_threshold_)
                ).select(softplus_big, x)

                r_g_value = -fast_exp(r_A_log) * softplus_x
                r_beta = f32_1 / (f32_1 + fast_exp(-r_b))
                r_g = fast_exp(r_g_value)

                r_g_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(r_g), fx.Float32
                )

                sq_vecs = [0] * WARP_TILE_K_ITERS
                sk_vecs = [0] * WARP_TILE_K_ITERS

                scale_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(scale), fx.Float32
                )

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    q_vec = q_tensor.vec_load(
                        (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                    )
                    k_vec = k_tensor.vec_load(
                        (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                    )
                    sq_vecs[ki] = q_vec.extf(acc_vec_t)
                    sk_vecs[ki] = k_vec.extf(acc_vec_t)

                if const_expr(use_qk_l2norm):
                    sum_q_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_k_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sum_q_partial_vec = (
                            sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                        )
                        sum_k_partial_vec = (
                            sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
                        )
                    sum_q_partial = fx.Vector(sum_q_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    sum_k_partial = fx.Vector(sum_k_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_q_partial = sum_q_partial + sum_q_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                        sum_k_partial = sum_k_partial + sum_k_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                    local_sum_q = mlir_gpu.ShuffleOp(
                        _to_raw(sum_q_partial),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    local_sum_k = mlir_gpu.ShuffleOp(
                        _to_raw(sum_k_partial),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    inv_norm_q = fx.math.rsqrt(local_sum_q + 1e-6)
                    inv_norm_k = fx.math.rsqrt(local_sum_k + 1e-6)
                    inv_norm_q_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_q), fx.Float32
                    )
                    inv_norm_k_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_k), fx.Float32
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * inv_norm_q_vec * scale_vec
                        sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec
                else:
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec

                dot_kq_vec = fx.Vector.from_elements(
                    [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)], fx.Float32
                )
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    dot_kq_vec = fx.math.fma(sk_vecs[ki], sq_vecs[ki], dot_kq_vec)
                dot_kq = dot_kq_vec.reduce(fx.ReductionOp.ADD)
                for offset in WARP_THREADS_K_SHFL_OFFSETS:
                    dot_kq = dot_kq + dot_kq.shuffle_xor(offset, WARP_SIZE)

                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    r_v = r_v_raw[vi].extf(T.f32)

                    sum_hk = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_hq_old = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                        h_cur = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        sum_hk = fx.math.fma(h_cur, sk_vecs[ki], sum_hk)
                        sum_hq_old = fx.math.fma(h_cur, sq_vecs[ki], sum_hq_old)

                    sum_hk = sum_hk.reduce(fx.ReductionOp.ADD)
                    sum_hq_old = sum_hq_old.reduce(fx.ReductionOp.ADD)

                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_hk = sum_hk + sum_hk.shuffle_xor(offset, WARP_SIZE)
                        sum_hq_old = sum_hq_old + sum_hq_old.shuffle_xor(
                            offset, WARP_SIZE
                        )

                    v_new = (r_v - sum_hk) * r_beta
                    v_new = mlir_gpu.ShuffleOp(
                        _to_raw(v_new),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    sum_hq = sum_hq_old + v_new * dot_kq
                    v_new_bcast = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(v_new), fx.Float32
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        h_new = fx.math.fma(
                            sk_vecs[ki],
                            v_new_bcast,
                            state_vecs[vi * WARP_TILE_K_ITERS + ki],
                        )
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new

                    sum_hq = sum_hq.to(fx_dtype_)

                    # Only k-vec lane 0 writes the q output; closure keeps the
                    # GTensor store opaque to the runtime-if state analysis.
                    def _write_q(_sum_hq=sum_hq, _gv=global_v_i, _sq=sq_i):
                        out_tensor[b_i, _sq, hv_i, _gv] = _sum_hq

                    if warp_k_vec_start == 0:
                        _write_q()

                # Per-token record. This is the whole difference between verify
                # and running the draft straight through: without it there is
                # nothing for a later rejection to roll back to.
                if const_expr(CHAIN):
                    write_slot = _slot_at(sq_i)
                    write_view = _state_at(write_slot)

                    def _checkpoint(_view=write_view):
                        for vi in range_constexpr(WARP_TILE_V_ITERS):
                            gv = global_v_start + vi * WARP_GROUP_TILE_V
                            for ki in range_constexpr(WARP_TILE_K_ITERS):
                                kv = warp_k_vec_start + ki * WARP_TILE_K
                                acc = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                                if const_expr("f32" in state_dtype):
                                    out_vec = acc
                                else:
                                    out_vec = acc.truncf(state_vec_t)
                                _view.vec_store(
                                    (hv_i, gv, kv), out_vec, VALUES_PER_THREAD_K
                                )

                    if write_slot >= MIN_LIVE_SLOT:
                        _checkpoint()

                if const_expr(snapshot != "no"):
                    snap_view = _inter_at(cache_idx, sq_i)
                    inter_vec_t = T.vec(
                        VALUES_PER_THREAD_K, get_dtype_in_kernel(inter_dtype)
                    )

                    def _snapshot(_view=snap_view, _vec_t=inter_vec_t):
                        for vi in range_constexpr(WARP_TILE_V_ITERS):
                            gv = global_v_start + vi * WARP_GROUP_TILE_V
                            for ki in range_constexpr(WARP_TILE_K_ITERS):
                                kv = warp_k_vec_start + ki * WARP_TILE_K
                                acc = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                                if const_expr("f32" in inter_dtype):
                                    out_vec = acc
                                else:
                                    out_vec = acc.truncf(_vec_t)
                                _view.vec_store(
                                    (hv_i, gv, kv), out_vec, VALUES_PER_THREAD_K
                                )

                    if const_expr(snapshot == "always"):
                        _snapshot()
                    else:
                        if cache_idx >= 0:
                            _snapshot()

            # The chain has already written the last token's state to its own
            # slot, so only the snapshot mode has a final store left to make --
            # and a verify pass asks for it to be suppressed.
            if const_expr(SNAPSHOT and not NO_STATE_WRITE):
                write_view = _state_at(read_slot)
                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                        acc = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        if const_expr("f32" in state_dtype):
                            out_vec = acc
                        else:
                            out_vec = acc.truncf(state_vec_t)
                        write_view.vec_store(
                            (hv_i, global_v_i, warp_k_vec_i),
                            out_vec,
                            VALUES_PER_THREAD_K,
                        )

        # One entry test per traced body, flat rather than nested, so no scf.if
        # ever has to carry a value out of itself. In tree mode the two arms are
        # complementary: a sequence with a snapshot slot reloads its parents, a
        # sequence without one has nothing to reload and no snapshot to write,
        # which is the same thing the chain does.
        if const_expr(TREE):
            if (read_slot >= MIN_LIVE_SLOT) & (cache_idx >= 0):
                _do_mtp(reload_parents=True, snapshot="always")
            if (read_slot >= MIN_LIVE_SLOT) & (cache_idx < 0):
                _do_mtp(reload_parents=False, snapshot="no")
        else:
            if read_slot >= MIN_LIVE_SLOT:
                _do_mtp(
                    reload_parents=False,
                    snapshot="guarded" if SAVE_INTER else "no",
                )

    @flyc.jit
    def launch_gdr_mtp_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        state_indices: fx.Tensor,
        num_accepted: fx.Tensor,
        inter_indices: fx.Tensor,
        parent_tokens: fx.Tensor,
        state: fx.Tensor,
        inter_buffer: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        gx = batch_size * num_v_heads * NUM_BLOCKS_PER_V_DIM
        gdr_mtp_kernel._func.__name__ = KERNEL_NAME
        gdr_mtp_kernel(
            query,
            key,
            value,
            a,
            b,
            dt_bias,
            A_log,
            state_indices,
            num_accepted,
            inter_indices,
            parent_tokens,
            state,
            inter_buffer,
            out,
            batch_size,
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_gdr_mtp_kernel
