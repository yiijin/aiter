# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Linear Attention APIs."""

from __future__ import annotations

import csv
import functools
import os
from pathlib import Path

import torch
from flydsl.runtime.device import get_rocm_arch

from .kernels.gdr_decode import (
    MTP_MODE_CHAIN,
    MTP_MODE_SNAPSHOT,
    create_vk_gdr_decode_kernel,
    create_vk_gdr_mtp_kernel,
)
from .kernels.tensor_shim import _run_compiled, get_dtype_str

__all__ = [
    "flydsl_gdr_decode",
    "flydsl_gdr_mtp",
    "flydsl_gdr_mtp_sglang",
]


GDR_GLOBAL_CONFIG_MAP = None
GDR_GPU_ARCH = get_rocm_arch()

# Which kernel a tuned row was measured against. The three MTP contracts tile
# differently at the same shape -- at batch 32 the chain wants (4, 2, 8), the
# snapshot (8, 1, 16) and the tree (1, 4, 8), with no overlap -- so the row has
# to name its contract or the three overwrite each other.
GDR_VARIANT_DECODE = "decode"


def _mtp_variant(mode, has_tree):
    """The table's name for an MTP contract."""
    return f"{mode}_tree" if has_tree else mode


def _tuned_config(
    dtype_str,
    state_dtype_str,
    batch_size,
    seq_length,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    variant=GDR_VARIANT_DECODE,
):
    """The tuned row for this shape and kernel variant, or None.

    Split out of ``get_default_kwargs`` so the MTP path can consult the same
    table while starting from a different default. The table wins wherever it
    has a row, on either path.
    """
    global GDR_GLOBAL_CONFIG_MAP
    if GDR_GLOBAL_CONFIG_MAP is None:
        _dict = {}
        fname = os.path.join(Path(__file__).resolve().parent, "gdr_decode_tuned.csv")
        with open(fname, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = dict(row)
                arch, b, sq, nkh, nvh, khd, vhd = (
                    obj["arch"],
                    int(obj["b"]),
                    int(obj["sq"]),
                    int(obj["num_k_heads"]),
                    int(obj["num_v_heads"]),
                    int(obj["head_k_dim"]),
                    int(obj["head_v_dim"]),
                )
                d_str, sd_str = obj["dtype"], obj["state_dtype"]
                var = obj.get("variant") or GDR_VARIANT_DECODE
                if float(obj["duration"]) < 10000.0:
                    _dict[(d_str, sd_str, arch, var, b, sq, nkh, nvh, khd, vhd)] = {
                        "NUM_BLOCKS_PER_V_DIM": int(obj["NUM_BLOCKS_PER_V_DIM"]),
                        "NUM_WARPS": int(obj["NUM_WARPS"]),
                        "WARP_THREADS_K": int(obj["WARP_THREADS_K"]),
                    }
        GDR_GLOBAL_CONFIG_MAP = _dict
    return GDR_GLOBAL_CONFIG_MAP.get(
        (
            dtype_str,
            state_dtype_str,
            GDR_GPU_ARCH,
            variant,
            batch_size,
            seq_length,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        ),
        None,
    )


def get_default_kwargs(
    dtype_str,
    state_dtype_str,
    batch_size,
    seq_length,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    d = {}
    d["NUM_BLOCKS_PER_V_DIM"] = 1
    d["NUM_WARPS"] = 4
    d["WARP_THREADS_K"] = 8
    config = _tuned_config(
        dtype_str,
        state_dtype_str,
        batch_size,
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    if config:
        d.update(config)
    return d


# Raising NUM_BLOCKS_PER_V_DIM past this stopped paying: by then the grid
# already covers the machine, and the narrower value tile costs more in warps
# than the extra blocks return.
_MTP_MAX_V_SPLIT = 8
# Blocks per CU to aim for. Two was not enough -- at 2x the rule stops splitting
# while batch 2 and 4 still gain from more blocks.
_MTP_BLOCKS_PER_CU = 4
_MTP_WARPS = 4
_CU_COUNT = {}


def _cu_count(device):
    idx = 0 if device.index is None else device.index
    if idx not in _CU_COUNT:
        _CU_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _CU_COUNT[idx]


def _mtp_tiling(batch_size, num_v_heads, head_k_dim, head_v_dim, state_dtype, target):
    """Split the value dimension until the grid covers the machine.

    ``NUM_BLOCKS_PER_V_DIM`` and ``NUM_WARPS`` are not independent: a warp group
    covers ``NUM_WARPS * (64 // WARP_THREADS_K)`` of a value tile that is
    ``head_v_dim // NUM_BLOCKS_PER_V_DIM`` wide, so their product has to divide
    the tile. Splitting therefore costs warps, and they are given back here
    rather than left for the builder to reject.

    ``WARP_THREADS_K`` is the second lever and the reason for the retry.
    Widening the K group narrows a warp's value footprint, which raises the
    product the tiling admits and so allows a further split -- at the price of
    one more stage in the cross-lane reduction. That trade is worth taking only
    where splitting alone cannot fill the machine: measured, the wider group is
    the best config at batch 1 and 2 and costs 6-12% from batch 32 up. So the
    narrow group is tried first and kept unless it leaves the grid short.
    """
    values_per_thread_k = 4 if state_dtype == torch.float32 else 8
    best = None
    for warp_threads_k in (8, 16):
        if head_k_dim % (warp_threads_k * values_per_thread_k):
            continue
        warp_threads_v = 64 // warp_threads_k
        limit = head_v_dim // warp_threads_v  # largest num_blocks * num_warps
        if limit < 1:
            continue
        num_blocks = 1
        while (
            num_blocks < _MTP_MAX_V_SPLIT
            and num_blocks * _MTP_WARPS <= limit
            and head_v_dim % (num_blocks * 2) == 0
            and batch_size * num_v_heads * num_blocks < target
        ):
            num_blocks *= 2
        best = {
            "NUM_BLOCKS_PER_V_DIM": num_blocks,
            "NUM_WARPS": max(1, min(_MTP_WARPS, limit // num_blocks)),
            "WARP_THREADS_K": warp_threads_k,
        }
        if batch_size * num_v_heads * num_blocks >= target:
            break
    return best


def get_mtp_default_kwargs(*args):
    """Wrapper returning a fresh dict, so no caller can edit the cached one."""
    return dict(_mtp_kwargs(*args))


@functools.lru_cache(maxsize=1024)
def _mtp_kwargs(
    dtype_str,
    state_dtype_str,
    state_dtype,
    batch_size,
    seq_length,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    device,
    variant,
):
    """Pick a tiling for the MTP kernel, then let the tuned table override it.

    The decode default splits the value dimension not at all, which is right
    for it: decode is called at the batch a serving step accumulates, so
    ``batch * num_v_heads`` already covers the machine. Verify is called at the
    batch that has draft tokens outstanding, which is small by construction,
    and the same default then launches ``num_v_heads`` blocks -- 32 on a
    Qwen3-Next-shaped model -- onto a part with hundreds of CUs.

    Measured on gfx950 at that shape, bf16 in / fp32 state, against a full sweep
    of the 54 valid configs with the kernel timed under graph capture so the
    launch path cannot mask it: this recovers 1.73x / 1.50x / 1.23x / 1.18x at
    batch 1 / 2 / 4 / 8 with a 4-token window and 1.66x / 1.47x / 1.17x / 1.04x
    with a 2-token window, is flat from batch 16 up where the default already
    fills the grid, and lands a mean 1.02x off the per-shape optimum against the
    default's 1.22x.

    What it does not close is the last 1.02x, which is concentrated above batch
    32 and reaches 1.08x at the worst shape. That is left to the table rather
    than to a sharper rule because the three contracts want different splits at
    the same shape -- at batch 32, (4, 2, 8), (8, 1, 16) and (1, 4, 8), with no
    overlap -- so rows are keyed by contract and this stays the fallback for the
    shapes a sweep has not reached.
    """
    d = _mtp_tiling(
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state_dtype,
        _MTP_BLOCKS_PER_CU * _cu_count(device),
    )
    if d is None:
        # No tiling fits; hand back the decode default and let the builder be
        # the one to refuse it, with its own message.
        d = {"NUM_BLOCKS_PER_V_DIM": 1, "NUM_WARPS": 4, "WARP_THREADS_K": 8}
    config = _tuned_config(
        dtype_str,
        state_dtype_str,
        batch_size,
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        variant,
    )
    if config:
        d.update(config)
    return d


def flydsl_gdr_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    stream: torch.cuda.Stream = None,
    read_indices: torch.Tensor | None = None,
    write_indices: torch.Tensor | None = None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    device = query.device
    dtype = query.dtype
    read_indices = indices if read_indices is None else read_indices
    write_indices = indices if write_indices is None else write_indices
    for input in [
        query,
        key,
        value,
        a,
        b,
        dt_bias,
        A_log,
        read_indices,
        write_indices,
        out,
    ]:
        assert input.device == device
    assert state.data_ptr() % 16 == 0
    for input in [key, value, a, b, dt_bias, out]:
        assert input.dtype == dtype
    assert state.dtype in [torch.float, torch.bfloat16]
    assert A_log.dtype in [torch.float, torch.bfloat16]
    assert read_indices.dtype == torch.int32
    assert write_indices.dtype == torch.int32
    if query.stride(-1) != 1:
        raise ValueError(
            "`query` must have a contiguous last dimension for vectorized loads; "
            f"got stride {query.stride()}."
        )
    if key.stride(-1) != 1:
        raise ValueError(
            "`key` must have a contiguous last dimension for vectorized loads; "
            f"got stride {key.stride()}."
        )

    if need_shuffle_state:
        state_ = state.permute(0, 1, 3, 2).contiguous()
    else:
        state_ = state
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]
    kwargs_ = get_default_kwargs(
        str(dtype),
        str(state_.dtype),
        batch_size,
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    exe = create_vk_gdr_decode_kernel(
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        get_dtype_str(state_.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        query.stride(),
        key.stride(),
        value.stride(),
        state_.stride(),
        a.stride(),
        b.stride(),
        use_qk_l2norm,
        **kwargs_,
    )
    with torch.cuda.device(query.device.index):
        _run_compiled(
            exe,
            query,
            key,
            value,
            a,
            b,
            dt_bias.contiguous(),
            A_log.contiguous(),
            read_indices.contiguous(),
            write_indices.contiguous(),
            state_,
            out,
            batch_size,
            stream,
        )
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_SUPPORTED_STATE_DTYPES = (torch.float32, torch.bfloat16)
_SUPPORTED_ARCHS = ("gfx942", "gfx950")


def _is_supported_arch(device: torch.device) -> bool:
    try:
        arch = str(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:  # noqa: BLE001 - no live device, meta/CPU tensor
        return False
    return arch.split(":")[0] in _SUPPORTED_ARCHS


def _unit_strided(t: torch.Tensor) -> bool:
    """Whether FlyDSL can wrap ``t`` as a memref.

    It needs one axis it can call the fastest-moving one, and a length-1 tensor
    sliced out of a wider row has none -- torch still calls that contiguous, so
    the usual check passes it through to a compile error. The index vectors are
    the ones this happens to, since a caller reaches for a column of its slot
    map.
    """
    return any(s == 1 for s in t.stride())


def _mtp_shapes_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
) -> bool:
    """Shape / dtype / placement screening shared by both MTP interfaces.

    The tiling the kernel builds assumes a head dimension it can cover with
    whole 16-byte vectors, and the state's K axis is what those vectors walk, so
    a state whose K is not the fastest-moving axis cannot be served at all.
    """
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4 or state.dim() != 4:
        return False
    if query.dtype not in _SUPPORTED_DTYPES:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if state.dtype not in _SUPPORTED_STATE_DTYPES:
        return False
    if not (query.is_cuda and state.is_cuda):
        return False
    if query.device != state.device or query.device != value.device:
        return False
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return False
    # state is [pool, HV, V, K]; the kernel vector-loads along K.
    if state.stride(-1) != 1:
        return False

    head_k_dim = query.shape[-1]
    head_v_dim = value.shape[-1]
    num_k_heads = query.shape[-2]
    num_v_heads = value.shape[-2]
    if num_v_heads % num_k_heads != 0:
        return False
    if state.shape[1] != num_v_heads or state.shape[2] != head_v_dim:
        return False
    if state.shape[3] != head_k_dim:
        return False

    # A 16-byte vector holds 4 fp32 or 8 bf16 state elements, and the default
    # 8-lane K split has to tile head_k_dim with whole vectors.
    values_per_thread_k = 4 if state.dtype == torch.float32 else 8
    if head_k_dim % (8 * values_per_thread_k) != 0:
        return False
    if head_v_dim % 32 != 0:
        return False
    return _is_supported_arch(query.device)


def _flydsl_gdr_mtp_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
) -> bool:
    """Whether ``flydsl_gdr_mtp`` can serve this problem.

    The chain interface needs both halves of vLLM's contract: a 2-D
    ``[batch, token]`` slot map to checkpoint into, and the accepted-token count
    that says which of those slots to roll back to.
    """
    if ssm_state_indices is None or num_accepted_tokens is None:
        return False
    if ssm_state_indices.dim() != 2:
        return False
    if ssm_state_indices.shape[1] < query.shape[1]:
        return False
    if ssm_state_indices.dtype != torch.int32:
        return False
    if num_accepted_tokens.dtype != torch.int32:
        return False
    if not _unit_strided(ssm_state_indices) or not _unit_strided(num_accepted_tokens):
        return False
    return _mtp_shapes_supported(query, key, value, state)


def _flydsl_gdr_mtp_sglang_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    initial_state_indices: torch.Tensor | None,
    intermediate_states_buffer: torch.Tensor | None,
    intermediate_state_indices: torch.Tensor | None,
    retrieve_parent_token: torch.Tensor | None,
) -> bool:
    """Whether ``flydsl_gdr_mtp_sglang`` can serve this problem.

    The tree needs somewhere to read parents from, so a parent map without a
    snapshot buffer is refused rather than silently degraded to a chain.
    """
    if initial_state_indices is None or initial_state_indices.dim() != 1:
        return False
    if initial_state_indices.dtype != torch.int32:
        return False
    if not _unit_strided(initial_state_indices):
        return False
    if (intermediate_states_buffer is None) != (intermediate_state_indices is None):
        return False
    if intermediate_states_buffer is not None:
        if intermediate_states_buffer.dim() != 5:
            return False
        if intermediate_states_buffer.stride(-1) != 1:
            return False
        if intermediate_states_buffer.dtype not in _SUPPORTED_STATE_DTYPES:
            return False
        if intermediate_states_buffer.shape[1] < query.shape[1]:
            return False
        if intermediate_state_indices.dtype != torch.int32:
            return False
        if not _unit_strided(intermediate_state_indices):
            return False
    if retrieve_parent_token is not None:
        if intermediate_states_buffer is None:
            return False
        if retrieve_parent_token.dim() != 2:
            return False
        if retrieve_parent_token.dtype != torch.int32:
            return False
        if retrieve_parent_token.shape[1] < query.shape[1]:
            return False
        if not _unit_strided(retrieve_parent_token):
            return False
    return _mtp_shapes_supported(query, key, value, state)


def _mtp_common_checks(query, key, value, a, b, dt_bias, A_log, state, out):
    device = query.device
    dtype = query.dtype
    for t in (key, value, a, b, dt_bias, A_log, state, out):
        assert t.device == device, "every MTP operand must sit on one device"
    for t in (key, value, a, b, dt_bias, out):
        assert t.dtype == dtype
    assert state.dtype in _SUPPORTED_STATE_DTYPES
    assert A_log.dtype in (torch.float32, torch.bfloat16)
    assert state.data_ptr() % 16 == 0
    if query.stride(-1) != 1:
        raise ValueError(
            "`query` must have a contiguous last dimension for vectorized loads; "
            f"got stride {query.stride()}."
        )
    if key.stride(-1) != 1:
        raise ValueError(
            "`key` must have a contiguous last dimension for vectorized loads; "
            f"got stride {key.stride()}."
        )
    if state.stride(-1) != 1:
        raise ValueError(
            "`state` must be [pool, HV, V, K] with K contiguous; got stride "
            f"{state.stride()}. Shuffling it here would copy the whole pool, "
            "which costs more than the kernel it feeds."
        )


def _mtp_launch(
    *,
    mode,
    query,
    key,
    value,
    a,
    b,
    dt_bias,
    A_log,
    state,
    out,
    state_indices,
    num_accepted,
    inter_indices,
    parent_tokens,
    inter_buffer,
    use_qk_l2norm,
    has_tree,
    disable_state_update,
    stream,
):
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]

    kwargs_ = get_mtp_default_kwargs(
        str(query.dtype),
        str(state.dtype),
        state.dtype,
        batch_size,
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        query.device,
        _mtp_variant(mode, has_tree),
    )

    # Unused operands are handed an existing tensor rather than a null pointer:
    # the const_expr guards mean the kernel never builds a descriptor for them,
    # but the launch still needs a valid address in the slot.
    filler = state_indices
    inter_strides = tuple(inter_buffer.stride()) if inter_buffer is not None else ()
    parent_strides = tuple(parent_tokens.stride()) if parent_tokens is not None else ()

    exe = create_vk_gdr_mtp_kernel(
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        get_dtype_str(state.dtype),
        get_dtype_str(inter_buffer.dtype) if inter_buffer is not None else "f32",
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        query.stride(),
        key.stride(),
        value.stride(),
        state.stride(),
        a.stride(),
        b.stride(),
        tuple(state_indices.stride()) + ((1,) if state_indices.dim() == 1 else ()),
        inter_strides,
        parent_strides,
        use_qk_l2norm,
        mode,
        has_tree,
        disable_state_update,
        **kwargs_,
    )
    with torch.cuda.device(query.device.index):
        _run_compiled(
            exe,
            query,
            key,
            value,
            a,
            b,
            dt_bias.contiguous(),
            A_log.contiguous(),
            state_indices,
            num_accepted if num_accepted is not None else filler,
            inter_indices if inter_indices is not None else filler,
            parent_tokens if parent_tokens is not None else filler,
            state,
            inter_buffer if inter_buffer is not None else state,
            out,
            batch_size,
            stream,
        )


def flydsl_gdr_mtp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    use_qk_l2norm: bool = False,
    stream: torch.cuda.Stream = None,
):
    """Gated delta rule over a linear draft chain, vLLM's MTP contract.

    Rolls back to ``ssm_state_indices[n, num_accepted_tokens[n] - 1]`` and
    checkpoints each token into ``ssm_state_indices[n, t]``, so a later
    rejection has a slot per draft position to resume from. ``state`` is both
    the initial and the final store, as it is upstream.

    A negative slot is the skip sentinel, matching aiter's Triton kernel and
    SGLang's. vLLM's own kernel additionally reads slot 0 as null; callers who
    mean that must not hand slot 0 out as a live slot here.
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    _mtp_common_checks(query, key, value, a, b, dt_bias, A_log, state, out)
    assert ssm_state_indices.dtype == torch.int32
    assert num_accepted_tokens.dtype == torch.int32
    assert ssm_state_indices.dim() == 2, "the chain contract needs [batch, token]"
    assert ssm_state_indices.shape[0] == query.shape[0]
    assert ssm_state_indices.shape[1] >= query.shape[1]

    _mtp_launch(
        mode=MTP_MODE_CHAIN,
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        dt_bias=dt_bias,
        A_log=A_log,
        state=state,
        out=out,
        state_indices=ssm_state_indices.contiguous(),
        num_accepted=num_accepted_tokens.contiguous(),
        inter_indices=None,
        parent_tokens=None,
        inter_buffer=None,
        use_qk_l2norm=use_qk_l2norm,
        has_tree=False,
        disable_state_update=False,
        stream=stream,
    )


def flydsl_gdr_mtp_sglang(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    initial_state_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
    retrieve_parent_token: torch.Tensor | None = None,
    disable_state_update: bool = False,
    use_qk_l2norm: bool = False,
    stream: torch.cuda.Stream = None,
):
    """Gated delta rule over a draft tree, SGLang's MTP contract.

    The sequence keeps one pool slot, ``initial_state_indices[n]``, and the
    per-token record lives in ``intermediate_states_buffer`` instead. With
    ``retrieve_parent_token`` the draft is an EAGLE tree and each token restarts
    from its parent's snapshot; without it the tokens run in a chain, which is
    the same computation as a parent map of ``t - 1``. ``disable_state_update``
    leaves the pool untouched, which is what a verify pass wants.
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    _mtp_common_checks(query, key, value, a, b, dt_bias, A_log, state, out)
    assert initial_state_indices.dtype == torch.int32
    assert initial_state_indices.dim() == 1
    assert initial_state_indices.shape[0] == query.shape[0]
    if (intermediate_states_buffer is None) != (intermediate_state_indices is None):
        raise ValueError(
            "`intermediate_states_buffer` and `intermediate_state_indices` are "
            "one feature: pass both or neither."
        )
    if retrieve_parent_token is not None and intermediate_states_buffer is None:
        raise ValueError(
            "`retrieve_parent_token` needs `intermediate_states_buffer`: the tree "
            "restarts each token from a snapshot, so there has to be one."
        )
    if intermediate_states_buffer is not None:
        assert intermediate_states_buffer.dim() == 5, "[slot, step, HV, V, K]"
        assert intermediate_states_buffer.shape[1] >= query.shape[1]
        assert intermediate_states_buffer.shape[2] == value.shape[-2]
        assert intermediate_states_buffer.shape[3] == value.shape[-1]
        assert intermediate_states_buffer.shape[4] == query.shape[-1]
        if intermediate_states_buffer.stride(-1) != 1:
            raise ValueError(
                "`intermediate_states_buffer` must have K contiguous; got stride "
                f"{intermediate_states_buffer.stride()}."
            )
        assert intermediate_state_indices.dtype == torch.int32
    if retrieve_parent_token is not None:
        assert retrieve_parent_token.dtype == torch.int32
        assert retrieve_parent_token.dim() == 2
        assert retrieve_parent_token.shape[1] >= query.shape[1]

    _mtp_launch(
        mode=MTP_MODE_SNAPSHOT,
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        dt_bias=dt_bias,
        A_log=A_log,
        state=state,
        out=out,
        state_indices=initial_state_indices.contiguous(),
        num_accepted=None,
        inter_indices=(
            intermediate_state_indices.contiguous()
            if intermediate_state_indices is not None
            else None
        ),
        parent_tokens=(
            retrieve_parent_token.contiguous()
            if retrieve_parent_token is not None
            else None
        ),
        inter_buffer=intermediate_states_buffer,
        use_qk_l2norm=use_qk_l2norm,
        has_tree=retrieve_parent_token is not None,
        disable_state_update=disable_state_update,
        stream=stream,
    )
