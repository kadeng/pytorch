import functools
import logging
import math

import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import (
    filter_nodes,
    inference_graph,
    register_replacement,
    training_graph,
    ReplacementPatternEntry,
    create_replacement_pattern_from_functions
)
from collections import namedtuple
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)
aten = torch.ops.aten


def _sfdp_pattern_1(query, key, value, inv_scale):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale)
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_1(query, key, value, inv_scale):
    counters["inductor"]["fuse_attention"] += 1
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_2(query, key, value, scale_factor):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .mul(scale_factor)
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_replacement_2(query, key, value, scale_factor):
    counters["inductor"]["fuse_attention"] += 1
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale_factor,
    )


def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_3(query, key, value, inv_scale_factor, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )


def _sfdp_pattern_4(query, key, value, scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_replacement_4(query, key, value, scale_factor, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale_factor,
    )


def _sfdp_pattern_5(query, key, value, attn_mask):
    attn_weight = torch.softmax(
        (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1
    )
    # attn_weight = torch.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def _sfdp_replacement_5(query, key, value, attn_mask):
    counters["inductor"]["fuse_attention"] += 1
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=0.0,
        is_causal=False,
    )


def _sfdp_pattern_6(query, key, value, attn_mask, dropout_p):
    attn_weight = torch.softmax(
        (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1
    )
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value


def _sfdp_replacement_6(query, key, value, attn_mask, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    return aten.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=dropout_p,
        is_causal=False,
    )


def _sfdp_pattern_7(query, key, value, dropout_p):
    # in real workloads inputs to matmul are permuted
    # causing matmul to expand to a series of expand and clone calls
    # we want the same to happen during pattern tracing
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v


def _sfdp_replacement_7(query, key, value, dropout_p):
    # sdpa prefers inputs in permuted format
    # it makes a copy to put them in this format
    # if they aren't already
    # to make replacement efficient ensure that inputs to sdpa
    # are in required order
    counters["inductor"]["fuse_attention"] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )


def _sfdp_pattern_8(query, key, value):
    # no dropout version of pattern 7
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v


def _sfdp_replacement_8(query, key, value):
    counters["inductor"]["fuse_attention"] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )


def _sfdp_pattern_9(query, key, value, dropout_p):
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    q = q / math.sqrt(q.size(-1))
    div = q @ k.transpose(-2, -1)
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v


def _sfdp_replacement_9(query, key, value, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )


def _sfdp_pattern_10(query, key, value):
    # no dropout version of 9
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    q = q / math.sqrt(q.size(-1))
    div = q @ k.transpose(-2, -1)
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v


def _sfdp_replacement_10(query, key, value):
    counters["inductor"]["fuse_attention"] += 1
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return aten.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )

def _sfdp_pattern_11(
                       query : torch.Tensor,
                       key : torch.Tensor,
                       value : torch.Tensor,
                       causal_mask : torch.Tensor,
                       # scale : float,
                       dropout_p : float):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    attn_weights = attn_weights / torch.full(
        [], 1.0, dtype=attn_weights.dtype, device=attn_weights.device
    )
    mask_value = torch.finfo(attn_weights.dtype).min
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, dropout_p, True, False)

    attn_output = torch.matmul(attn_weights, value)

    return attn_output

def _sfdp_replacement_11(
                       query : torch.Tensor,
                       key : torch.Tensor,
                       value : torch.Tensor,
                       causal_mask : torch.Tensor,
                       # scale : float,
                       dropout_p : float):
    counters["inductor"]["fuse_attention"] += 1
    return aten.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=causal_mask.to(torch.bool),
        dropout_p=dropout_p,
        is_causal=False,
    )
    return attn_output


def _sfdp_params_check(match):
    assert all(k in match.kwargs for k in ("query", "key", "value"))
    query = match.kwargs["query"].meta["val"]
    key = match.kwargs["key"].meta["val"]
    value = match.kwargs["value"].meta["val"]
    if not (query.dtype == key.dtype == value.dtype) or not (
        query.device == key.device == value.device
    ):
        return False
    add_mask_node = filter_nodes(match.nodes, aten.add.Tensor)
    # Has attn_mask add.
    if len(add_mask_node) > 0:
        attn_mask_node = add_mask_node[0].args[1]
        # attn_mask_node may be a float/int number.
        if not hasattr(attn_mask_node, "meta"):
            return False
        attn_mask = attn_mask_node.meta["val"]
        # Make sure attn_mask.dtype == query.dtype or attn_mask.dtype == torch.bool
        if (
            not isinstance(attn_mask, torch.Tensor)
            or not (attn_mask.dtype == query.dtype or attn_mask.dtype == torch.bool)
            or query.device != attn_mask.device
        ):
            return False
    return True


def _sfdp_scale_factor_check(scale_factor_op):
    def fn(match):
        scale_factor_node = filter_nodes(match.nodes, scale_factor_op)[0]
        # Note: args[1] of the scale_factor_node is always the scale_factor for the current patterns.
        scale_factor = scale_factor_node.args[1]
        # make sure the scale_factor a float/int. SymInt?
        if not isinstance(scale_factor, (float, int)):
            return False
        return _sfdp_params_check(match)

    return fn

ReplacementInfo = namedtuple('ReplacementInfo', ['pattern', 'replacement', 'args', 'workaround', 'extra_check'])

def _create_sfdp_replacement_patterns() -> List[ReplacementInfo]:
    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"

    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    g = functools.partial(torch.empty, (2, 4, 8, 16), device=device, requires_grad=True)
    gp = functools.partial(
        torch.empty, (2, 8, 4, 16), device=device, requires_grad=True, dtype=torch.half
    )
    b = functools.partial(torch.empty, (1, 1, 8, 8), device=device)
    c = functools.partial(torch.tensor, 2.0, device=device)
    pdrop = functools.partial(torch.tensor, 0.1, device=device)
    bool_scalar = lambda: True
    mask_tensor = functools.partial(
        torch.empty, (1, 1, 8, 8), device=device, requires_grad=False, dtype=torch.bool
    )
    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    d = {"dropout_p": 0.113377}
    # same here, except we need another magic value for scale
    ds = {"scale": 1.113377, "dropout_p": 0.113377}

    return [
        ReplacementInfo(
                _sfdp_pattern_1,
                _sfdp_replacement_1,
                [g(), g(), g(), c()],
                {},
                _sfdp_scale_factor_check(aten.div.Tensor),
        ),
        ReplacementInfo(
                _sfdp_pattern_2,
                _sfdp_replacement_2,
                [g(), g(), g(), c()],
                {},
                _sfdp_scale_factor_check(aten.mul.Tensor),
        ),
        ReplacementInfo(
                _sfdp_pattern_3,
                _sfdp_replacement_3,
                [g(), g(), g(), c()],
                d,
                _sfdp_scale_factor_check(aten.div.Tensor),
        ),
        ReplacementInfo(
                _sfdp_pattern_4,
                _sfdp_replacement_4,
                [g(), g(), g(), c()],
                d,
                _sfdp_scale_factor_check(aten.mul.Tensor),
        ),
        ReplacementInfo(
                _sfdp_pattern_5,
                _sfdp_replacement_5,
                [g(), g(), g(), b()],
                {},
                _sfdp_params_check,
        ),
        ReplacementInfo(
                _sfdp_pattern_6,
                _sfdp_replacement_6,
                [g(), g(), g(), b()],
                d,
                _sfdp_params_check,
        ),
        ReplacementInfo(
                _sfdp_pattern_7,
                _sfdp_replacement_7,
                [gp(), gp(), gp()],
                d,
                _sfdp_params_check,
        ),
        ReplacementInfo(
                _sfdp_pattern_8,
                _sfdp_replacement_8,
                [gp(), gp(), gp()],
                {},
                _sfdp_params_check,
        ),
        ReplacementInfo(
                _sfdp_pattern_9,
                _sfdp_replacement_9,
                [gp(), gp(), gp()],
                d,
                _sfdp_params_check,
        ),
        ReplacementInfo(
                _sfdp_pattern_10,
                _sfdp_replacement_10,
                [gp(), gp(), gp()],
                {},
                _sfdp_params_check,
        ),
        ReplacementInfo(
                _sfdp_pattern_11,
                _sfdp_replacement_11,
                [g(), g(), g(), mask_tensor()],
                d,
                _sfdp_params_check,
        ),
    ]

def replacement_pattern_from_replacement_info(rinfo : ReplacementInfo, inference : bool = False ) -> ReplacementPatternEntry:
    if inference:
        trace_fn = inference_graph
    else:
        trace_fn = training_graph
    pattern, replacement, non_workaround_args, workaround, extra_check = rinfo
    args = [*non_workaround_args, *workaround.values()]
    return create_replacement_pattern_from_functions(
        pattern,
        replacement,
        args,
        trace_fn,
        extra_check=extra_check,
        scalar_workaround=workaround,
    )

@functools.lru_cache(None)
def _sfdp_init():
    from ..._dynamo.utils import counters

    counters_ref = counters["inductor"].copy()
    from .joint_graph import patterns

    for pattern, replacement, args, workaround, extra_check in _create_sfdp_replacement_patterns():
        args = [*args, *workaround.values()]

        register_replacement(
            pattern,
            replacement,
            args,
            training_graph,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )
        register_replacement(
            pattern,
            replacement,
            args,
            inference_graph,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )

    counters[
        "inductor"
    ] = counters_ref  # clear view matches encountered during sdpa tracing
