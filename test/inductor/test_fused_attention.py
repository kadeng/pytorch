# Owner(s): ["module: inductor"]
import itertools
from typing import Callable, List

import math

import torch
import torch._inductor.config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._dynamo.variables.param_decorators import PlaceholderAsScalar
from torch._inductor import config
import torch._inductor.fx_passes.fuse_attention
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_SDPA,
    SM80OrLater,
)
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CUDA
import unittest.mock as mock
import logging

@config.patch(fallback_random=True)
class TestSDPAPatternRewriter(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _check_common(
        self,
        dot_prod_attention,
        args1=None,
        contains=True,
        atol=1e-5,
        has_fuse_pattern=True,
        has_dropout=False,
    ):
        if args1 is None:
            tensor_shape = (4, 2, 16, 32)
            args1 = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
            ]
        args2 = self._clone_inputs(args1)

        for training in [False, True]:
            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training

            torch.manual_seed(1234)
            result1 = dot_prod_attention(*args1)

            counters.clear()
            torch.manual_seed(1234)
            result2, (source_code,) = run_and_get_code(
                torch.compile(dot_prod_attention, fullgraph=True), *args2
            )
            if has_fuse_pattern:
                self.assertGreaterEqual(counters["inductor"]["fuse_attention"], 1)
            if contains:
                # many of the patterns get re-expanded in dispatcher
                self.assertIn(
                    "aten._scaled_dot_product",
                    source_code,
                )
            if not has_dropout:
                self.assertEqual(result1, result2, atol=atol, rtol=1.3e-6)

            if training:
                result1.sum().backward()
                result2.sum().backward()
                for arg1, arg2 in zip(args1, args2):
                    if (
                        isinstance(arg1, torch.Tensor)
                        and arg1.is_floating_point()
                        and not has_dropout
                    ):
                        self.assertEqual(arg1.grad, arg2.grad, atol=atol, rtol=1.3e-6)

    def _capture_joint_graph(self, fn, args1, training=True) -> torch.fx.GraphModule:
        """
        Capture the fx graph of a function as it passes through joint_graph.joint_graph_passes

        :param fn: The function to capture the graph of
        :param args1: The first set of arguments for the function
        :param training: Indicates whether training mode is enabled or not (default: True)

        :return: The captured graph of the function ( a torch.fx.Graph )
        """
        import torch._inductor.fx_passes
        import torch._inductor.fx_passes.joint_graph
        import torch._inductor.fx_passes.fuse_attention
        old_pattern_matcher_cfg = torch._inductor.config.pattern_matcher
        try:
            torch._inductor.config.pattern_matcher = False # we don't want the pattern matcher to run
            joint_graph_passes = torch._inductor.fx_passes.joint_graph.joint_graph_passes
            captured_graph = None

            def joint_graph_passes_patched(graph: torch.fx.GraphModule):
                nonlocal captured_graph
                res = joint_graph_passes(graph)
                captured_graph = res
                return res

            args2 = self._clone_inputs(args1)

            for x in itertools.chain(args1[:], args2[:]):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    x.requires_grad = training


            torch.manual_seed(1234)
            with mock.patch('torch._inductor.fx_passes.joint_graph.joint_graph_passes', joint_graph_passes_patched):
                result2, (source_code,) = run_and_get_code(
                    torch.compile(fn, fullgraph=True), *args2
                )
            counters.clear()
            assert captured_graph is not None, "No graph was captured"
            return captured_graph
        finally:
            torch._inductor.config.pattern_matcher = old_pattern_matcher_cfg

    @skipIfRocm
    def test_sdpa_rewriter_1(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        self._check_common(dot_prod_attention)

    def test_pattern_fails_with_reuse(self):
        """
        This test checks that the replacement is not done
        when an intermediate result is being used / returned downstream
        """

        @skipIfRocm
        @torch.compile(fullgraph=True)
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            attn_weights = (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
            )
            return attn_weights.matmul(value), attn_weights

        tensor_shape = (2, 4, 8, 16)
        args = [
            torch.randn(tensor_shape, device="cuda"),
            torch.randn(tensor_shape, device="cuda"),
            torch.randn(tensor_shape, device="cuda"),
        ]
        _, (source_code,) = run_and_get_code(dot_prod_attention, *args)
        self.assertNotIn("aten._scaled_dot_product_efficient_attention", source_code)

    @skipIfRocm
    def test_sdpa_rewriter_2(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        self._check_common(dot_prod_attention)

    def test_sdpa_rewriter_3(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).div(3.0).softmax(dim=-1),
                p=0.4,
                training=True,
                inplace=False,
            ).matmul(value)

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)

    def test_sdpa_rewriter_4(self):
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.dropout(
                torch.matmul(query, key.transpose(-2, -1)).mul(0.4).softmax(dim=-1),
                p=0.2,
                training=True,
                inplace=False,
            ).matmul(value)

        self._check_common(dot_prod_attention, contains=False, has_dropout=True)

    def test_sdpa_rewriter_5(self):
        def sfdp_pattern_5_v1(query, key, value):
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        def sfdp_pattern_5_v2(query, key, value):
            # https://github.com/pytorch/pytorch/issues/100318.
            attn_mask = torch.zeros(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).bool()
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            return attn_weight @ value

        self._check_common(sfdp_pattern_5_v1, contains=False)
        self._check_common(sfdp_pattern_5_v2, contains=False)

    @skipIfRocm
    def test_sdpa_rewriter_6(self):
        def sfdp_pattern_6(query, key, value):
            attn_mask = torch.ones(
                query.size(-2), key.size(-2), dtype=torch.bool, device=query.device
            ).tril(diagonal=0)
            attn_mask = attn_mask.masked_fill(
                torch.logical_not(attn_mask), -float("inf")
            )
            attn_weight = torch.softmax(
                (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask,
                dim=-1,
            )
            attn_weight = torch.dropout(attn_weight, 0.5, True)
            return attn_weight @ value

        self._check_common(sfdp_pattern_6, contains=False, has_dropout=True)

    @skipIfRocm
    def test_sdpa_rewriter_7(self):
        def sfdp_pattern_7(query, key, value):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # very small dropout to make sure test passes
            attn_weight = torch.dropout(attn_weight, 0.0000, True)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )

        self._check_common(sfdp_pattern_7, args, contains=SM80OrLater, atol=2e-3)

    @skipIfRocm
    def test_sdpa_rewriter_8(self):
        def sfdp_pattern_8(query, key, value):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )

        self._check_common(sfdp_pattern_8, args, atol=2e-3)

    @skipIfRocm
    def test_sdpa_rewriter_9(self):
        def sfdp_pattern_9(query, key, value):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / math.sqrt(q.size(-1))
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            # very low dropout to make test pass
            attn_weight = torch.dropout(attn_weight, 0.9999, True)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )
        self._check_common(sfdp_pattern_9, args, contains=SM80OrLater, atol=2e-3)

    @skipIfRocm
    def test_sdpa_rewriter_10(self):
        def sfdp_pattern_10(query, key, value):
            q = query.permute(0, 2, 1, 3)
            k = key.permute(0, 2, 1, 3)
            v = value.permute(0, 2, 1, 3)
            q = q / math.sqrt(q.size(-1))
            div = q @ k.transpose(-2, -1)
            div = div.to(torch.float32)
            attn_weight = torch.softmax(div, dim=-1)
            attn_weight = attn_weight.to(torch.float16)
            return attn_weight @ v

        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.half),
        )

        self._check_common(sfdp_pattern_10, args, atol=2e-3)

    def test_sdpa_rewriter_11_a(self):
        def sfdp_pattern_11(
                query,
                key,
                value,
                causal_mask,
                scale,
                dropout_p):
            attn_weights = torch.matmul(query, key.transpose(-1, -2))

            attn_weights = attn_weights / torch.full(
                [], scale, dtype=attn_weights.dtype, device=attn_weights.device
            )
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

            attn_weights = attn_weights.type(value.dtype)
            attn_weights = torch.nn.functional.dropout(attn_weights, dropout_p, True, False)

            attn_output = torch.matmul(attn_weights, value)

            return attn_output

        args = (
            torch.randn((2, 8, 4, 4), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 4), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 4), device="cuda", dtype=torch.half),
            torch.randn((1,1,4,4), device="cuda").to(torch.bool),
            2.0,
            0.5
        )

        self._check_common(sfdp_pattern_11, args, atol=2e-3, has_dropout=True)

    def test_something(self):
        import torch._dynamo.config as dynamo_config
        args = (
            torch.randn((2, 8, 4, 4), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 4), device="cuda", dtype=torch.half),
            torch.randn((2, 8, 4, 4), device="cuda", dtype=torch.half),
            torch.randn((1, 1, 4, 4), device="cuda").to(torch.bool),
            2.0,
            0.5
        )
        new_allowlist = set(dynamo_config.skipfiles_inline_module_allowlist)
        new_allowlist.add(torch._inductor.fx_passes.fuse_attention)
        with dynamo_config.patch(skipfiles_inline_module_allowlist=new_allowlist):
            g3 = torch._dynamo.export(torch._inductor.fx_passes.fuse_attention._sfdp_replacement_11, aten_graph=True, pre_dispatch=False, tracing_mode="real")(*args)
            g3[0].graph.print_tabular()
            assert False, ".."

    def test_sdpa_rewriter_11(self):

        # pattern should match huggingface GPT2Attention in certain configs
        def sfdp_pattern_11(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                causal_mask: torch.Tensor,
                scale:  PlaceholderAsScalar(float),
                dropout_p : PlaceholderAsScalar(float)):
            attn_weights = torch.matmul(query, key.transpose(-1, -2))

            attn_weights = attn_weights / torch.full(
                [], scale, dtype=attn_weights.dtype, device=attn_weights.device
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

        args = (
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.float32),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.float32),
            torch.randn((2, 8, 4, 16), device="cuda", dtype=torch.float32),
            torch.randn((1, 4, 4), device="cuda", dtype=torch.float32)>0.0,
            torch.scalar_tensor(2.0, device="cuda", dtype=torch.float32),
            torch.scalar_tensor(0.1, device="cuda", dtype=torch.float32),
        )
        #ris: List[ReplacementInfo] = _create_sfdp_replacement_patterns()

        #graph = self._capture_joint_graph(sfdp_pattern_11, args, training=True)

        #pmpass = PatternMatcherPass()
        #repentry : ReplacementPatternEntry  = replacement_pattern_from_replacement_info(ris[-1], False)
        #from torch._inductor.pattern_matcher import log
        #repentry.register(pmpass)
        #pmpass.apply(graph)
        res, g = torch._dynamo.export(sfdp_pattern_11, *args)
        res.graph.print_tabular()
        assert False, "stop here"


    def test_pattern_fails_with_tensor_factor(self):
        # https://github.com/pytorch/pytorch/issues/99124
        class Model(torch.nn.Module):
            def __init__(self, is_inv_factor):
                super().__init__()
                self.is_inv_factor = is_inv_factor

            def forward(self, query, key, value, scale_factor) -> torch.Tensor:
                y = torch.matmul(query, key.transpose(-2, -1))
                if self.is_inv_factor:
                    y = y.div(scale_factor)
                else:
                    y = y.mul(scale_factor)
                return y.softmax(dim=-1).matmul(value)

        tensor_shape = (2, 4, 4, 4)
        for is_inv_factor in [True, False]:
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn((4, 1, 1), device="cuda"),
            ]
            model = Model(is_inv_factor).eval()
            # The training path has an accuracy gap compared with eager mode.
            self._check_common(
                model, args1=args, contains=False, atol=1e-4, has_fuse_pattern=False
            )

    def test_pattern_fails_with_unsupported_mask(self):
        # https://github.com/pytorch/pytorch/issues/100315
        class Model(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()

            def forward(self, query, key, value, attn_mask) -> torch.Tensor:
                attn_weight = torch.softmax(
                    query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
                    + attn_mask,
                    dim=-1,
                )
                return attn_weight @ value

        tensor_shape = (2, 4, 4, 4)

        upsupported_masks = [
            torch.randn((2, 4, 4, 4), device="cuda").to(dtype=torch.int),
            2.0,
        ]
        for atte_mask in upsupported_masks:
            args = [
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                torch.randn(tensor_shape, device="cuda"),
                atte_mask,
            ]
            model = Model().eval()
            # The training path has an accuracy gap compared with eager mode.
            self._check_common(
                model, args1=args, contains=False, atol=1e-4, has_fuse_pattern=False
            )


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA and PLATFORM_SUPPORTS_FUSED_SDPA:
        run_tests()
