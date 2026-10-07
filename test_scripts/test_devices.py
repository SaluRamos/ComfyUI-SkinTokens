"""Dispatch tests run without model weights or a GPU; numerical tests use PyTorch when available."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]

def device(value):
    if hasattr(value, 'type'): return value
    kind, _, index = value.partition(':')
    return types.SimpleNamespace(type=kind, index=int(index) if index else None)

class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.torch = types.ModuleType('torch')
        self.torch.device = device
        self.torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.torch.xpu = types.SimpleNamespace(is_available=lambda: True)
        self.torch.bfloat16 = 'bfloat16'
        self.torch.autocast = Mock(side_effect=lambda *a, **k: nullcontext())
        nn = types.ModuleType('torch.nn')
        functional = types.ModuleType('torch.nn.functional')
        self.torch.nn = nn
        nn.functional = functional
        self.functional = functional
        utils = types.ModuleType('transformers.utils')
        utils.is_flash_attn_2_available = lambda: True
        spec = importlib.util.spec_from_file_location('skin_device', ROOT / 'src/device.py')
        self.support = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': self.torch, 'torch.nn': nn, 'torch.nn.functional': functional, 'transformers.utils': utils}):
            spec.loader.exec_module(self.support)
        self.utils = utils

    def test_auto_selects_intel_without_cuda(self):
        self.assertEqual(self.support.resolve_device().type, 'xpu')
        self.assertEqual(self.support.resolve_device('xpu:1').index, 1)
        with self.assertRaises(RuntimeError): self.support.resolve_device('cuda')

    def test_transformer_sdpa_even_if_cuda_flash_package_is_installed(self):
        with patch.dict(sys.modules, {'transformers.utils': self.utils}):
            with self.support.loading_device('xpu'):
                self.assertEqual(self.support.transformer_attention(), 'sdpa')
                with self.support.loading_device('cpu'):
                    self.assertEqual(self.support.transformer_attention(), 'sdpa')
                self.assertEqual(self.support._loading_device.get().type, 'xpu')
            self.assertIsNone(self.support._loading_device.get())

    def test_model_autocast_uses_actual_parameter_device(self):
        model = types.SimpleNamespace(parameters=lambda: iter([types.SimpleNamespace(device=device('xpu'))]))
        method = self.support.model_autocast(lambda self: 'result')
        self.assertEqual(method(model), 'result')
        self.torch.autocast.assert_called_once_with('xpu', dtype='bfloat16')
        model.parameters = lambda: iter([types.SimpleNamespace(device=device('cpu'))])
        method(model)
        self.assertEqual(self.torch.autocast.call_count, 1)

    def test_intel_attention_never_calls_installed_cuda_kernel(self):
        class Tensor:
            device = device('xpu')
            shape = (1, 2, 3, 4)
            def transpose(self, *args): return self
            def contiguous(self): return self
        t = Tensor()
        self.functional.scaled_dot_product_attention = Mock(return_value=t)
        cuda_module = types.ModuleType('flash_attn_interface')
        cuda_module.flash_attn_func = Mock(side_effect=AssertionError('CUDA kernel called'))
        with patch.dict(sys.modules, {'flash_attn_interface': cuda_module}):
            self.assertEqual(self.support.packed_attention(t, t, t), (t, None))
        cuda_module.flash_attn_func.assert_not_called()
        self.functional.scaled_dot_product_attention.assert_called_once()

try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, 'PyTorch not installed')
class AttentionNumericalTests(unittest.TestCase):
    def test_gqa_matches_explicit_attention(self):
        spec = importlib.util.spec_from_file_location('numerical_skin_device', ROOT / 'src/device.py')
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        devices = ['cpu'] + (['xpu'] if hasattr(torch, 'xpu') and torch.xpu.is_available() else [])
        for target in devices:
            with self.subTest(device=target):
                torch.manual_seed(13)
                q = torch.randn(2, 5, 4, 8, device=target)
                k, v = (torch.randn(2, 7, 2, 8, device=target) for _ in range(2))
                actual = support.packed_attention(q, k, v)[0]
                qh = q.transpose(1, 2)
                kh, vh = (t.transpose(1, 2).repeat_interleave(2, dim=1) for t in (k, v))
                reference = ((qh @ kh.transpose(-2, -1) / 8**0.5).softmax(-1) @ vh).transpose(1, 2)
                torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)

if __name__ == '__main__':
    unittest.main()
