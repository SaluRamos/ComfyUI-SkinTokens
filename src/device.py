"""Native PyTorch CUDA/XPU dispatch; optional CUDA kernels stay CUDA-only."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import torch
import torch.nn.functional as F

_loading_device = ContextVar("skintokens_loading_device", default=None)

def resolve_device(device=None):
    if device is not None:
        result = torch.device(device)
    else:
        result = torch.device("cuda" if torch.cuda.is_available() else
                              "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    if result.type not in ("cpu", "cuda", "xpu"):
        raise ValueError(f"SkinTokens backend unsupported: {result}")
    if result.type != "cpu":
        backend = getattr(torch, result.type, None)
        if backend is None or not backend.is_available():
            raise RuntimeError(f"{result.type.upper()} unavailable; install the corresponding PyTorch build.")
    return result

@contextmanager
def loading_device(device):
    token = _loading_device.set(resolve_device(device))
    try:
        yield
    finally:
        _loading_device.reset(token)

def transformer_attention():
    from transformers.utils import is_flash_attn_2_available
    device = _loading_device.get()
    device = resolve_device() if device is None else device
    return "flash_attention_2" if device.type == "cuda" and is_flash_attn_2_available() else "sdpa"

def model_autocast(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        device = next(self.parameters()).device
        context = torch.autocast(device.type, dtype=torch.bfloat16) if device.type in ("cuda", "xpu") else nullcontext()
        with context:
            return function(self, *args, **kwargs)
    return wrapped

def packed_attention(q, k, v):
    """B,L,H,D -> B,L,H,D, with the tuple interface used by these models."""
    if q.device.type == "cuda":
        try:
            from flash_attn_interface import flash_attn_func
        except ImportError:
            try:
                from flash_attn.flash_attn_interface import flash_attn_func
            except ImportError:
                flash_attn_func = None
        if flash_attn_func is not None:
            result = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous())
            return result if isinstance(result, tuple) else (result, None)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if q.shape[1] != k.shape[1]:
        if q.shape[1] % k.shape[1]:
            raise ValueError("Query heads must be divisible by key/value heads.")
        repeat = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
    output = F.scaled_dot_product_attention(q, k, v)
    return output.transpose(1, 2).contiguous(), None
