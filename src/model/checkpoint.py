"""Restricted loading of the data/config objects used by SkinTokens checkpoints."""
import typing
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import (AnyNode, BooleanNode, BytesNode, EnumNode, FloatNode,
                            IntegerNode, PathNode, StringNode)


def load_checkpoint(path):
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 6):
        raise RuntimeError("Secure SkinTokens checkpoint loading requires PyTorch 2.6 or newer.")
    # Fixed inert data classes only. Never import globals suggested by a checkpoint.
    allowed = [DictConfig, ListConfig, ContainerMetadata, Metadata,
               AnyNode, BooleanNode, BytesNode, EnumNode, FloatNode, IntegerNode,
               PathNode, StringNode, typing.Any, dict, list, tuple, set, frozenset,
               str, int, float, bool, bytes, defaultdict, np.ndarray, np.dtype,
               np.core.multiarray._reconstruct, np.core.multiarray.scalar]
    allowed.extend([
        (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (np.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np.core.multiarray._reconstruct, "numpy._core.multiarray._reconstruct"),
        (np.core.multiarray.scalar, "numpy._core.multiarray.scalar"),
    ])
    allowed.extend({type(np.dtype(dtype)) for dtype in (
        bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
        np.uint32, np.uint64, np.float16, np.float32, np.float64,
        np.complex64, np.complex128, object, str, bytes)})
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)
