"""Data-only transport for mesh assets and tensor packets."""
import io
import json
import zipfile
import base64
from dataclasses import fields
from pathlib import Path

import numpy as np
from ..numpy_io import read_array_header

MAX_PACKET_BYTES = 2 * 1024 ** 3


def encode_packet(value, types):
    arrays = []

    def encode(item):
        if item is None or type(item) in (bool, int, float, str):
            return item
        if type(item) is bytes:
            return {"type": "bytes", "value": base64.b64encode(item).decode("ascii")}
        if isinstance(item, np.generic):
            return encode(item.item())
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                return {"type": "object_array", "shape": list(item.shape),
                        "dtype": encode(np.lib.format.dtype_to_descr(item.dtype)),
                        "items": [encode(value) for value in item.flat]}
            index = len(arrays)
            arrays.append(item)
            return {"type": "array", "index": index}
        if type(item) in types.values():
            return {"type": type(item).__name__, "fields": {
                field.name: encode(getattr(item, field.name)) for field in fields(item)}}
        if isinstance(item, Path):
            return {"type": "path", "value": str(item)}
        if type(item) in (list, tuple):
            return {"type": "tuple" if type(item) is tuple else "list",
                    "items": [encode(x) for x in item]}
        if type(item) is dict:
            return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in item.items()]}
        # Torch is optional in the Blender Python environment.
        if type(item).__module__ == "torch" and type(item).__name__ == "Tensor":
            tensor = item.detach().cpu()
            dtype = str(tensor.dtype).removeprefix("torch.")
            if dtype == "bfloat16":
                tensor = tensor.float()
            return {"type": "tensor", "dtype": dtype, "array": encode(tensor.numpy())}
        raise TypeError(f"Unsupported packet value: {type(item).__name__}")

    metadata = json.dumps(encode(value), ensure_ascii=True).encode("utf-8")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("packet.json", metadata)
        for index, array in enumerate(arrays):
            with archive.open(f"arrays/{index}.npy", "w", force_zip64=True) as target:
                np.save(target, array, allow_pickle=False)
    if stream.tell() > MAX_PACKET_BYTES:
        raise ValueError("SkinTokens packet exceeds 2 GiB")
    return stream.getvalue()


def decode_packet(data, types, map_location=None):
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("SkinTokens packet exceeds 2 GiB")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 100000 or len({x.filename for x in entries}) != len(entries):
            raise ValueError("Invalid packet archive")
        if sum(x.file_size for x in entries) > MAX_PACKET_BYTES:
            raise ValueError("SkinTokens packet exceeds 2 GiB")
        if archive.getinfo("packet.json").file_size > 64 * 1024 ** 2:
            raise ValueError("Packet metadata is too large")

        def decode(item):
            if item is None or type(item) in (bool, int, float, str):
                return item
            if type(item) is not dict:
                raise ValueError("Invalid packet value")
            kind = item.get("type")
            if kind == "bytes":
                return base64.b64decode(item["value"], validate=True)
            if kind == "array":
                index = item["index"]
                if type(index) is not int or index < 0:
                    raise ValueError("Invalid array index")
                with archive.open(f"arrays/{index}.npy") as source:
                    shape, order, dtype = read_array_header(source)
                    count = 1
                    for dimension in shape:
                        if dimension < 0:
                            raise ValueError("Invalid array shape")
                        count *= dimension
                    if dtype.hasobject or count * dtype.itemsize > archive.getinfo(f"arrays/{index}.npy").file_size - source.tell():
                        raise ValueError("Invalid array size or dtype")
                with archive.open(f"arrays/{index}.npy") as source:
                    return np.load(source, allow_pickle=False)
            if kind == "object_array":
                shape = item["shape"]
                count = 1
                for dimension in shape:
                    if type(dimension) is not int or dimension < 0:
                        raise ValueError("Invalid object array shape")
                    count *= dimension
                dtype = np.dtype(decode(item["dtype"]))
                if count != len(item["items"]) or not dtype.hasobject or count * dtype.itemsize > MAX_PACKET_BYTES:
                    raise ValueError("Invalid object array size or dtype")
                array = np.empty(count, dtype=dtype)
                for index, value in enumerate(item["items"]):
                    array[index] = decode(value)
                return array.reshape(shape)
            if kind in types:
                values = item["fields"]
                allowed = {field.name for field in fields(types[kind])}
                if not isinstance(values, dict) or not set(values) <= allowed:
                    raise ValueError("Invalid dataclass fields")
                return types[kind](**{key: decode(value) for key, value in values.items()})
            if kind == "path":
                return Path(item["value"])
            if kind in ("list", "tuple"):
                values = [decode(value) for value in item["items"]]
                return tuple(values) if kind == "tuple" else values
            if kind == "dict":
                return {decode(key): decode(value) for key, value in item["items"]}
            if kind == "tensor":
                import torch
                dtypes = {str(dtype).removeprefix("torch."): dtype for dtype in (
                    torch.float16, torch.float32, torch.float64, torch.bfloat16,
                    torch.int8, torch.int16, torch.int32, torch.int64,
                    torch.uint8, torch.bool, torch.complex64, torch.complex128)}
                array = decode(item["array"])
                if not isinstance(array, np.ndarray) or item["dtype"] not in dtypes:
                    raise ValueError("Invalid tensor")
                return torch.from_numpy(array).to(device=map_location or "cpu", dtype=dtypes[item["dtype"]])
            raise ValueError("Unknown packet type")

        return decode(json.loads(archive.read("packet.json")))
