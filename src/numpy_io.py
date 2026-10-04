import ast
import struct

import numpy as np


def read_array_header(source):
    version = np.lib.format.read_magic(source)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(source)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(source)
    if version != (3, 0):
        raise ValueError("Unsupported NumPy array format")
    length = struct.unpack("<I", source.read(4))[0]
    if length > 10000:
        raise ValueError("NumPy array header is too large")
    header = ast.literal_eval(source.read(length).decode("utf-8"))
    if not isinstance(header, dict) or set(header) != {"shape", "fortran_order", "descr"}:
        raise ValueError("Invalid NumPy array header")
    if not isinstance(header["shape"], tuple) or any(type(x) is not int or x < 0 for x in header["shape"]):
        raise ValueError("Invalid NumPy array shape")
    if type(header["fortran_order"]) is not bool:
        raise ValueError("Invalid NumPy array order")
    return header["shape"], header["fortran_order"], np.lib.format.descr_to_dtype(header["descr"])
