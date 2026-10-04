"""Read legacy NumPy object arrays without accepting arbitrary Python globals."""
import io
import pickle
import zipfile

import numpy as np
from ..numpy_io import read_array_header


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np.core.multiarray, name)
        raise pickle.UnpicklingError(f"Unsupported NPZ global: {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent NPZ objects are not supported")


def load_npz(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 2 * 1024 ** 3:
            raise ValueError("NPZ exceeds 2 GiB")
        for entry in archive.infolist():
            if not entry.filename.endswith(".npy"):
                continue
            data = archive.read(entry)
            stream = io.BytesIO(data)
            shape, fortran, dtype = read_array_header(stream)
            count = 1
            for dimension in shape:
                if dimension < 0:
                    raise ValueError("Invalid NPZ shape")
                count *= dimension
            if count * dtype.itemsize > 2 * 1024 ** 3:
                raise ValueError("NPZ array exceeds 2 GiB")
            if dtype.hasobject:
                array = ArrayUnpickler(stream).load()
                if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != dtype:
                    raise ValueError("NPZ array does not match its header")
            else:
                if count * dtype.itemsize > len(data) - stream.tell():
                    raise ValueError("Invalid NPZ array size")
                array = np.load(io.BytesIO(data), allow_pickle=False)
            result[entry.filename[:-4]] = array
    return result
