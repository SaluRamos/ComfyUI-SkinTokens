"""Run with: python -m unittest discover -s test_scripts -p test_security.py -v."""
import hashlib
import ast
import builtins
import io
import json
import os
import pickle
import queue
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from wsgiref.simple_server import make_server, WSGIRequestHandler

import numpy as np
import requests
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.server.spec import Asset, TensorPacket, bytes_to_object, object_to_bytes
from src.server.transport import (TOKEN_ENV, PAYLOAD_DIR_ENV, prepare_server_session,
                                  payload_directory, resolve_payload_path, server_request)
from src.data.safe_npz import load_npz
from src.model.checkpoint import load_checkpoint


class Malicious:
    def __reduce__(self):
        return os.system, ("echo SHOULD_NOT_EXECUTE",)


class QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):
        pass


class SecurityTests(unittest.TestCase):
    def test_asset_roundtrip(self):
        values = dict(vertices=np.arange(36, dtype=np.float32).reshape(12, 3),
                      faces=np.array([[0, 1, 2]], dtype=np.int64),
                      matrix_basis=np.arange(32, dtype=np.float64).reshape(2, 1, 4, 4),
                      skin=np.ones((12, 1), dtype=np.float32),
                      joint_names=["mão"], mesh_names=["body"], path="example.fbx",
                      parents=np.array([-1], dtype=np.int32),
                      vertex_groups={"body": np.array([0, 1, 2])},
                      meta={"tuple": (1, None, True), "bytes": b"abc", "path": Path("a/b"),
                            "object": np.array([None, "name"], dtype=object)})
        restored = bytes_to_object(object_to_bytes(Asset(**values)))
        self.assertIsInstance(restored, Asset)
        for key, value in values.items():
            actual = getattr(restored, key)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual, value)
                self.assertEqual(actual.dtype, value.dtype)
            elif key != "meta" and key != "vertex_groups":
                self.assertEqual(actual, value)
        self.assertEqual(restored.meta["tuple"], (1, None, True))
        self.assertEqual(restored.meta["bytes"], b"abc")
        self.assertEqual(restored.meta["path"], Path("a/b"))
        np.testing.assert_array_equal(restored.meta["object"], values["meta"]["object"])

    def test_tensor_packet(self):
        packet = TensorPacket(vertices=torch.arange(9).reshape(3, 3),
                              cond_latents=torch.ones((1, 2), dtype=torch.bfloat16),
                              output_ids=torch.tensor([1, 2]), start_tokens_list=[[3, 4]])
        actual = bytes_to_object(packet.to_bytes())
        self.assertTrue(torch.equal(actual.vertices, packet.vertices))
        self.assertTrue(torch.equal(actual.cond_latents, packet.cond_latents))
        self.assertEqual(actual.cond_latents.dtype, torch.bfloat16)
        self.assertEqual(actual.start_tokens_list, packet.start_tokens_list)

    def test_vae_attention_without_external_wheel(self):
        source = ast.parse((ROOT / "src/model/skin_vae_model.py").read_text())
        block = next(node for node in source.body if isinstance(node, ast.Try))
        real_import = builtins.__import__

        def without_flash(name, *args, **kwargs):
            if name.startswith("flash_attn"):
                raise ImportError("No external Flash Attention in this test")
            return real_import(name, *args, **kwargs)

        namespace = {"F": torch.nn.functional}
        with patch("builtins.__import__", without_flash):
            exec(compile(ast.Module(body=[block], type_ignores=[]), "attention_fallback", "exec"), namespace)
        q, k, v = torch.randn(2, 3, 2, 4), torch.randn(2, 5, 2, 4), torch.randn(2, 5, 2, 4)
        actual, auxiliary = namespace["flash_attn_func"](q, k, v)
        scores = torch.einsum("blhd,bshd->bhls", q, k) / 2
        expected = torch.einsum("bhls,bshd->blhd", scores.softmax(-1), v)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(actual.is_contiguous())
        self.assertIsNone(auxiliary)

    def test_array_types(self):
        for array in (np.empty((0, 3), dtype=np.float32), np.array(4),
                      np.array(["á", "東京"]), np.array([1+2j]),
                      np.array([(1, 2.0)], dtype=[("x", "i4"), ("y", "f4")]),
                      np.array([(1, 2.0)], dtype=[("東京", "i4"), ("y", "f4")]),
                      np.arange(20)[::2]):
            actual = bytes_to_object(object_to_bytes(array))
            np.testing.assert_array_equal(actual, array)
            self.assertEqual(actual.dtype, array.dtype)

    def test_nested_object_arrays(self):
        array = np.empty(2, dtype=object)
        array[0] = np.array([1, 2])
        array[1] = np.array([3, 4])
        actual = bytes_to_object(object_to_bytes(array))
        self.assertEqual(actual.shape, (2,))
        for index in range(2):
            np.testing.assert_array_equal(actual[index], array[index])
        structured = np.array([(1, "bone")], dtype=[("id", "i4"), ("name", "O")])
        actual = bytes_to_object(object_to_bytes(structured))
        self.assertEqual(actual.dtype, structured.dtype)
        np.testing.assert_array_equal(actual, structured)

    def test_pickle_transport_rejected(self):
        with self.assertRaises(zipfile.BadZipFile):
            bytes_to_object(pickle.dumps(Malicious()))

    def test_unknown_class_rejected(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("packet.json", json.dumps({"type": "Malicious", "fields": {}}))
        with self.assertRaises(ValueError):
            bytes_to_object(data.getvalue())
        with self.assertRaises(TypeError):
            object_to_bytes(Malicious())

    def test_object_pickle_array_rejected(self):
        data = io.BytesIO()
        array = io.BytesIO()
        np.save(array, np.array([Malicious()], dtype=object))
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("packet.json", json.dumps({"type": "array", "index": 0}))
            archive.writestr("arrays/0.npy", array.getvalue())
        with self.assertRaises(ValueError):
            bytes_to_object(data.getvalue())

    def test_oversized_array_header_rejected(self):
        data, array = io.BytesIO(), io.BytesIO()
        np.lib.format.write_array_header_1_0(array, {"shape": (10 ** 15,), "fortran_order": False, "descr": "<f8"})
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("packet.json", json.dumps({"type": "array", "index": 0}))
            archive.writestr("arrays/0.npy", array.getvalue())
        with self.assertRaises(ValueError):
            bytes_to_object(data.getvalue())

    def test_legacy_npz_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            np.savez(path, names=np.array(["bone", None], dtype=object),
                     matrix=np.eye(4), parents=np.array([None, 0], dtype=object))
            loaded = load_npz(path)
            np.testing.assert_array_equal(loaded["names"], ["bone", None])
            np.testing.assert_array_equal(loaded["matrix"], np.eye(4))
            np.testing.assert_array_equal(loaded["parents"], [None, 0])

    def test_malicious_npz_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            np.savez(path, names=np.array([Malicious()], dtype=object))
            with self.assertRaises(pickle.UnpicklingError):
                load_npz(path)

    def test_checkpoint_plain_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save({"state_dict": {"weight": torch.ones(2)}, "hyper_parameters": {"model_config": {"layers": 4}}}, path)
            actual = load_checkpoint(path)
            self.assertTrue(torch.equal(actual["state_dict"]["weight"], torch.ones(2)))
            self.assertEqual(actual["hyper_parameters"]["model_config"], {"layers": 4})

    def test_checkpoint_omegaconf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            config = OmegaConf.create({"model_config": {"layers": [1, 2], "enabled": True, "name": "qwen"}})
            torch.save({"state_dict": {}, "hyper_parameters": config}, path)
            actual = load_checkpoint(path)
            self.assertEqual(OmegaConf.to_container(actual["hyper_parameters"]), OmegaConf.to_container(config))

    def test_checkpoint_numpy_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save({"state_dict": {}, "data": np.array([1, 2]), "scalar": np.float32(1)}, path)
            actual = load_checkpoint(path)
            np.testing.assert_array_equal(actual["data"], [1, 2])

    def test_malicious_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save(Malicious(), path)
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(path)

    def test_payload_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {PAYLOAD_DIR_ENV: directory}):
                path = Path(directory) / "skintokens_export_test.pt"
                path.write_bytes(object_to_bytes({"value": 1}))
                self.assertEqual(resolve_payload_path(path.name), path.resolve())
                for name in (str(path.resolve()), "../skintokens_export_test.pt", "..\\skintokens_export_test.pt", "C:skintokens_x.pt", "other.pt"):
                    with self.assertRaises(ValueError):
                        resolve_payload_path(name)

    def test_bundled_assets(self):
        directory = ROOT / "web/vendor"
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(len(manifest), 7)
        for name, details in manifest.items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), details["sha256"])
        self.assertEqual(len(list((ROOT / "web").rglob("*.js"))), 1)
        self.assertNotIn('loadScript("https://', (ROOT / "web/js/preview3d.js").read_text())

    def test_http_auth_and_safe_file_payload(self):
        # Test the actual server routes without importing Blender or starting inference.
        fake_bpy = types.ModuleType("src.rig_package.parser.bpy")
        fake_bpy.BpyParser = object
        fake_bpy.transfer_rigging = lambda **kwargs: None
        with patch.dict(sys.modules, {"src.rig_package.parser.bpy": fake_bpy}):
            from src.server.bpy_server import create_app, _resolve_payload
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {TOKEN_ENV: "a" * 64, PAYLOAD_DIR_ENV: directory}):
                path_queue, result_queue = queue.Queue(), queue.Queue()
                httpd = make_server("127.0.0.1", 0, create_app(path_queue, result_queue), handler_class=QuietHandler)
                thread = threading.Thread(target=httpd.serve_forever, daemon=True)
                thread.start()
                url = f"http://127.0.0.1:{httpd.server_port}"
                session = requests.Session()
                session.trust_env = False
                try:
                    self.assertEqual(session.get(url + "/ping").status_code, 403)
                    self.assertEqual(session.post(url + "/export", data=pickle.dumps(Malicious())).status_code, 403)
                    headers = {"Authorization": "Bearer " + "a" * 64, "Origin": "https://example.com"}
                    self.assertEqual(session.get(url + "/ping", headers=headers).status_code, 403)
                    self.assertEqual(server_request("GET", url + "/ping").text, "pong")
                    path = Path(directory) / "skintokens_export_test.pt"
                    path.write_bytes(object_to_bytes({"filepath": "a.glb", "asset": Asset(vertices=np.zeros((3, 3)))}))
                    result_queue.put("ok")
                    response = server_request("POST", url + "/export", data=object_to_bytes({"payload_path": path.name}))
                    self.assertEqual(bytes_to_object(response.content), "ok")
                    operation, data = path_queue.get(timeout=2)
                    self.assertEqual(operation, "export")
                    payload = _resolve_payload(bytes_to_object(data))
                    self.assertIsInstance(payload["asset"], Asset)
                    self.assertTrue(path.exists(), "Server must not delete a supplied path")
                    with self.assertRaises(ValueError):
                        _resolve_payload({"payload_path": str(path.resolve())})
                    self.assertTrue(path.exists())
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                    session.close()

    def test_installer_rejects_modified_wheel(self):
        import install
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flash_attn-2.8.3-py3-none-any.whl"
            path.write_bytes(b"tampered")
            with patch.object(install.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    install.install_flash_attn_section(str(path), "0" * 64)
                run.assert_not_called()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                install.install_flash_attn_section(str(path), digest)
                self.assertIn("--no-deps", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
