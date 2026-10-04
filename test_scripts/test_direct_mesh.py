import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import trimesh
from PIL import Image

COMFY_ROOT = Path(r"D:\new-comfyui\ComfyUI (1)\ComfyUI")
PLUGIN_ROOT = COMFY_ROOT / "custom_nodes/ComfyUI-SkinTokens"
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))
source = Path(os.environ.get("SKINTOKENS_TEST_NODES", str(PLUGIN_ROOT / "nodes.py")))
spec = importlib.util.spec_from_file_location("skintokens_mesh_test", source)
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)
from comfy_api.latest import Types
from comfy_execution.validation import validate_node_input

VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
FACES = np.array([[0, 1, 2]], dtype=np.int64)
UVS = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=np.float32)


class DirectMeshTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.temp_patch = patch.object(nodes.folder_paths, "get_temp_directory", return_value=self.directory.name)
        self.temp_patch.start()

    def tearDown(self):
        self.temp_patch.stop()
        self.directory.cleanup()

    def test_connector_validation(self):
        accepted = nodes.SkinTokensGenerator.INPUT_TYPES()["required"]["input_mesh"][0]
        for type_name in ("TRIMESH", "MESH", "STRING"):
            self.assertTrue(validate_node_input(type_name, accepted, strict=True))
        self.assertFalse(validate_node_input("IMAGE", accepted, strict=True))

    def test_string_and_path_unchanged(self):
        for original in ("character.fbx", Path("character.glb")):
            with nodes.prepare_input_mesh(original) as filename:
                self.assertEqual(filename, os.fspath(original))
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_trimesh_uv_and_texture(self):
        mesh = trimesh.Trimesh(vertices=VERTICES, faces=FACES, process=False)
        image = Image.new("RGB", (2, 2), (250, 20, 10))
        mesh.visual = trimesh.visual.TextureVisuals(uv=UVS, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=image, metallicFactor=0.3, roughnessFactor=0.7))
        with nodes.prepare_input_mesh(mesh) as filename:
            self.assertTrue(Path(filename).exists())
            restored = next(iter(trimesh.load(filename, force="scene", process=False).geometry.values()))
            np.testing.assert_allclose(restored.vertices, VERTICES)
            np.testing.assert_array_equal(restored.faces, FACES)
            np.testing.assert_allclose(restored.visual.uv, UVS)
            self.assertEqual(restored.visual.material.baseColorTexture.getpixel((0, 0))[:3], (250, 20, 10))
            self.assertAlmostEqual(restored.visual.material.metallicFactor, 0.3)
            self.assertAlmostEqual(restored.visual.material.roughnessFactor, 0.7)
        self.assertFalse(Path(filename).exists())
        np.testing.assert_allclose(mesh.visual.uv, UVS)

    def test_native_mesh_texture_and_material(self):
        mesh = Types.MESH(torch.tensor(VERTICES).unsqueeze(0), torch.tensor(FACES).unsqueeze(0),
                          uvs=torch.tensor(UVS).unsqueeze(0), texture=torch.ones((1, 2, 2, 3)),
                          material={"metallic_factor": 0.4, "roughness_factor": 0.6})
        with nodes.prepare_input_mesh(mesh) as filename:
            restored = next(iter(trimesh.load(filename, force="scene", process=False).geometry.values()))
            np.testing.assert_allclose(restored.vertices, VERTICES)
            np.testing.assert_array_equal(restored.faces, FACES)
            self.assertIsNotNone(restored.visual.uv)
            self.assertIsNotNone(restored.visual.material.baseColorTexture)
            self.assertAlmostEqual(restored.visual.material.metallicFactor, 0.4)
            self.assertAlmostEqual(restored.visual.material.roughnessFactor, 0.6)
        self.assertFalse(Path(filename).exists())

    def test_scene_transforms_preserved(self):
        scene = trimesh.Scene()
        transform = np.eye(4)
        transform[0, 3] = 4
        scene.add_geometry(trimesh.Trimesh(vertices=VERTICES, faces=FACES, process=False), transform=transform)
        with nodes.prepare_input_mesh(scene) as filename:
            restored = trimesh.load(filename, force="scene", process=False)
            np.testing.assert_allclose(restored.bounds, scene.bounds)

    def test_cleanup_when_generation_fails(self):
        mesh = trimesh.Trimesh(vertices=VERTICES, faces=FACES, process=False)
        captured = []

        def fail(model, filename, *args):
            captured.append(Path(filename))
            self.assertTrue(Path(filename).exists())
            raise RuntimeError("Simulated inference failure")

        generator = nodes.SkinTokensGenerator()
        with patch.object(generator, "_generate_from_path", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "Simulated"):
                generator.generate(None, mesh, 5, 0.95, 1., 2., 10, False, True, False,
                                   "articulated", ".glb", "Headless (Blender)")
        self.assertFalse(captured[0].exists())

    def test_invalid_input_rejected_and_cleaned(self):
        with self.assertRaises(TypeError):
            with nodes.prepare_input_mesh({"path": "arbitrary"}):
                self.fail("Unexpected input accepted")
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
