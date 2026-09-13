"""Test the notebook's actual validation cell without importing GPU packages."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

source = Path(__file__).with_name("Hunyuan3D_Blender_Turntable_fixed.py").read_text()
module = ast.parse(source)
cell = copy.deepcopy(next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "checkpoint_security"))
cell.decorator_list = []
namespace = {}
exec(compile(ast.Module(body=[cell], type_ignores=[]), "checkpoint_security", "exec"), namespace)
validate = namespace["checkpoint_security"]()[0]


class CheckpointSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "model"
        self.root.mkdir()
        (self.root / "weights.safetensors").write_bytes(b"dummy-weights")

    def index(self, shard):
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": shard}}))

    def test_regular_shards_inside_the_checkpoint_folder_work(self):
        self.index("weights.safetensors")
        self.assertEqual(validate(self.root), self.root)

    def test_traversal_and_absolute_shards_are_rejected(self):
        for name in ["../outside", "/etc/passwd", "C:\\outside.bin", "nested/../../outside"]:
            with self.subTest(name=name):
                self.index(name)
                with self.assertRaises(ValueError):
                    validate(self.root)

    def test_invalid_weight_map_is_rejected(self):
        for weights in [None, ["weights.safetensors"], {"layer": 42}]:
            with self.subTest(weights=weights):
                (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
                with self.assertRaises(ValueError):
                    validate(self.root)

    def test_symlink_escape_is_rejected(self):
        outside = self.root.parent / "outside.bin"
        outside.write_bytes(b"private")
        try:
            (self.root / "link.bin").symlink_to(outside)
        except OSError:
            self.skipTest("symlink creation not available")
        self.index("link.bin")
        with self.assertRaises(ValueError):
            validate(self.root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO support")
    def test_named_pipe_is_rejected_without_opening_or_blocking(self):
        os.mkfifo(self.root / "pipe.bin")
        self.index("pipe.bin")
        with self.assertRaises(ValueError):
            validate(self.root)

    def test_model_loaders_receive_only_the_validated_local_snapshot(self):
        generate = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
        loads = [n for n in ast.walk(generate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "from_pretrained"]
        self.assertEqual(len(loads), 2)
        self.assertTrue(all(isinstance(n.args[0], ast.Name) and n.args[0].id == "_model_path" for n in loads))
        generate_text = ast.get_source_segment(source, generate)
        self.assertLess(generate_text.index("validate_model_snapshot(_model_path)"), generate_text.index(".from_pretrained("))


if __name__ == "__main__":
    unittest.main()
