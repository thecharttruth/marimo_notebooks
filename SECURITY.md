# Model-loading safeguards

The notebook pins PyTorch to 2.13.0, which addresses
[GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47).
Existing compiled CUDA extensions such as `custom_rasterizer` may need rebuilding
against this PyTorch version. Full generation still requires a compatible GPU
runtime; the portable security tests do not exercise CUDA or Blender rendering.

Accelerate 1.14.0 remains subject to
[GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79).
The 1.15.0 package inspected during this fix still joined unvalidated checkpoint
shard paths, so a version bump alone was not used as the remedy. The notebook
mitigates the issue before calling its model loaders:

- Download the official `tencent/Hunyuan3D-2` model at the immutable revision
  `9cd649ba6913f7a852e3286bad86bfa9a2d83dcf` into a dedicated local directory.
- Reject absolute/traversing shard names, symlinks and special files, including
  named pipes, before model loading. Bound checkpoint JSON size.
- Pass only the validated local snapshot to both shape and paint loaders.
  Shape loading explicitly uses safetensors.

This protects this notebook's loading path; it does not patch every use of
Accelerate in the environment. Keep the model directory private and treat model
weights and custom pipeline code as executable inputs. Changing the pinned model
revision requires reviewing its contents again.

`hy3dgen` remains pinned to 2.0.2 in the explicit `--no-deps` installer; it is not
duplicated in the script's dependency block, which would pull unsupported rembg
dependencies into the Python 3.13 environment.

Run the portable regressions with `python -m unittest test_checkpoint_security -v`.
They cover legitimate files, traversal, absolute paths, malformed indexes,
symlink escapes and named pipes without downloading models or using a GPU.
