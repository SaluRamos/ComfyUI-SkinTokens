# Intel Arc / Arc Pro B70

The ComfyUI model loader uses ComfyUI's selected device. The standalone demo automatically selects CUDA, then Intel XPU, then CPU. Tensor batches follow the model device rather than hard-coded CUDA.

With a current Intel graphics/compute driver for your OS, install native PyTorch XPU in the inference environment and then the node dependencies:

```sh
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
python -c "import torch; assert torch.xpu.is_available(); print(torch.xpu.get_device_name(0))"
```

Restart ComfyUI. Intel uses native PyTorch SDPA, including grouped-query attention; the presence of a CUDA FlashAttention package cannot force XPU tensors into a CUDA kernel. CUDA retains optional FlashAttention. Autocast and full-precision quantization follow the actual tensor/model backend. Model checkpoints and the Blender bridge continue using their existing formats and setup.

Validation performed: backend selection, loading-context restoration, model AMP dispatch and prevention of CUDA-kernel dispatch for Intel, using mocked backends; Python compilation. A numerical grouped-query attention test is included for actual PyTorch CPU/XPU and runs when PyTorch is installed. It was skipped locally because PyTorch is unavailable. No complete rigging workflow has been run on Arc Pro B70 in this environment.

Run `python -m unittest discover -s test_scripts -p test_devices.py -v` in the installed environment, then generate a rig for a small mesh. Check bones, skin weights and export before using a production asset. Performance and peak VRAM on B70 remain unmeasured.
