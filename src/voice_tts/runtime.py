"""Explicit device selection for local CUDA, Apple Metal and CPU inference."""
import os


def model_load_options():
    import torch
    device = os.environ.get('VOICE_TTS_DEVICE', 'auto')
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device not in ('cuda:0', 'mps', 'cpu'):
        raise ValueError('VOICE_TTS_DEVICE must be auto, mps, cpu, or cuda:0')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple Metal is unavailable. Use native arm64 Python on macOS.')
    return {'device_map': device, 'dtype': torch.bfloat16 if device == 'cuda:0' else torch.float32,
            'attn_implementation': 'sdpa', 'low_cpu_mem_usage': True}
