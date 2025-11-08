import torch
import scipy.io as sio
import numpy as np

print("Testing setup...")

# 1. Check GPU
print(f"\n1. GPU Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 2. Check data file
print(f"\n2. Data Check:")
try:
    mat_data = sio.loadmat('data/train/train_001.mat')
    print(f"   ✓ Successfully loaded data file")
    print(f"   Keys in file: {list(mat_data.keys())}")
    if 'mixture_signal' in mat_data:
        print(f"   mixture_signal shape: {mat_data['mixture_signal'].shape}")
    if 'target_at_mics' in mat_data:
        print(f"   target_at_mics shape: {mat_data['target_at_mics'].shape}")
except Exception as e:
    print(f"   ✗ Error loading data: {e}")

# 3. Check metrics packages
print(f"\n3. Metrics Check:")
try:
    from pesq import pesq
    print(f"   ✓ PESQ installed")
except ImportError:
    print(f"   ✗ PESQ not installed: pip install pesq")

try:
    from pystoi import stoi
    print(f"   ✓ STOI installed")
except ImportError:
    print(f"   ✗ STOI not installed: pip install pystoi")

# 4. Quick model test
print(f"\n4. Model Test:")
from audio_zoom_transformer import AudioZoomingTransformer
try:
    model = AudioZoomingTransformer(n_fft=512, n_mics=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created: {n_params/1e6:.2f}M parameters")
except Exception as e:
    print(f"   ✗ Error creating model: {e}")

print("\n" + "="*50)
print("Setup test complete!")