import torch
import librosa
import numpy as np
import os

print(f"✓ Python version: {os.sys.version}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ Librosa version: {librosa.__version__}")
print("\n✓ All dependencies installed successfully!")