import torch
import torch.nn as nn
import time
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
print("Pytorch version: ", torch.__version__)
print("CUDA version: ", torch.version.cuda)
print("cuDNN version: ", torch.backends.cudnn.version())
print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
print("CUDA device name: ", torch.cuda.get_device_name())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

# Dummy CUDA computation (large matrix multiplication)
x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)

print("Starting CUDA computation...")
for _ in range(10000):
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # ensure computation starts immediately
print("CUDA computation done.")
