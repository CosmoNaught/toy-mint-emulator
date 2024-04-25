import random as rn
import numpy as np
import torch

def set_seed(int):
    # Setting seeds for different libraries
    rn.seed(int)       # Python's built-in random module
    np.random.seed(int)    # NumPy library
    torch.manual_seed(int) # PyTorch for CPU operations
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(int)  # PyTorch for all CUDA devices (GPUs)