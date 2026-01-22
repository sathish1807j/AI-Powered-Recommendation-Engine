import sys
import pandas as pd
import numpy as np
import torch

print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("Torch version:", torch.__version__)

# Quick tensor check
x = torch.tensor([1.0, 2.0, 3.0])
print("Torch tensor works:", x)
