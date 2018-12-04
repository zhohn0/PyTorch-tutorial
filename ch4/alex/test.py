import torch
import numpy as np
t = np.array([[1, 1], [2, 2]], dtype='float32')
print(t)
t.resize((2, 2, 2))
print(t)
