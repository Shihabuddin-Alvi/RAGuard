import numpy as np

def softmax(logits):
    x = np.array(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
