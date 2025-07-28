import random
import numpy as np
import torch

def seed(n, tf=False):
    if tf:
        import tensorflow as tf
        tf.random.set_seed(n)
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)

