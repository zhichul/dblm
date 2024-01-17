import random
import numpy as np
import torch
import tensorflow as tf

def seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    tf.random.set_seed(n)

