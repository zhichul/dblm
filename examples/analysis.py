import math
import random
import numpy as np

from dblm.utils import seeding

seeding.seed(12, tf=False)
N = 50
aa = sorted([(0.5-random.random()) * 8 + 5 for _ in range(N)])
print(aa)
lbds = np.array([1/N] * (N-1))
lr = 0.01
for i in range(10000):
    mean = sum(ai * li for ai, li in zip(aa[1:], lbds)) + (1-sum(lbds)) * aa[0]
    grad = np.array([0.] * len(lbds))
    for j in range(len(lbds)):
        grad[j] = 1/aa[j+1] - 1/aa[0] - (-1/(mean)**2) * (aa[j+1] - aa[0])
    if lbds.sum() == 1:
        grad = grad - np.ones(grad.shape) / math.sqrt(N-1) * np.dot(np.ones(grad.shape) / math.sqrt(N-1), grad)
    lbds = lbds + grad * lr
    lbds = lbds.clip(0, 1)
    if lbds.sum() > 1:
        lbds /= lbds.sum()
    if i % 1000 == 0:
        print(lbds, sum(1/ai * li for ai, li in zip(aa[1:], lbds)) + (1-sum(lbds)) * 1/aa[0] - 1/mean)
print((aa[-1]-(aa[-1]*aa[0]) ** 0.5)/(aa[-1]-aa[0]))
breakpoint()
