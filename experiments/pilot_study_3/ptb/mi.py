import torch
import sys
from collections import defaultdict
import math
from dblm.experiments.filter_ptb import filter_fn

nvals=27
length=12
c2i = {c:i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
zs = []
with open("ptb.train.txt", "rt") as f:
    for line in f:
        for token in filter(filter_fn, line.strip().split()):
            z = [0] * length
            for i, char in enumerate(token[:length]):
                z[i] = c2i[char]
            zs.append(z)
zs = torch.tensor(zs)
def mutual_info(xs, ys):
    total = len(xs)
    counter = defaultdict(float)
    for x, y in zip(xs, ys):
        counter[(x, None)] += 1
        counter[(None, y)] += 1
        counter[(x,y)] += 1
    for k in counter:
        counter[k] /= total
    mi = 0
    for i in range(nvals):
        for j in range(nvals):
            pij = counter[(i,j)]
            pi = counter[(i,None)]
            pj = counter[(None,j)]
            if pij==0:
                continue
            mi += pij * (math.log(pij) - math.log(pi) - math.log(pj))
    return mi

for i in range(length):
    for j in range(i, length):
        print(f"I(z{i}, z{j}) = {mutual_info(zs[:,i].tolist(), zs[:,j].tolist())}")
    print(" ****************************** ")
