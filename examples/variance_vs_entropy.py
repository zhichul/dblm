import bisect
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random

d = 10


for d, color in zip([10], ["red", "green", "blue", "yellow"]):#np.linspace(0, 100, num=10, dtype=np.int64):
    ents = []
    vars = []
    np.random.seed(42)
    x = np.random.lognormal(0, 1, size=d)[:, None]
    z = np.array(np.meshgrid(*([np.array([0.,1.])] * d))).reshape(d, -1)
    Zs = 1/ (np.random.lognormal(0,1) + (z * x).sum(axis=0))
    print(Zs)
    print(((Zs.max() - Zs.min())**2/4)**0.5)
    for i in range(10000):
        for p in [np.random.dirichlet(np.array([k**2] * 2**d)) for k in np.linspace(0.2, 1, 20)]:
            ent = -(p * np.log(p)).sum()
            mean = np.dot(Zs, p).sum()
            var = np.dot((Zs-mean)**2, p).sum() ** 0.5

            ents.append(ent/d)
            vars.append(var)
    print(ents[:10], vars[:10])
    plt.scatter(ents, vars, s=2, alpha=0.05, c=color)
    ents = np.array(ents)
    vars = np.array(vars)
    entmin = ents.min()
    entmax = ents.max()
    bins = np.linspace(entmin, entmax+0.01, 20)
    points_by_bin = defaultdict(list)
    for ent, var in zip(ents, vars):
        points_by_bin[bisect.bisect(bins, ent)].append(var)
    xs = []
    ys = []
    for i in sorted(points_by_bin.keys()):
        xs.append(bins[i-1])
        ys.append(np.array(points_by_bin[i]).mean())
    plt.plot(xs, ys, c="blue")
plt.show()
