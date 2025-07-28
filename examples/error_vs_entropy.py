import bisect
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda import FDataGrid
import torch
import tqdm

d = 10

def plot_fit(ax, xs, ys, color):
    gpr = GaussianProcessRegressor(kernel=RBF((xs.max() - xs.min()) * 0.01), random_state=0).fit(xs[:,None], ys)
    sorted_xs = np.array(sorted(xs.tolist()))
    ymean = gpr.predict(sorted_xs[:,None])
    fd = FDataGrid(data_matrix=ymean[None, :], grid_points=sorted_xs) # type:ignore
    fd_os = KernelSmoother(
    kernel_estimator=NadarayaWatsonHatMatrix(bandwidth=(xs.max() - xs.min()) * 0.1),
    ).fit_transform(fd)
    ax.plot(sorted_xs, fd_os.data_matrix.reshape(-1), c=color, linewidth=0.5, alpha=0.2, linestyle="dashed")

for d, color in zip([10, 50, 100], ["red", "green", "yellow", "brown"]):#np.linspace(0, 100, num=10, dtype=np.int64):
    ents =             []
    errs = []
    x = np.random.lognormal(0, 2, size=d)
    for i in range(10000):
        for p in [np.random.dirichlet(np.linspace(5*0.5, 5*2.0, d)), np.random.dirichlet(np.linspace(4*0.5, 4*2.0, d)), np.random.dirichlet(np.linspace(2*0.5, 2*2.0, d)), np.random.dirichlet(np.linspace(1*0.5, 1*2.0, d)),np.random.dirichlet(np.linspace(0.75*0.5, 0.75*2.0, d)), np.random.dirichlet(np.linspace(0.5*0.5, 0.5*2.0, d)), np.random.dirichlet(np.linspace(0.1*0.5, 0.1*2.0, d)), np.random.dirichlet(np.linspace(0.05*0.5, 0.05*2.0, d))]:
            ent = -(p * np.log(p)).sum()
            samples = np.random.choice(x, size=1000, replace=True, p=p)
            sample_mean = samples.mean()
            mean = np.dot(x, p).sum()
            error = abs(sample_mean - mean)
            ents.append(ent/d)
            errs.append(error)

    print(ents[:10], errs[:10])
    ents = np.array(ents)
    errs = np.array(errs)
    entmin = ents.min()
    entmax = ents.max()
    bins = np.linspace(entmin, entmax+0.01, 20)
    points_by_bin = defaultdict(list)
    for ent, err in zip(ents, errs):
        points_by_bin[bisect.bisect(bins, ent)].append(err)
    xs = []
    ys = []
    for i in sorted(points_by_bin.keys()):
        xs.append(bins[i-1])
        ys.append(np.array(points_by_bin[i]).mean())
    plt.plot(xs, ys, c="blue")
    plt.scatter(ents, errs, s=2, alpha=0.5, c=color)
    plt.show()
