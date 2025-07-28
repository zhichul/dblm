from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy.stats import truncnorm
from sampling_vs_albo_utils import set_errorbar_data
# simple worst case
zmin = 1
ratio = 3
nsamples=10

def expected_value(zmin, zmax, lbd):
    exp = zmin * lbd + zmax * (1-lbd)
    return exp

def albo_err(zmin, zmax, lbd):
    target = 1/zmin * lbd + 1/zmax * (1-lbd)
    albo = 1/(zmin * lbd + zmax * (1-lbd))
    return target - albo

def sample_err(zmin, zmax, lbd, n=nsamples, simulations=100):
    target = 1/zmin * lbd + 1/zmax * (1-lbd)
    samples = np.random.choice([1, 0.], p=[lbd, 1-lbd], size=(simulations, n))
    estimate_lbd = samples.mean(axis=1)
    estimates = 1/zmin * estimate_lbd + 1/zmax * (1-estimate_lbd)
    errs = np.abs(target - estimates)
    return errs.mean(), errs.std()


fig, ax = plt.subplots()
fig.subplots_adjust(left=0.25, bottom=0.25)

axratio = fig.add_axes([0.25, 0.1, 0.65, 0.03])
ratio_slider = Slider(
    ax=axratio,
    label="ratio",
    valmin=1.001,
    valmax=100,
    valinit=ratio,
    orientation="horizontal"
)
axnsample = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
nsample_slider = Slider(
    ax=axnsample,
    label="# samples",
    valmin=1,
    valmax=100,
    valinit=nsamples,
    orientation="vertical"
)
x = np.linspace(0, 1, num=100)
xtrans = expected_value(zmin, zmin * ratio, x)
albo_y = [albo_err(zmin, zmin * ratio, lbd) for lbd in x]
sample_y_pair = [sample_err(zmin, zmin * ratio, lbd) for lbd in x]
sample_y = [mean for mean, std in sample_y_pair]
sample_y_std = [std for mean, std in sample_y_pair]

def update(val):
    xtrans = expected_value(zmin, zmin * ratio_slider.val, x)
    line.set_xdata(xtrans)
    line.set_ydata([albo_err(zmin, zmin*ratio_slider.val, lbd) for lbd in x])
    sample_y_pair = [sample_err(zmin, zmin * ratio_slider.val, lbd, n=int(nsample_slider.val)) for lbd in x]
    sample_y = np.array([mean for mean, std in sample_y_pair])
    sample_y_std = np.array([std for mean, std in sample_y_pair])
    set_errorbar_data(errbar, xtrans, sample_y, y_error=sample_y_std)
    ax.set_ylim(0, max(line.get_ydata()) * 1.1)
    ax.set_xlim(1, max(xtrans))
    global hline
    hline.remove()
    hline = ax.hlines(0.1, 1, max(xtrans), color="red", linewidth=1)
    fig.canvas.draw_idle()

ratio_slider.on_changed(update)
nsample_slider.on_changed(update)

line, = ax.plot(xtrans, albo_y)
errbar = ax.errorbar(xtrans,sample_y, sample_y_std)
ax.set_ylim(0, max(line.get_ydata()) * 1.1)
ax.set_xlim(1, max(xtrans))
hline = ax.hlines(0.1, 1, max(xtrans), color="red", linewidth=1)
plt.show()

