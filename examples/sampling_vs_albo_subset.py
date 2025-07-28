r"""
This code compares the L1 error of sampling vs ALBO

FOR APPROXIMATING A SINGLE EXPECTED ATTENTION WEIGHT

with various knobs that can be turned:

* the probability P(r \in S)
* the ratio between Zmin and Zmax
* the number of samples
* whether to sample estimate P(r \in S) or use exact knowledge of it
"""
import math
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np
from scipy.stats import binom
from sampling_vs_albo_utils import set_errorbar_data
from matplotlib.cm import rainbow # type:ignore


# simple worst case
zmin = 1 # DONT CHANGE
ratio = 1.74
offset = 0
nsamples=10
d = 2
pexist = 1
low_ent_bias=4.0
def expected_value(zs, ps):
    return np.dot(zs, ps)

def albo_err(zs, ps, pexist=1, approx=False, n=10):
    target = expected_value(1/zs, ps) * pexist
    pestimate = np.random.choice([0,1.], p=np.array([1-pexist, pexist]), size=n).mean()
    albo = 1/expected_value(zs, ps) * (pestimate if approx else pexist)
    return np.abs(target - albo)

def sample_err(zs, ps, n=nsamples, simulations=100, pexist=1, approx=False):
    target = expected_value(1/zs, ps) * pexist
    samples = np.random.choice((1/zs), p=ps, size=(simulations, n))
    pestimate = np.random.choice([0,1.], p=np.array([1-pexist, pexist]), size=(simulations, n)).mean(axis=1)
    estimates = samples.mean(axis=1) * (pestimate if approx else pexist)
    errs = np.abs(target - estimates)
    return errs.mean(), errs.std()

def expected_value2(zmin, zmax, lbd):
    exp = zmin * lbd + zmax * (1-lbd)
    return exp

def albo2_err(zmin, zmax, lbd):
    target = 1/zmin * lbd + 1/zmax * (1-lbd)
    albo = 1/(zmin * lbd + zmax * (1-lbd))
    return target - albo

def sample2_err(zmin, zmax, lbd, n=nsamples, simulations=100):
    target = 1/zmin * lbd + 1/zmax * (1-lbd)
    samples = np.random.choice([1, 0.], p=[lbd, 1-lbd], size=(simulations, n))
    estimate_lbd = samples.mean(axis=1)
    estimates = 1/zmin * estimate_lbd + 1/zmax * (1-estimate_lbd)
    errs = np.abs(target - estimates)
    return errs.mean(), errs.std()


def get_point(ps, zs, nsamlpes, pexist, approx_spl, approx_albo):
    return expected_value(zs, ps), sample_err(zs, ps, n=nsamlpes, pexist=pexist, approx=approx_spl)[0], albo_err(zs, ps, pexist=pexist, approx=approx_albo, n=nsamlpes), (-ps * np.log(ps)).sum()

def get_points(zmin, ratio, offset, nsamples, d, pexist, approx_spl, approx_albo, low_ent_bias=low_ent_bias):
    print(approx_spl, approx_albo)
    d = int(d)
    zs = np.linspace(zmin, zmin * ratio, d) + offset
    ps = [np.random.dirichlet(np.array([k**low_ent_bias] * d)) for k in np.linspace(0.1, 1, 20) for _ in range(1000)]
    # ps = [np.random.dirichlet(np.array([k**4] * d)) for k in np.linspace(0.2, 1, 20) for _ in range(1000)]
    xs = []
    y1s = []
    y2s = []
    os = []
    for p in ps:
        if np.isnan(p).any() or np.isinf(p).any(): continue
        x,y1,y2,o = get_point(p, zs, nsamples, pexist, approx_spl, approx_albo)
        xs.append(x)
        y1s.append(y1)
        y2s.append(y2)
        os.append(o)
    return np.array(xs), np.array(y1s), np.array(y2s), np.array(os), math.log(d)

fig, axs = plt.subplots(1,2)
print(axs)
fig.subplots_adjust(left=0.3, bottom=0.3, right=0.9)

axd = fig.add_axes([0.25, 0.05, 0.65, 0.03])
d_slider = Slider(
    ax=axd,
    label="|V|",
    valmin=2,
    valmax=100,
    valinit=ratio,
    orientation="horizontal",
    valstep=1
)
axlow_ent = fig.add_axes([0.25, 0.01, 0.65, 0.03])
lowent_slider = Slider(
    ax=axlow_ent,
    label="low ent. bias",
    valmin=-4,
    valmax=4,
    valinit=low_ent_bias,
    orientation="horizontal",
    valstep=0.1
)
axratio = fig.add_axes([0.25, 0.1, 0.65, 0.03])
ratio_slider = Slider(
    ax=axratio,
    label="log(Z_max - Zmin) -log (Zmin - offset)",
    valmin=-10,
    valmax=10,
    valinit=math.log(ratio-1),
    orientation="horizontal"
)
axslider = fig.add_axes([0.25, 0.15, 0.65, 0.03])
offset_slider = Slider(
    ax=axslider,
    label="Z_min - 1",
    valmin=0,
    valmax=100,
    valinit=offset,
    orientation="horizontal"
)
axpexist = fig.add_axes([0.25, 0.20, 0.65, 0.03])
pexist_slider = Slider(
    ax=axpexist,
    label="pexist",
    valmin=0,
    valmax=1,
    valinit=pexist,
    orientation="horizontal"
)
axnsample = fig.add_axes([0.02, 0.25, 0.0225, 0.63])
nsample_slider = Slider(
    ax=axnsample,
    label="# samples",
    valmin=1,
    valmax=100,
    valinit=nsamples,
    orientation="vertical"
)
hist_axes = fig.add_axes([0.08, 0.3, 0.18, 0.63])
hist_axes.set_xlabel("count")

rax = fig.add_axes([0.9, 0.3, 0.085, 0.1])
check = CheckButtons(
    ax=rax,
    labels=["spl approx. m", "albo approx. m"],
    actives=[False, False],
    label_props={'color': ["black", "black"]}, # type:ignore
    frame_props={'edgecolor': ["black", "black"]}, # type:ignore
    check_props={'facecolor': ["black", "black"]}, # type:ignore
)


xs, y1s, y2s, zs, entmax = get_points(zmin, ratio, offset, nsamples, d, pexist, False, False)
cs = rainbow(zs/entmax)
scatter1 = axs[0].scatter(xs, y1s, c=cs, s=2, alpha=0.1)
hline1 = axs[0].hlines(0.1, zmin, max(xs), color="red", linewidth=1)
scatter2 = axs[1].scatter(xs, y2s, c=cs, s=2, alpha=0.1)
hline2 = axs[1].hlines(0.1, zmin, max(xs), color="red", linewidth=1)
x = np.linspace(0, 1, num=100)
xtrans = expected_value2(zmin+offset, zmin * ratio+offset, x)
albo_y = [albo2_err(zmin+offset, zmin * ratio+offset, lbd) * pexist for lbd in x]
albo_bound, = axs[1].plot(xtrans, albo_y,color='black', linestyle="dashed")
ratio_anno = axratio.text(0.5, 0.5, str(round(1+math.exp(ratio_slider.val),2)))

histspl = hist_axes.hist(y1s, orientation="horizontal", bins=np.linspace(min(min(y1s), min(y2s)), max(max(y1s),max(y2s)), 50), color="orange", alpha=0.5, density=True)
histalb = hist_axes.hist(y2s, orientation="horizontal", bins=np.linspace(min(min(y1s), min(y2s)), max(max(y1s),max(y2s)), 50), color="blue", alpha=0.5, density=True)
axs[0].set_ylim(0, max(max(y1s), max(y2s), max(albo_y)))
axs[0].set_xlim(zmin+offset, max(xs))
axs[1].set_ylim(0, max(max(y1s), max(y2s), max(albo_y)))
axs[1].set_xlim(zmin+offset, max(xs))
def update(val):
    global scatter1
    global hline1
    global scatter2
    global hline2
    global histalb
    global histspl
    scatter1.remove()
    hline1.remove()
    scatter2.remove()
    hline2.remove()

    histspl[-1].remove() #type:ignore
    histalb[-1].remove() #type:ignore


    xs, y1s, y2s, zs, entmax = get_points(zmin, 1+math.exp(ratio_slider.val), offset_slider.val, int(nsample_slider.val), d_slider.val, pexist_slider.val, check.get_status()[0], check.get_status()[1], low_ent_bias=lowent_slider.val)
    cs = rainbow(zs/entmax)
    scatter1 = axs[0].scatter(xs, y1s, c=cs, s=2, alpha=0.1)
    hline1 = axs[0].hlines(0.1, zmin, max(xs), color="red", linewidth=1)
    scatter2 = axs[1].scatter(xs, y2s, c=cs, s=2, alpha=0.1)
    hline2 = axs[1].hlines(0.1, zmin, max(xs), color="red", linewidth=1)
    albo_y = np.array([albo2_err(zmin+offset_slider.val, zmin * 1+math.exp(ratio_slider.val)+offset_slider.val, lbd) for lbd in x]) * pexist_slider.val
    albo_bound.set_ydata(albo_y)
    albo_bound.set_xdata(expected_value2(zmin+offset_slider.val, zmin * 1+math.exp(ratio_slider.val)+offset_slider.val, x))
    histspl = hist_axes.hist(y1s, orientation="horizontal", bins=np.linspace(min(min(y1s), min(y2s)), max(max(y1s),max(y2s)), 50), color="orange", alpha=0.5, density=True)
    histalb = hist_axes.hist(y2s, orientation="horizontal", bins=np.linspace(min(min(y1s), min(y2s)), max(max(y1s),max(y2s)), 50), color="blue", alpha=0.5, density=True)
    axs[0].set_ylim(0, max(max(y1s), max(y2s), max(albo_y)))
    axs[0].set_xlim(zmin+offset_slider.val, max(xs))
    axs[1].set_ylim(0, max(max(y1s), max(y2s), max(albo_y)))
    axs[1].set_xlim(zmin+offset_slider.val, max(xs))
    ratio_anno.set_text(str(round(1+math.exp(ratio_slider.val),2)))
    fig.canvas.draw_idle()
    
ratio_slider.on_changed(update)
nsample_slider.on_changed(update)
offset_slider.on_changed(update)
d_slider.on_changed(update)
pexist_slider.on_changed(update)
check.on_clicked(update)
lowent_slider.on_changed(update)
axs[0].set_title("Sample approx. error (L1) to E_{S | r in S}[a_r/sum_S a_u]")
axs[1].set_title("ALBO approx. error (L1) to E_{S | r in S}[a_r/sum_S a_u]")
axs[0].set_xlabel("E_{S | r in S}[sum_S a_u]")
axs[1].set_xlabel("E_{S | r in S}[sum_S a_u]")
fig.set_size_inches(16, 9)

plt.show()

