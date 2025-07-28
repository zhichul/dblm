import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np

CI = False
PLOT="test"
# theoretical_min = 2.516202449798584
theoretical_min = 2.517962694168091

def best_by(ds, fieldname, mode="max"):
    if mode not in ["max", "min"]:
        raise NotImplementedError(mode)
    best_metric = None
    best_line = None
    for d in ds:
        if best_metric is None or (mode == "max" and best_metric < d[fieldname]) or (mode == "min" and best_metric > d[fieldname]):
            best_metric = d[fieldname]
            best_line = d
    return best_line

def max_by(ds, fieldname):
    return best_by(ds, fieldname, mode="max")

def min_by(ds, fieldname):
    return best_by(ds, fieldname, mode="min")

SEED = "42"
NVARS = "10"
NVALS = "7"
Z_SEED = "42"
SEQ_LEN = "10"
NBRANCHES = "3"
X_SEED = "42"
MEAN = "0.0"
STD = "1.0"
BATCH_SIZE = "64"
GPU_BATCH_SIZE = "64"
LR ="1e-5"
NLAYER = "12"
TRAIN_STEPS = "5000"
SAMPLE_SEED = "42"

data = dict()

def extract_runs(path, ns=(1, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000), lbds=("0", "0.001", "0.01", "0.1", "1","10", "100", "1000")):
    xs = []
    ys = []
    zs = []
    fs = []
    cis = []
    for N in ns:
        if N > 1:
            points = []
            for MULTI_LAMBDA in lbds:
                OUT_DIR=f"{os.environ['BLU_ARTIFACTS2']}" + path.format(N=N, MULTI_LAMBDA=MULTI_LAMBDA)
                with open(os.path.join(OUT_DIR, "log.jsonl")) as f:
                    log = [json.loads(line) for line in f]
                best_by_dev = min_by(log, "dev_loss")
                points.append((MULTI_LAMBDA, best_by_dev[f"{PLOT}_loss"], OUT_DIR)) #type:ignore
            point = min(points, key=lambda x: x[1])
        else:
            OUT_DIR=f"{os.environ['BLU_ARTIFACTS2']}" + path.format(N=100, MULTI_LAMBDA=1000)
            with open(os.path.join(OUT_DIR, "log.jsonl")) as f:
                log = [json.loads(line) for line in f]
            best_by_dev = min_by(log, "step")
            point = (1000, best_by_dev[f"{PLOT}_loss"], OUT_DIR) # type:ignore
        if CI:
            if N > 1:
                with open(os.path.join(point[2], "checkpoint-early-stopping", "bootstrap.json"), "r") as f:
                    bootstrap = json.load(f)
                cis.append((bootstrap[f"{PLOT}_loss_low"]- theoretical_min, bootstrap[f"{PLOT}_loss_high"]- theoretical_min))
            else:
                cis.append((point[1]- theoretical_min - 0.0001, point[1]- theoretical_min + 0.0001,))
        xs.append(N)
        ys.append(point[1] - theoretical_min)
        zs.append(float(point[0]))
        fs.append(point[2])
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    fs = np.array(fs)
    return xs, ys, zs, fs, cis

# xs, ys, zs, fs, cis = extract_runs(f"/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/24_888/10000/{MEAN}/{STD}/{{N}}/{{MULTI_LAMBDA}}")
# data["Decoder"] = dict()
# data["Decoder"]["x"] = xs
# data["Decoder"]["y"] = ys
# data["Decoder"]["lambda"] = zs
# data["Decoder"]["best_models"] = fs
# data["Decoder"]["cis"] = cis

xs, ys, zs, fs, cis = extract_runs(f"/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{{N}}/{{MULTI_LAMBDA}}")
data["ALBO-b"] = dict()
data["ALBO-b"]["x"] = xs
data["ALBO-b"]["y"] = ys
data["ALBO-b"]["lambda"] = zs
data["ALBO-b"]["best_models"] = fs
data["ALBO-b"]["cis"] = cis

# xs, ys, zs, fs, cis = extract_runs(f"/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_with_uniform_noise/0.5/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{{N}}/{{MULTI_LAMBDA}}/0.5")
# data["ALBO-bm.5"] = dict()
# data["ALBO-bm.5"]["x"] = xs
# data["ALBO-bm.5"]["y"] = ys
# data["ALBO-bm.5"]["lambda"] = zs
# data["ALBO-bm.5"]["best_models"] = fs
# data["ALBO-bm.5"]["cis"] = cis

xs, ys, zs, fs, cis = extract_runs(f"/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{{N}}/{{MULTI_LAMBDA}}")
data["ALBO"] = dict()
data["ALBO"]["x"] = xs
data["ALBO"]["y"] = ys
data["ALBO"]["lambda"] = zs
data["ALBO"]["best_models"] = fs
data["ALBO"]["cis"] = cis



xs, ys, zs, fs, cis = extract_runs(f"/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{{N}}/{{MULTI_LAMBDA}}")
data["No-Inference"] = dict()
data["No-Inference"]["x"] = xs
data["No-Inference"]["y"] = ys
data["No-Inference"]["lambda"] = zs
data["No-Inference"]["best_models"] = fs
data["No-Inference"]["cis"] = cis

xs, ys, zs, fs, cis = extract_runs(f"/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_mult_gate_post_scale_gate/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{{N}}/{{MULTI_LAMBDA}}")
data["Mult-Gate"] = dict()
data["Mult-Gate"]["x"] = xs
data["Mult-Gate"]["y"] = ys
data["Mult-Gate"]["lambda"] = zs
data["Mult-Gate"]["best_models"] = fs
data["Mult-Gate"]["cis"] = cis

for k, gpu_batch_size in [(10, 64), (100, 64), (500, 64), (1000, 16)]:
    xs, ys, zs, fs, cis = extract_runs(f"/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_shared_sample/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{gpu_batch_size}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{{N}}/{{MULTI_LAMBDA}}/{k}")
    data[f"Sample-b-{k}"] = dict()
    data[f"Sample-b-{k}"]["x"] = xs
    data[f"Sample-b-{k}"]["y"] = ys
    data[f"Sample-b-{k}"]["lambda"] = zs
    data[f"Sample-b-{k}"]["best_models"] = fs
    data[f"Sample-b-{k}"]["cis"] = cis

for k, gpu_batch_size in [(500, 64)]:
    xs, ys, zs, fs, cis = extract_runs(f"/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi_post_scale_gate_full_sample/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{gpu_batch_size}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{{N}}/{{MULTI_LAMBDA}}/{k}",
                                  ns=[1, 100, 250, 500, 1000],
                                  lbds=[0.1,1,10,100,1000])
    data[f"Sample-{k}"] = dict()
    data[f"Sample-{k}"]["x"] = xs
    data[f"Sample-{k}"]["y"] = ys
    data[f"Sample-{k}"]["lambda"] = zs
    data[f"Sample-{k}"]["best_models"] = fs
    data[f"Sample-{k}"]["cis"] = cis

# NLAYER = "24"
# TRAIN_STEPS = "10000"
# xs = []
# ys = []
# zs = []
# fs  = []
# for N in [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]:
#     points = []
#     for MULTI_LAMBDA in ["0", "0.001", "0.01", "0.1", "1","10", "100", "1000"]:
#         OUT_DIR=f"{os.environ['BLU_ARTIFACTS2']}/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}_888/{TRAIN_STEPS}/{MEAN}/{STD}/{N}/{MULTI_LAMBDA}"
#         with open(os.path.join(OUT_DIR, "log.jsonl")) as f:
#             log = [json.loads(line) for line in f]
#         best_by_dev = min_by(log, "dev_loss")
#         points.append((MULTI_LAMBDA, best_by_dev["dev_loss"], OUT_DIR)) #type:ignore
#     point = min(points, key=lambda x: x[1])
#     xs.append(N)
#     ys.append(point[1] - theoretical_min)
#     zs.append(float(point[0]))
#     fs.append(point[2])
# data["Regular-l24-h888"] = dict()
# data["Regular-l24-h888"]["x"] = xs
# data["Regular-l24-h888"]["y"] = ys
# data["Regular-l24-h888"]["lambda"] = zs
# data["Regular-l24-h888"]["best_models"] = fs


ARCHS = ["ALBO", "ALBO-b", "No-Inference", "Mult-Gate"]
# ARCHS = ["ALBO", "ALBO-b", "No-Inference", "Mult-Gate", "Sample-b-10", "Sample-b-100", "Sample-b-500", "Sample-b-1000", "Sample-500"]
COLORS = {
"ALBO": "blue",
"ALBO-b": "blue",
"No-Inference": "grey",
"Mult-Gate": "cyan",
"Sample-b-10": "lightcoral",
"Sample-b-100": "indianred",
"Sample-b-500": "brown",
"Sample-b-1000": "firebrick",
"Sample-500": "yellow",
"Decoder": "green",
"ALBO-bm.5": "turquoise"
}
def plot_results():
    for arch in ARCHS:
        print(arch, [data[arch]["y"][i] + theoretical_min for i in [2,5,8]])
        plt.plot(data[arch]["x"], data[arch]["y"], label=arch, color=COLORS[arch], linewidth=0.3)
    plt.xscale("log")
    plt.ylim(0,1)
    plt.legend()
    plt.ylabel("best_dev_kl")
    plt.xlabel("data_size")
    plt.savefig("results.png", dpi=600)
    plt.clf()

def plot_lambda():
    for arch in ARCHS:
        plt.plot(data[arch]["x"], data[arch]["lambda"], label=arch, color=COLORS[arch], linewidth=0.3)
        plt.yscale("log")
        plt.xscale("log")

        plt.legend()
        plt.ylabel("best_lambda")
        plt.xlabel("data_size")
        plt.savefig(f"lambda.{arch}.png", dpi=600)
        plt.clf()

def save_best_models():
    with open("best_models.txt", "w") as ofile:
        for arch in ARCHS:
            print(arch, file=ofile)
            for x, f in zip(data[arch]["x"].tolist(), data[arch]["best_models"].tolist()):
                print(f"{x}\t{f}", file=ofile)
            print("", file=ofile)
    with open("best_models_path_only.txt", "w") as ofile:
        for arch in ARCHS:
            print(arch, file=ofile)
            for x, f in zip(data[arch]["x"].tolist(), data[arch]["best_models"].tolist()):
                print(f"{f}", file=ofile)
            print("", file=ofile)
    with open("best_models.json", "w") as ofile:
        d = dict()
        for arch in ARCHS:
            d[arch] = dict()
            for x, f in zip(data[arch]["x"].tolist(), data[arch]["best_models"].tolist()):
                d[arch][x] = f
        json.dump(d, ofile, indent=4)

def plot_results_with_ci():
    def first(some_list):
        return [a[0] for a in some_list]
    def second(some_list):
        return [a[1] for a in some_list]

    for arch in ARCHS:
        plt.plot(data[arch]["x"], data[arch]["y"], label=arch, color=COLORS[arch], linewidth=0.3)
        plt.fill_between(data[arch]["x"], first(data[arch]["cis"]), second(data[arch]["cis"]), color=COLORS[arch], alpha=.2, linewidth=0.3) # type:ignore
    plt.xscale("log")
    plt.ylim(0,1)
    plt.legend()
    plt.ylabel("best_dev_kl")
    plt.xlabel("data_size")
    plt.savefig("results_ci.png", dpi=600)
    plt.clf()

plot_results()
# # plot_lambda()
# # save_best_models()
# if CI:
#     plot_results_with_ci()
