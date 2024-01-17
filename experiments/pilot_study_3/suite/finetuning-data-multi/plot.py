import json
import math
import os
import matplotlib.pyplot as plt

theoretical_min = 2.516202449798584

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

xs = []
ys = []
zs = []
fs = []
for N in [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]:
    points = []
    for MULTI_LAMBDA in ["0", "0.001", "0.01", "0.1", "1","10", "100", "1000"]:
        OUT_DIR=f"{os.environ['BLU_ARTIFACTS2']}/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_with_inference_vary_data_multi/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{N}/{MULTI_LAMBDA}"
        with open(os.path.join(OUT_DIR, "log.jsonl")) as f:
            log = [json.loads(line) for line in f]
        best_by_dev = min_by(log, "dev_loss")
        points.append((MULTI_LAMBDA, best_by_dev["dev_loss"], OUT_DIR)) #type:ignore
    point = min(points, key=lambda x: x[1])
    xs.append(N)
    ys.append(point[1] - theoretical_min)
    zs.append(float(point[0]))
    fs.append(point[2])
data["Nested-Oracle"] = dict()
data["Nested-Oracle"]["x"] = xs
data["Nested-Oracle"]["y"] = ys
data["Nested-Oracle"]["lambda"] = zs
data["Nested-Oracle"]["best_models"] = fs

xs = []
ys = []
zs = []
fs = []
for N in [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]:
    points = []
    for MULTI_LAMBDA in ["0", "0.001", "0.01", "0.1", "1","10", "100", "1000"]:
        OUT_DIR=f"{os.environ['BLU_ARTIFACTS2']}/dblm/experiments/pilot_study_3/finetuned_seq2seq_models_vary_data_multi/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}/{TRAIN_STEPS}/{MEAN}/{STD}/{N}/{MULTI_LAMBDA}"
        with open(os.path.join(OUT_DIR, "log.jsonl")) as f:
            log = [json.loads(line) for line in f]
        best_by_dev = min_by(log, "dev_loss")
        points.append((MULTI_LAMBDA, best_by_dev["dev_loss"], OUT_DIR)) #type:ignore
    point = min(points, key=lambda x: x[1])
    xs.append(N)
    ys.append(point[1] - theoretical_min)
    zs.append(float(point[0]))
    fs.append(point[2])
data["No-Inference"] = dict()
data["No-Inference"]["x"] = xs
data["No-Inference"]["y"] = ys
data["No-Inference"]["lambda"] = zs
data["No-Inference"]["best_models"] = fs

NLAYER = "24"
TRAIN_STEPS = "10000"
xs = []
ys = []
zs = []
fs  = []
for N in [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]:
    points = []
    for MULTI_LAMBDA in ["0", "0.001", "0.01", "0.1", "1","10", "100", "1000"]:
        OUT_DIR=f"{os.environ['BLU_ARTIFACTS2']}/dblm/experiments/pilot_study_3/finetuned_decoder_models_vary_data_multi/{SEED}/{NVARS}/{NVALS}/{SEQ_LEN}/{NBRANCHES}/{X_SEED}/{BATCH_SIZE}/{GPU_BATCH_SIZE}/{LR}/{NLAYER}_888/{TRAIN_STEPS}/{MEAN}/{STD}/{N}/{MULTI_LAMBDA}"
        with open(os.path.join(OUT_DIR, "log.jsonl")) as f:
            log = [json.loads(line) for line in f]
        best_by_dev = min_by(log, "dev_loss")
        points.append((MULTI_LAMBDA, best_by_dev["dev_loss"], OUT_DIR)) #type:ignore
    point = min(points, key=lambda x: x[1])
    xs.append(N)
    ys.append(point[1] - theoretical_min)
    zs.append(float(point[0]))
    fs.append(point[2])
data["Regular-l24-h888"] = dict()
data["Regular-l24-h888"]["x"] = xs
data["Regular-l24-h888"]["y"] = ys
data["Regular-l24-h888"]["lambda"] = zs
data["Regular-l24-h888"]["best_models"] = fs

def plot_results():
    plt.plot(data["Nested-Oracle"]["x"], data["Nested-Oracle"]["y"], label="Nested-Oracle")
    plt.plot(data["No-Inference"]["x"], data["No-Inference"]["y"], label="No-Inference")
    plt.plot(data["Regular-l24-h888"]["x"], data["Regular-l24-h888"]["y"], label="Regular-l24-h888")

    plt.xscale("log")
    plt.ylim(0,0.4)
    plt.legend()
    plt.ylabel("best_dev_kl")
    plt.xlabel("data_size")
    plt.savefig("results.png", dpi=300)
    plt.clf()

def plot_lambda():
    plt.plot(data["Nested-Oracle"]["x"], data["Nested-Oracle"]["lambda"], label="Nested-Oracle")
    plt.plot(data["No-Inference"]["x"], data["No-Inference"]["lambda"], label="No-Inference")
    plt.plot(data["Regular-l24-h888"]["x"], data["Regular-l24-h888"]["lambda"], label="Regular-l24-h888")
    plt.yscale("log")
    plt.xscale("log")

    plt.legend()
    plt.ylabel("best_lambda")
    plt.xlabel("data_size")
    plt.savefig("lambda.png", dpi=300)
    plt.clf()

def save_best_models():
    with open("best_models.txt", "w") as ofile:
        for arch in ["Nested-Oracle", "No-Inference", "Regular-l24-h888"]:
            print(arch, file=ofile)
            for x, f in zip(data[arch]["x"], data[arch]["best_models"]):
                print(f"{x}\t{f}", file=ofile)
            print("", file=ofile)
    with open("best_models.json", "w") as ofile:
        d = dict()
        for arch in ["Nested-Oracle", "No-Inference", "Regular-l24-h888"]:
            d[arch] = dict()
            for x, f in zip(data[arch]["x"], data[arch]["best_models"]):
                d[arch][x] = f
        json.dump(d, ofile, indent=4)

plot_results()
plot_lambda()
save_best_models()

