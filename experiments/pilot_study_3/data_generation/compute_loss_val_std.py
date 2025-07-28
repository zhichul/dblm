import code
import json
import math
import os
import sys

import torch
import tqdm
import wandb

wandb.login()

for dir in ["../data/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000"]:
    with open(os.path.join(dir, "args.json")) as f:
        config = json.load(f)
        seq_len = config["seq_len"]
        nvars = config["nvars"]
        nvals = config["nvals"]
        n_branches = config["n_branches"]

    dev = torch.load(os.path.join(dir, "dev.bin"))
    logpz = dev["inf"][:,:-1,:].gather(dim=-1,index=dev["x"].unsqueeze(-1)).squeeze(-1)
    for N in tqdm.tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]):
        dist = []
        for n in range(N+1):
            logpn = math.log(math.comb(N,n)) + n * logpz + torch.log1p(-logpz.exp()) * (N-n)
            dist.append((logpn, torch.tensor(math.log(-math.log(0.000001 + 0.999993 * n/N)))))
        mean = sum((x + y).exp() for x,y in dist)
        variance = sum((x + 2 * (y.exp()-mean).abs().log()).exp() for x, y in dist)
        print(mean.mean(), variance.mean() / variance.numel(), (variance.mean() / variance.numel()) ** 0.5) # type:ignore
        pmean = mean.mean(dim=0) # type:ignore
        pstd = (variance.mean(dim=0) / variance.size(0)) ** 0.5 # type:ignore

        run = wandb.init(
            # Set the project where this run will be logged
            name=f"sample-{N}-mean",
            project="dblm",
            group=f"pilot_study_3/finetune-data-multi-factored") #type:ignore
        for i in range(10): #type:ignore
            run.log({"loss_val": pmean[i]}, step=i+1) #type:ignore
        run.finish() #type:ignore
        run = wandb.init(
            # Set the project where this run will be logged
            name=f"sample-{N}-mean+1std",
            project="dblm",
            group=f"pilot_study_3/finetune-data-multi-factored") #type:ignore
        for i in range(10): #type:ignore
            run.log({"loss_val": pmean[i] + pstd[i]}, step=i+1) #type:ignore
        run.finish() #type:ignore

