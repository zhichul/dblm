import code
import json
import math
import os
import sys

import torch
import wandb


run = wandb.init(
    # Set the project where this run will be logged
    name="reference",
    project="dblm",
    group=f"pilot_study_3/finetune-data-multi-factored")

for dir in ["../data/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000"]:
    with open(os.path.join(dir, "args.json")) as f:
        config = json.load(f)
        seq_len = config["seq_len"]
        nvars = config["nvars"]
        nvals = config["nvals"]
        n_branches = config["n_branches"]

    dev = torch.load(os.path.join(dir, "dev.bin"))
    logpz = dev["inf"][:,:-1,:].gather(dim=-1, index=(dev["x"]).unsqueeze(-1)).squeeze(-1).mean(0)
    print(logpz)
    for i, p in enumerate(logpz):
        print(i+1, p)
        wandb.log({"loss_val": -p}, step=i+1)