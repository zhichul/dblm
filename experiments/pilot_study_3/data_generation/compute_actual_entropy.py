import code
import json
import math
import os
import sys

import torch


for dir in sys.argv[1:]:
    with open(os.path.join(dir, "args.json")) as f:
        config = json.load(f)
        seq_len = config["seq_len"]
        nvars = config["nvars"]
        nvals = config["nvals"]
        n_branches = config["n_branches"]
    log_px_given_z = -math.log(n_branches ** (seq_len - n_branches) * math.factorial(n_branches))
    dev = torch.load(os.path.join(dir, "dev.bin"))
    train = torch.load(os.path.join(dir, "train.bin"))
    test = torch.load(os.path.join(dir, "test.bin"))
    def compute_cross_entropy(inputs, seq_len):
        logpz = inputs["inf"][:,:-1,:].gather(dim=-1, index=(inputs["x"]).unsqueeze(-1)).squeeze(-1).sum(-1)
        ce = -((logpz + log_px_given_z) / seq_len).mean()
        return ce.item()

    with open(os.path.join(dir, "cross_entropy.json"), "wt") as f:
        json.dump({
            "train": compute_cross_entropy(train, nvars, nvals, seq_len),
            "dev": compute_cross_entropy(dev, nvars, nvals, seq_len),
            "test": compute_cross_entropy(test, nvars, nvals, seq_len),
        },f,indent=4)