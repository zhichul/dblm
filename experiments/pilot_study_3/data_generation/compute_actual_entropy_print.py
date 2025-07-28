import code
import json
import math
import os
import sys
import numpy as np

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
    joint = {key: torch.cat([train[key], dev[key], test[key]], dim=0) for key in train}
    def compute_cross_entropy(inputs, seq_len):
        logpz = inputs["inf"][:,:-1,:].gather(dim=-1, index=(inputs["x"]).unsqueeze(-1)).squeeze(-1).sum(-1)
        ce = -((logpz + log_px_given_z) / seq_len).mean()
        return ce.item()
    def bootstrap_cross_entropy_variance(inputs, seq_len, n=100):
        ces = []
        for _ in range(n):
            logpz = inputs["inf"][:,:-1,:].gather(dim=-1, index=(inputs["x"]).unsqueeze(-1)).squeeze(-1).sum(-1)
            inds = torch.randint(0, logpz.size(0), (logpz.size(0),))
            breakpoint()
            logpz = logpz[inds]
            ce = -((logpz + log_px_given_z) / seq_len).mean()
            ces.append(ce.item())
        return np.array(ces).std()
    print(
        json.dumps({
            "train": compute_cross_entropy(train, seq_len),
            "dev": compute_cross_entropy(dev, seq_len),
            "test": compute_cross_entropy(test, seq_len),
            "joint": compute_cross_entropy(joint, seq_len),
        },indent=4)
    )
    print(
        json.dumps({
            "train": bootstrap_cross_entropy_variance(train, seq_len),
            "dev": bootstrap_cross_entropy_variance(dev, seq_len),
            "test": bootstrap_cross_entropy_variance(test, seq_len),
            "joint": bootstrap_cross_entropy_variance(joint, seq_len),
        },indent=4)
   )
