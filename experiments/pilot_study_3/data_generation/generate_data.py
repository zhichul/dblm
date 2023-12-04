from __future__ import annotations
import argparse
import code
import json
import math
import os
import random

import torch
import tqdm
from dblm.core.inferencers import belief_propagation
from dblm.core.modeling import constants, markov_networks
from dblm.core.samplers import tree
from dblm.utils import seeding
import dblm.experiments.pilot_study_3.distributions as distributions

def parse_args():
    parser = argparse.ArgumentParser()
    # z
    parser.add_argument("--nvars", type=int, required=True)
    parser.add_argument("--nvals", type=int, required=True)
    parser.add_argument("--initializer_mean", type=float, required=True)
    parser.add_argument("--initializer_std", type=float, required=True)
    parser.add_argument("--initializer_min", type=float, required=True)
    parser.add_argument("--initializer_max", type=float, required=True)
    parser.add_argument("--z_model_seed", type=int, required=True)

    # x
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--n_branches", type=int, required=True)
    parser.add_argument("--x_model_seed", type=int, required=True)
    parser.add_argument("--sample_seed", type=int, required=True)

    # N
    parser.add_argument("--N", type=int, required=True)

    # save
    parser.add_argument("--save_name", type=str, required=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    indices = list(range(args.nvars))

    # seed the model
    seeding.seed(args.z_model_seed)
    z_model_ = markov_networks.TreeMRF(args.nvars, args.nvals, constants.GaussianInitializer(args.initializer_mean,
                                                                                            args.initializer_std,
                                                                                            args.initializer_min,
                                                                                            args.initializer_max)) # type:ignore
    z_model = z_model_.to_factor_graph_model() # type:ignore
    z_samples = tree.BatchTreeSampler().sample(args.N, z_model.to_factor_graph_model()).tolist()

    # seed the samples
    random.seed(args.sample_seed)
    sample_seeds = [random.random() for _ in range(args.N)]
    index_samples: list[tuple[int,...]] = []
    for ss in tqdm.tqdm(sample_seeds):
        index_samples.append(distributions.sample_indices(args.seq_len, args.n_branches, indices=indices, model_seed=args.x_model_seed, sample_seed=ss)[0])

    bp = belief_propagation.FactorGraphBeliefPropagation()
    train_z = []
    train_x = []
    train_inf = []
    dev_z = []
    dev_x = []
    dev_inf = []
    test_z = []
    test_x = []
    test_inf = []
    for z_sample, index_sample in tqdm.tqdm(list(zip(z_samples, index_samples))):
        z = z_sample
        x = [idx * args.nvals + z_sample[idx] for idx in index_sample]
        inf = []
        for j in range(args.seq_len+1):
            inference_results = bp.inference(z_model, observation={idx:torch.tensor([z_sample[idx]]) for idx in index_sample[:j]}, query=list(range(args.nvars)), allow_query_observation=True, iterations= 2 * args.nvars)
            inf.append(sum((q.log_potential_table().tolist()[0] if j > 0 else q.log_potential_table().tolist()  # first iteration does not have batch size
                            for q in inference_results.query_marginals), []))
        hash_value = hash(tuple(z)) % 100
        if hash_value < 70:
            train_z.append(z)
            train_x.append(x)
            train_inf.append(inf)
        elif hash_value < 85:
            dev_z.append(z)
            dev_x.append(x)
            dev_inf.append(inf)
        else:
            test_z.append(z)
            test_x.append(x)
            test_inf.append(inf)
    train_x = torch.tensor(train_x)
    train_z = torch.tensor(train_z)
    train_inf = torch.tensor(train_inf)
    dev_x = torch.tensor(dev_x)
    dev_z = torch.tensor(dev_z)
    dev_inf = torch.tensor(dev_inf)
    test_x = torch.tensor(test_x)
    test_z = torch.tensor(test_z)
    test_inf = torch.tensor(test_inf)
    train_obj = {
        "x": train_x,
        "z": train_z,
        "inf": train_inf,
    }
    dev_obj = {
        "x": dev_x,
        "z": dev_z,
        "inf": dev_inf,
    }
    test_obj = {
        "x": test_x,
        "z": test_z,
        "inf": test_inf,
    }
    torch.save(train_obj, os.path.join(args.save_name, "train.bin"))
    torch.save(dev_obj, os.path.join(args.save_name, "dev.bin"))
    torch.save(test_obj, os.path.join(args.save_name, "test.bin"))

    with open(os.path.join(args.save_name, "args.json"), "w") as f:
        print(json.dumps(
            vars(args)
        ,indent=4), file=f)

    z_model_.save(os.path.join(args.save_name, "z_model"))
    with open(os.path.join(args.save_name, "meta.json"), "w") as g:
        lp = z_model.to_probability_table().log_probability_table().reshape(-1)
        print(json.dumps(
            {"z_entropy":(-lp[lp > -math.inf] * lp[lp > -math.inf].exp()).sum().item()}
        ,indent=4), file=g)

if __name__ == "__main__":
    with torch.no_grad():
        main()
