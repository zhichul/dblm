
import argparse
import collections
import csv
import json
import os

import torch
from dblm.core.modeling import constants, factory
from dblm.core.samplers import ancestral
from dblm.experiments.pilot_study_1 import tree_mrf_noiseless_emission, tree_mrf_noisy_emission

from dblm.utils import seeding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--z0_num_variables", type=int, required=True)
    parser.add_argument("--z0_num_values", type=int, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--z0_noise_weight", type=float, required=False)
    parser.add_argument("--zt_noise_weight", type=float, required=False)
    parser.add_argument("--model_type", choices=["interleaved", "nested"], required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_predefined_model", type=str, default=None, required=False)
    parser.add_argument("--z0_initializer_type", type=str, default=None, required=False)
    parser.add_argument("--z0_initializer_min", type=float, default=None, required=False)
    parser.add_argument("--z0_initializer_max", type=float, default=None, required=False)
    parser.add_argument("--z0_initializer_mean", type=float, default=None, required=False)
    parser.add_argument("--z0_initializer_std", type=float, default=None, required=False)
    parser.add_argument("--zt_initializer_type", type=str, default=None, required=False)
    parser.add_argument("--zt_initializer_min", type=float, default=None, required=False)
    parser.add_argument("--zt_initializer_max", type=float, default=None, required=False)
    parser.add_argument("--zt_initializer_mean", type=float, default=None, required=False)
    parser.add_argument("--zt_initializer_std", type=float, default=None, required=False)
    return parser.parse_args()

def get_initializer(initializer_type, mean=None, std=None, min=None, max=None):
    initializer = constants.TensorInitializer.UNIFORM
    if initializer_type is not None:
        if initializer_type == "uniform":
            initializer = constants.UniformlyRandomInitializer(min, max) # type:ignore
        elif initializer_type == "gaussian":
            initializer = constants.GaussianInitializer(mean, std, min, max) # type:ignore
    return initializer

def main():
    args = parse_args()

    seeding.seed(args.seed)
    z0_initializer = get_initializer(args.z0_initializer_type, args.z0_initializer_mean, args.z0_initializer_std, args.z0_initializer_min, args.z0_initializer_max)
    zt_initializer = get_initializer(args.zt_initializer_type, args.zt_initializer_mean, args.zt_initializer_std, args.zt_initializer_min, args.zt_initializer_max)

    if args.z0_noise_weight is not None:
        model = tree_mrf_noisy_emission.TreeMrfNoisyEmission(
            args.z0_num_variables,
            args.z0_num_values,
            z0_initializer, # type:ignore
            (1-args.z0_noise_weight, args.z0_noise_weight),
            args.sequence_length,
            zt_initializer, # type:ignore
            (1-args.zt_noise_weight, args.zt_noise_weight),
            seed=args.seed)
    else:
        model = tree_mrf_noiseless_emission.TreeMrfNoiselessEmission(
            args.z0_num_variables,
            args.z0_num_values,
            z0_initializer, # type:ignore
            args.sequence_length,
            zt_initializer, # type:ignore
            seed=args.seed)
    if args.use_predefined_model is not None:
        if args.z0_noise_weight is not None:
            model_dir = args.use_predefined_model
            model = tree_mrf_noisy_emission.TreeMrfNoisyEmission.load(model_dir)
        else:
            model_dir = args.use_predefined_model
            model = tree_mrf_noiseless_emission.TreeMrfNoiselessEmission.load(model_dir)
    else:
        model_dir = os.path.join(args.output_dir, "ground_truth_models")
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
        with open(os.path.join(model_dir, "sampling.json"), "w") as f:
            json.dump({
                "samples": args.samples,
                "model_type": args.model_type,
                "seed": args.seed,
            }, f, indent=4)
    sampler = ancestral.BatchAncestralSamplerWithPotentialTables()
    sample_model = model.bayes_net_model if args.model_type == "interleaved" else model.nested_model
    data_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(data_dir, exist_ok=True)
    seeding.seed(args.seed)
    with open(os.path.join(data_dir, "sample.csv"), "w") as f:
        counter = 0
        writer = csv.writer(f)
        while True:
            data = sampler.sample(10000, sample_model) # type:ignore
            for row in data:
                if counter >= args.samples:
                    return
                writer.writerow(row.tolist())
                counter += 1


if __name__ == "__main__":
    main()
