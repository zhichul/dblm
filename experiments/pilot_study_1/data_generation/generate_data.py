
import argparse
import collections
import csv
import json
import os

import torch
from dblm.core.modeling import constants, factory
from dblm.core.samplers import ancestral

from dblm.utils import seeding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--z0_num_variables", type=int, required=True)
    parser.add_argument("--z0_num_values", type=int, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--noise_weight", type=float, required=True)
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

def main():
    args = parse_args()

    seeding.seed(args.seed)
    z0_initializer = constants.TensorInitializer.UNIFORM
    zt_initializer = constants.TensorInitializer.UNIFORM
    if args.z0_initializer_type is not None:
        if args.z0_initializer_type == "uniform":
            z0_initializer = constants.UniformlyRandomInitializer(args.z0_initializer_min, args.z0_initializer_max)
        elif args.z0_initializer_type == "gaussian":
            z0_initializer = constants.GaussianInitializer(args.z0_initializer_mean,
                                                                  args.z0_initializer_std,
                                                                  args.z0_initializer_min,
                                                                  args.z0_initializer_max)
    if args.zt_initializer_type is not None:
        if args.zt_initializer_type == "uniform":
            zt_initializer = constants.UniformlyRandomInitializer(args.zt_initializer_min, args.zt_initializer_max)
        elif args.zt_initializer_type == "gaussian":
            zt_initializer = constants.GaussianInitializer(args.zt_initializer_mean,
                                                                  args.zt_initializer_std,
                                                                  args.zt_initializer_min,
                                                                  args.zt_initializer_max)

    factor_graph_model, nested_model, bayes_net_model, name = factory.tree_mrf_with_term_frequency_based_transition_and_separate_noise_per_token(
        args.z0_num_variables,
        args.z0_num_values,
        z0_initializer, # type:ignore
        (1-args.noise_weight, args.noise_weight),
        args.sequence_length,
        zt_initializer, # type:ignore
        (1-args.noise_weight, args.noise_weight))
    if args.use_predefined_model is not None:
        model_dir = args.use_predefined_model
        factor_graph_model.load_state_dict(torch.load(os.path.join(model_dir, "factor_graph.pt")))
        bayes_net_model.load_state_dict(torch.load(os.path.join(model_dir, "bayes_net.pt")))
        nested_model.load_state_dict(torch.load(os.path.join(model_dir, "nested.pt")))
    else:
        model_dir = os.path.join(args.output_dir, "ground_truth_models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(factor_graph_model.state_dict(), os.path.join(model_dir, "factor_graph.pt"))
        torch.save(bayes_net_model.state_dict(), os.path.join(model_dir, "bayes_net.pt"))
        torch.save(nested_model.state_dict(), os.path.join(model_dir, "nested.pt"))
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({
                "z0_num_variables": args.z0_num_variables,
                "z0_num_values": args.z0_num_values,
                "sequence_length": args.sequence_length,
                "noise_weight": args.noise_weight,
                "samples": args.samples,
                "model_type": args.model_type,
                "seed": args.seed,
            }, f, indent=4)
        with open(os.path.join(model_dir, "mapping.json"), "w") as f:
            json.dump(collections.OrderedDict(
                (name, i) for i, name in enumerate(name)
            ), f, indent=4)
        with open(os.path.join(model_dir, "factor_variables.json"), "w") as f:
            json.dump(factor_graph_model.factor_variables(), f, indent=4)

    sampler = ancestral.BatchAncestralSamplerWithPotentialTables()
    sample_model = bayes_net_model if args.model_type == "interleaved" else nested_model
    data_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(data_dir, exist_ok=True)
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
