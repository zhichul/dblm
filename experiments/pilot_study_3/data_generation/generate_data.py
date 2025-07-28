from __future__ import annotations
import argparse
import json
import math
import os
import random

import torch
import tqdm
from dblm.core.inferencers import belief_propagation
from dblm.core.modeling import constants, markov_networks
from dblm.core.samplers import tree
from dblm.rva.discrete import tree_belief_propagation
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
    parser.add_argument("--generate_conditional_marginals", action="store_true")
    parser.add_argument("--generate_inf_samples", type=int, required=False)

    # x
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--n_branches", type=int, required=True)
    parser.add_argument("--x_model_seed", type=int, required=True)
    parser.add_argument("--sample_seed", type=int, required=True)

    # N
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--segments", type=int, nargs="+", required=False)
    parser.add_argument("--segment_length", type=int, default=10000, required=False)

    # other flags
    parser.add_argument("--never_top", action="store_true", required=False)

    # save
    parser.add_argument("--save_name", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    RAND_CHECK_PROB = 0.001
    mismatch_thresh = 1e-6
    args = parse_args()
    SEGMENT_LENGTH=args.segment_length
    indices = list(range(args.nvars))
    nvals =  [args.nvals for _ in range(args.nvars)]


    # seed the model
    seeding.seed(args.z_model_seed)
    z_model_ = markov_networks.TreeMRF(args.nvars, args.nvals, constants.GaussianInitializer(args.initializer_mean,
                                                                                            args.initializer_std,
                                                                                            args.initializer_min,
                                                                                            args.initializer_max)) # type:ignore
    z_model = z_model_.to_factor_graph_model() # type:ignore
    z_samples = tree.BatchTreeSampler().sample(args.N, z_model.to_factor_graph_model()).tolist()

    # encode as fast tree bp
    tbp = tree_belief_propagation.TreeBeliefPropagation()
    factor_fns = [f.log_potential_table() for f in z_model.factor_functions()]
    factor_vars = z_model.factor_variables()
    if not all(list(fv) == sorted(fv) for fv in factor_vars):
        raise ValueError(f"{factor_vars} factors must have variable ordered small to large")
    adjacency_matrix = torch.zeros((args.nvars, len(factor_fns)), dtype=torch.long)
    for index_factor, (va, vb) in enumerate(factor_vars):
        adjacency_matrix[va][index_factor] = 1
        adjacency_matrix[vb][index_factor] = 1

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
    train_cond_inf = []
    train_inf_samples = []
    dev_z = []
    dev_x = []
    dev_inf = []
    dev_cond_inf = []
    dev_inf_samples = []
    test_z = []
    test_x = []
    test_inf = []
    test_cond_inf = []
    test_inf_samples = []
    count = 0
    segments = set(args.segments)
    for z_sample, index_sample in tqdm.tqdm(list(zip(z_samples, index_samples))):
        if len(segments) > 0 and count // SEGMENT_LENGTH not in segments:
            count += 1
            continue
        if not args.never_top:
            z = z_sample
            x = [idx * args.nvals + z_sample[idx] for idx in index_sample]
        else:
            z = [None] * args.nvars
            x = []
        inf = []
        cond_inf = []
        inf_samples =  []
        for j in range(args.seq_len+1):
            observation = {idx:torch.tensor([z_sample[idx]]) for idx in index_sample[:j]}
            # # ***** Slow implementation of Tree BP ******
            # inference_results = bp.inference(z_model, observation=observation, query=list(range(args.nvars)), allow_query_observation=True, iterations= 2 * args.nvars)
            # inf.append(sum((q.log_potential_table().tolist()[0] if j > 0 else q.log_potential_table().tolist()  # first iteration does not have batch size
            #                 for q in inference_results.query_marginals), []))
            # # ***** Fast implementation of Tree BP ******
            # # Manually setup the new factor graph given observations
            # outer_observation_factors = [torch.zeros(args.nvals).fill_(-math.inf) for _ in range(len(observation))]
            # outer_adjacency_columns = torch.zeros(args.nvars, len(observation), dtype=torch.long)
            # for oind, (ovar, oval) in enumerate(observation.items()):
            #     outer_observation_factors[oind][oval] = 0.0
            #     outer_adjacency_columns[ovar, oind] = 1
            # outer_adj = torch.cat((adjacency_matrix, outer_adjacency_columns), dim=1)
            # outer_inference_results = tbp.infer(outer_adj, nvals, factor_fns + outer_observation_factors)
            if args.generate_inf_samples is None:
                outer_inference_results = tbp.infer(adjacency_matrix, nvals, factor_fns, observations=observation)
                outer_marginals = tbp.marginals(outer_inference_results)
                infj = sum((q.view(-1).tolist() for q in outer_marginals), [])
                inf.append(infj)
            if args.never_top and j < args.seq_len:
                if args.generate_inf_samples is not None: raise ValueError()
                # assume we have inference results
                idxj = index_sample[j]
                weights = outer_marginals[idxj].exp().view(-1).tolist()
                top_i = outer_marginals[idxj].view(-1).argmax(-1).item()
                weights.pop(top_i)
                choices = [i for i in range(args.nvals) if i != top_i]
                z[idxj] = random.choices(choices, weights)[0]
                x.append(idxj * args.nvals + z[idxj])
            # # ***** Check Match *****
            # if random.random() < RAND_CHECK_PROB: # random check
            #     inference_results_check = bp.inference(z_model, observation=observation, query=list(range(args.nvars)), allow_query_observation=True, iterations= 2 * args.nvars)
            #     marginal_check = sum((q.log_potential_table().tolist()[0] if j > 0 else q.log_potential_table().tolist() for q in inference_results_check.query_marginals), [])
            #     if (torch.tensor(inf[-1]) - torch.tensor(marginal_check)).abs().max() > mismatch_thresh:
            #         print("Uh Oh! Mismatch between two BP algos outer")
            #         breakpoint()
            #     # else:
            #     #     print("matched outer!")
            cond_inf_j = []
            if args.generate_conditional_marginals:
                observation = {idx:torch.tensor([z_sample[idx]]) for idx in index_sample[:j]}
                for ivar in range(args.nvars):
                    for ival in range(args.nvals):
                        if ivar in observation and ival != z_sample[ivar]:
                            cond_inf_j.append([0.0 for _ in range(args.nvals * args.nvars)])
                        elif ivar in observation and ival == z_sample[ivar]:
                            cond_inf_j.append(infj)
                        else:
                            observation[ivar] = torch.tensor([ival])
                            # observation_factors = [torch.zeros(args.nvals).fill_(-math.inf) for _ in range(len(observation))]
                            # adjacency_columns = torch.zeros(args.nvars, len(observation), dtype=torch.long)
                            # for oind, (ovar, oval) in enumerate(observation.items()):
                            #     observation_factors[oind][oval] = 0.0
                            #     adjacency_columns[ovar, oind] = 1
                            # adj = torch.cat((adjacency_matrix, adjacency_columns), dim=1)
                            # cond_inference_results = tbp.infer(adj, nvals, factor_fns + observation_factors)
                            cond_inference_results = tbp.infer(adjacency_matrix, nvals, factor_fns, observations=observation)
                            marginals = tbp.marginals(cond_inference_results)
                            cond_inf_j.append(sum((q.view(-1).tolist() for q in marginals), []))
                            # if random.random() < RAND_CHECK_PROB: # random check
                            #     cond_inference_results_check = bp.inference(z_model, observation=observation, query=list(range(args.nvars)), allow_query_observation=True, iterations= 2 * args.nvars)
                            #     marginal_check = sum((q.log_potential_table().tolist()[0] for q in cond_inference_results_check.query_marginals), [])
                            #     if (torch.tensor(cond_inf_j[-1]) - torch.tensor(marginal_check)).abs().max() > mismatch_thresh:
                            #         print("Uh Oh! Mismatch between two BP algos inner")
                            #         breakpoint()
                            #     # else:
                            #     #     print("matched inner!")
                            del observation[ivar]

                            # breakpoint()
                cond_inf.append(cond_inf_j)
            if args.generate_inf_samples is not None:
                observation = {idx:torch.tensor([z_sample[idx]]) for idx in index_sample[:j]}
                samples = tbp.sample(adjacency_matrix, nvals, factor_fns, args.generate_inf_samples, observation)
                inf_samples.append(samples.detach().cpu())
        if len(inf_samples) > 0:
            inf_samples = torch.stack(inf_samples)
        hash_value = hash(tuple(z)) % 100
        if hash_value < 70:
            train_z.append(z)
            train_x.append(x)
            train_inf.append(inf)
            train_cond_inf.append(cond_inf)
            train_inf_samples.append(inf_samples)
        elif hash_value < 85:
            dev_z.append(z)
            dev_x.append(x)
            dev_inf.append(inf)
            dev_cond_inf.append(cond_inf)
            dev_inf_samples.append(inf_samples)
        else:
            test_z.append(z)
            test_x.append(x)
            test_inf.append(inf)
            test_cond_inf.append(cond_inf)
            test_inf_samples.append(inf_samples)
        count += 1
        if count % SEGMENT_LENGTH == 0:
            train_obj = {
                "x": torch.tensor(train_x),
                "z": torch.tensor(train_z),
            }
            dev_obj = {
                "x": torch.tensor(dev_x),
                "z": torch.tensor(dev_z),
            }
            test_obj = {
                "x": torch.tensor(test_x),
                "z": torch.tensor(test_z),
            }
            if args.generate_conditional_marginals:
                train_obj["cond_inf"] = torch.tensor(train_cond_inf)
                dev_obj["cond_inf"] = torch.tensor(dev_cond_inf)
                test_obj["cond_inf"] = torch.tensor(test_cond_inf)
            if args.generate_inf_samples is not None:
                train_obj["inf_samples"] = torch.stack(train_inf_samples).to(torch.int8)
                dev_obj["inf_samples"] = torch.stack(dev_inf_samples).to(torch.int8)
                test_obj["inf_samples"] = torch.stack(test_inf_samples).to(torch.int8)
            else:
                train_obj["inf"] = torch.tensor(train_inf)
                dev_obj["inf"] = torch.tensor(dev_inf)
                test_obj["inf"] = torch.tensor(test_inf)
            torch.save(train_obj, os.path.join(args.save_name, f"train.{count // SEGMENT_LENGTH-1}.bin"))
            torch.save(dev_obj, os.path.join(args.save_name, f"dev.{count // SEGMENT_LENGTH-1}.bin"))
            torch.save(test_obj, os.path.join(args.save_name, f"test.{count // SEGMENT_LENGTH-1}.bin"))
            train_z = []
            train_x = []
            train_inf = []
            train_cond_inf = []
            train_inf_samples = []
            dev_z = []
            dev_x = []
            dev_inf = []
            dev_cond_inf = []
            dev_inf_samples = []
            test_z = []
            test_x = []
            test_inf = []
            test_cond_inf = []
            test_inf_samples = []

    # train_x = torch.tensor(train_x)
    # train_z = torch.tensor(train_z)
    # train_inf = torch.tensor(train_inf)
    # train_cond_inf = torch.tensor(train_cond_inf)
    # dev_x = torch.tensor(dev_x)
    # dev_z = torch.tensor(dev_z)
    # dev_inf = torch.tensor(dev_inf)
    # dev_cond_inf = torch.tensor(dev_cond_inf)
    # test_x = torch.tensor(test_x)
    # test_z = torch.tensor(test_z)
    # test_inf = torch.tensor(test_inf)
    # test_cond_inf = torch.tensor(test_cond_inf)
    # train_obj = {
    #     "x": train_x,
    #     "z": train_z,
    #     "inf": train_inf,
    # }
    # dev_obj = {
    #     "x": dev_x,
    #     "z": dev_z,
    #     "inf": dev_inf,
    # }
    # test_obj = {
    #     "x": test_x,
    #     "z": test_z,
    #     "inf": test_inf,
    # }
    # if args.generate_conditional_marginals:
    #     train_obj["cond_inf"] = train_cond_inf
    #     dev_obj["cond_inf"] = dev_cond_inf
    #     test_obj["cond_inf"] = test_cond_inf
    # torch.save(train_obj, os.path.join(args.save_name, "train.bin"))
    # torch.save(dev_obj, os.path.join(args.save_name, "dev.bin"))
    # torch.save(test_obj, os.path.join(args.save_name, "test.bin"))

if __name__ == "__main__":
    with torch.no_grad():
        main()
