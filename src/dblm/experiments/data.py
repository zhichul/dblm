from __future__ import annotations
import code
import csv
import json
import math
import os
import re
import uuid

import torch

from dblm.core.modeling import bayesian_networks, constants, factor_graphs, factory

def load_model(model_dir) -> tuple[factor_graphs.FactorGraph, factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph, bayesian_networks.BayesianNetwork]:
    # TODO refactor this together with saving models into the main dblm package
    config = load_json(os.path.join(model_dir, "config.json"))
    z0_num_variables = config["z0_num_variables"]
    z0_num_values = config["z0_num_values"]
    noise_weight = config["noise_weight"]
    sequence_length = config["sequence_length"]
    factor_graph_model, nested_model, bayes_net_model, name = factory.tree_mrf_with_term_frequency_based_transition_and_separate_noise_per_token(
        z0_num_variables,
        z0_num_values,
        constants.TensorInitializer.UNIFORM,
        (1-noise_weight, noise_weight),
        sequence_length,
        constants.TensorInitializer.UNIFORM,
        (1-noise_weight, noise_weight))
    factor_graph_model.load_state_dict(torch.load(os.path.join(model_dir, "factor_graph.pt")))
    nested_model.load_state_dict(torch.load(os.path.join(model_dir, "nested.pt")))
    bayes_net_model.load_state_dict(torch.load(os.path.join(model_dir, "bayes_net.pt")))
    factor_variables = [tuple(vars) for vars in load_json(os.path.join(model_dir, "factor_variables.json"))]
    factor_graph_model._factor_variables = factor_variables
    factor_graph_model._initialize_graph(factor_graph_model.nvars, factor_variables)
    nested_model._factor_variables = factor_variables
    nested_model._initialize_graph(factor_graph_model.nvars, factor_variables)
    return factor_graph_model, nested_model, bayes_net_model

def load_noiseless_model(model_dir, load_weights=True) -> tuple[factor_graphs.FactorGraph, factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph, bayesian_networks.BayesianNetwork]:
    # TODO refactor this together with saving models into the main dblm package
    config = load_json(os.path.join(model_dir, "config.json"))
    z0_num_variables = config["z0_num_variables"]
    z0_num_values = config["z0_num_values"]
    sequence_length = config["sequence_length"]
    factor_graph_model, nested_model, bayes_net_model, name = factory.tree_mrf_with_term_frequency_based_transition_and_no_noise(
        z0_num_variables,
        z0_num_values,
        constants.TensorInitializer.UNIFORM,
        sequence_length,
        constants.TensorInitializer.UNIFORM)
    if load_weights:
        factor_graph_model.load_state_dict(torch.load(os.path.join(model_dir, "factor_graph.pt")))
        nested_model.load_state_dict(torch.load(os.path.join(model_dir, "nested.pt")))
        bayes_net_model.load_state_dict(torch.load(os.path.join(model_dir, "bayes_net.pt")))
    factor_variables = [tuple(vars) for vars in load_json(os.path.join(model_dir, "factor_variables.json"))]
    factor_graph_model._factor_variables = factor_variables
    factor_graph_model._initialize_graph(factor_graph_model.nvars, factor_variables)
    nested_model._factor_variables = factor_variables
    nested_model._initialize_graph(factor_graph_model.nvars, factor_variables)
    return factor_graph_model, nested_model, bayes_net_model

def load_data(samples_file):
    data_matrix = []
    with open(samples_file) as f:
        reader = csv.reader(f)
        for row in reader:
            data_matrix.append([int(i) for i in row])
    return torch.tensor(data_matrix)

def load_json(mapping_file):
    with open(mapping_file) as f:
        return json.loads(f.read())

class DataMatrix:

    def __init__(self, samples_file, mapping_file) -> None:
        self.data_matrix = load_data(samples_file)
        self.mapping = load_json(mapping_file)

    def filter_variables_by_name_rule(self, f):
        names = filter(f, self.mapping.keys())
        indices = sorted(self.mapping[name] for name in names)
        return indices, self.subset_variables_by_indices(indices)

    def subset_variables_by_names(self, names):
        indices = sorted(self.mapping[name] for name in names)
        return self.subset_variables_by_indices(indices)

    def subset_variables_by_indices(self, indices):
        return self.data_matrix[:, indices]

class NgramLM:

    def __init__(self, model_file, name=None) -> None:
        self.model_file = model_file
        self.name = name


    def cross_entropy_of_sequences(self, sequences: list[list[int]]):
        data_file = f"/tmp/dblm-ngram-ppl-{self.name}"
        output_file = f"/tmp/dblm-ngram-ppl-results-{self.name}"
        with open(data_file, "w") as f:
            for sequence in sequences:
                print(" ".join(str(token) for token in sequence), file=f)
        os.system(f"bash << 'EOF'\n"
                  f"ngram -no-sos -no-eos -lm {self.model_file} -ppl {data_file} > {output_file}\n"
                  f"EOF")
        with open(output_file) as f:
            line = f.read()
        match = re.search(r".* ppl= (\d+\.\d+) .*", line)
        assert match is not None
        ppl = float(match.group(1))
        return math.log(ppl)

    @staticmethod
    def from_sequences(sequences: list[list[int]], save_model_file=None):
        name = uuid.uuid4()
        if save_model_file is None:
            save_model_file = f"/tmp/dblm-ngram-{name}"
        data_file = f"/tmp/dblm-ngram-train-{name}"
        with open(data_file, "w") as f:
            for sequence in sequences:
                print(" ".join(str(token) for token in sequence), file=f)
        os.system(f"bash << 'EOF'\n"
                 f"ngram-count -text {data_file} -no-sos -no-eos -lm {save_model_file} -wbdiscount -interpolate\n"
                f"EOF")
        return NgramLM(save_model_file, name)
