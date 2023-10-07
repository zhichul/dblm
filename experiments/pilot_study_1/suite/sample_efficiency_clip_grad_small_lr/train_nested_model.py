
import code
import json
import math
import os
import random
import torch
import tqdm
from dblm.utils import seeding
from dblm.experiments import data
from dblm.core.modeling import constants, factor_graphs, factory
from constants import TRAIN_ROOT, DEV_ROOT, TEST_ROOT


def main():
    seed = 42
    seeding.seed(seed)
    torch.set_anomaly_enabled(True)

    for seed in [11,12,13,14,15]:
        train_file = os.path.join(TRAIN_ROOT % f"{seed}", "samples", "sample.csv")
        dev_file = os.path.join(DEV_ROOT % f"{seed}", "samples", "sample.csv")
        test_file = os.path.join(TEST_ROOT % f"{seed}", "samples", "sample.csv")
        mapping_file = os.path.join(TRAIN_ROOT % f"{seed}", "ground_truth_models", "mapping.json")
        config_file = os.path.join(TRAIN_ROOT % f"{seed}", "ground_truth_models", "config.json")
        train_data_matrix = data.DataMatrix(train_file, mapping_file)
        dev_data_matrix = data.DataMatrix(dev_file, mapping_file)
        test_data_matrix = data.DataMatrix(test_file, mapping_file)
        config = data.load_json(config_file)
        # factor_graph_model, nested_model, bayes_net_model, name = factory.tree_mrf_with_term_frequency_based_transition_and_separate_noise_per_token(
        # config["z0_num_variables"],
        # config["z0_num_values"],
        # constants.TensorInitializer.UNIFORM,
        # (1-config["noise_weight"], config["noise_weight"]),
        # config["sequence_length"],
        # constants.TensorInitializer.UNIFORM,
        # (1-config["noise_weight"], config["noise_weight"]))
        factor_graph, nested_model, bayes_net = data.load_model(os.path.join(TRAIN_ROOT % f"{seed}", "ground_truth_models"))

        config = data.load_json(config_file)
        indices, assignments = train_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables
        dev_indices, dev_assignments = dev_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables
        test_indices, test_assignments = test_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables

        example_ids = list(range(assignments.size(0)))
        # training setup
        nepochs = 100
        lr = 0.001
        optimizer = torch.optim.Adam(nested_model.parameters(), lr)

        print(nested_model)

        # assignments = assignments.to("cuda:0")[:1]
        # dev_assignments = dev_assignments.to("cuda:0")[:1]
        # test_assignments = test_assignments.to("cuda:0")[:1]
        # nested_model.to("cuda:0")

        # train
        batch_size = 5000
        #batch_size = 1
        for epoch in range(nepochs):
            random.shuffle(example_ids)
            epoch_loss = 0
            bar = tqdm.tqdm(range(0, assignments.size(0), batch_size))
            for offset in bar:
                loss = -nested_model.log_marginal_probability(assignment=[(vi, assignments[offset: offset+ batch_size, col]) for col, vi in enumerate(indices)]).mean() / config["sequence_length"]

                loss.backward()
                grad_norm = math.sqrt(sum([(p.grad ** 2).sum().item() for p in nested_model.parameters() if p.grad is not None])) # type:ignore
                torch.nn.utils.clip_grad.clip_grad_norm_(nested_model.parameters(), max_norm=1.0)
                print(grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                # bookkeep
                epoch_loss += loss.item() / (assignments.size(0) / batch_size)
                bar.set_description_str(f"loss={loss.item():.4}")
            print(f"End of epoch {epoch}: average loss is {epoch_loss}")
            if epoch % 1 == 0:
                # evaluate
                dev_loss = (-nested_model.log_marginal_probability(assignment=[(vi, dev_assignments[:, col]) for col, vi in enumerate(dev_indices)]).mean() / config["sequence_length"]).item()
                test_loss = (-nested_model.log_marginal_probability(assignment=[(vi, test_assignments[:, col]) for col, vi in enumerate(test_indices)]).mean() / config["sequence_length"]).item()
                with open(f"saved_models/{config['seed']}/metrics_at_end_of_epoch_{epoch}.json", "w") as f:
                    json.dump({"train_loss": epoch_loss, "dev_loss": dev_loss, "test_loss": test_loss}, f)
                if epoch % 20 == 0:
                    torch.save(nested_model.state_dict(), f"saved_models/{config['seed']}/nested_model_at_end_of_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()
