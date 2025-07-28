
import json
import os
import random
import torch
import tqdm
from dblm.experiments.pilot_study_1 import tree_mrf_noiseless_emission
from dblm.utils import seeding
from dblm.experiments import data
from constants import TRAIN_ROOT, DEV_ROOT, TEST_ROOT
import transformers.models.gpt2.modeling_gpt2 as modeling_gpt2

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
        config = data.load_json(config_file)
        train_data_matrix = data.DataMatrix(train_file, mapping_file)
        dev_data_matrix = data.DataMatrix(dev_file, mapping_file)
        test_data_matrix = data.DataMatrix(test_file, mapping_file)
        indices, assignments = train_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables
        dev_indices, dev_assignments = dev_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables
        test_indices, test_assignments = test_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables
        example_ids = list(range(assignments.size(0)))
        transformer = modeling_gpt2.GPT2LMHeadModel(modeling_gpt2.GPT2Config(vocab_size=41, n_positions=11, bos_token_id=40))
        transformer.to("cuda:0")
        assignments = torch.cat([torch.ones(assignments.size(0), dtype=torch.long)[:, None]*40, assignments], dim=1)
        assignments = assignments.to("cuda:0")
        dev_assignments = dev_assignments.to("cuda:0")
        test_assignments = test_assignments.to("cuda:0")
        # training setup
        nepochs = 40
        lr = 6.25e-5
        optimizer = torch.optim.Adam(transformer.parameters(), lr)

        print(transformer)

        # train
        batch_size = 1000
        for epoch in range(nepochs):
            random.shuffle(example_ids)
            epoch_loss = 0
            bar = tqdm.tqdm(range(0, assignments.size(0), batch_size))
            for offset in bar:
                input_ids = assignments[offset: offset+ batch_size]
                loss = transformer(input_ids=input_ids, labels=input_ids)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # bookkeep
                epoch_loss += loss.item() / (assignments.size(0) / batch_size)
                bar.set_description_str(f"loss={loss.item():.4}")
            print(f"End of epoch {epoch}: average loss is {epoch_loss}")
            if epoch % 1 == 0:
                # evaluate
                with torch.no_grad():
                    dev_loss = transformer(input_ids=dev_assignments, labels=dev_assignments)["loss"].item()
                    test_loss = transformer(input_ids=test_assignments, labels=test_assignments)["loss"].item()
                with open(f"transformer_models/{config['seed']}/metrics_at_end_of_epoch_{epoch}.json", "w") as f:
                    json.dump({"train_loss": epoch_loss, "dev_loss": dev_loss, "test_loss": test_loss}, f)

if __name__ == "__main__":
    main()
