import json
import os
from dblm.experiments import data
from constants import TRAIN_ROOT, DEV_ROOT, TEST_ROOT

def main():
    for seed in [11,12,13,14,15]:
        train_file = os.path.join(TRAIN_ROOT % f"{seed}", "samples", "sample.csv")
        dev_file = os.path.join(DEV_ROOT % f"{seed}", "samples", "sample.csv")
        test_file = os.path.join(TEST_ROOT % f"{seed}", "samples", "sample.csv")
        mapping_file = os.path.join(TRAIN_ROOT % f"{seed}", "ground_truth_models", "mapping.json")
        # config_file = os.path.join(TRAIN_ROOT % f"{seed}", "ground_truth_models", "config.json")
        train_data_matrix = data.DataMatrix(train_file, mapping_file)
        dev_data_matrix = data.DataMatrix(dev_file, mapping_file)
        test_data_matrix = data.DataMatrix(test_file, mapping_file)
        indices, assignments = train_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables
        dev_indices, dev_assignments = dev_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables
        test_indices, test_assignments = test_data_matrix.filter_variables_by_name_rule(lambda name: name.startswith("x")) # get all the observed variables
        lm = data.NgramLM.from_sequences(assignments.tolist())
        train_ce = lm.cross_entropy_of_sequences(assignments.tolist())
        dev_ce = lm.cross_entropy_of_sequences(dev_assignments.tolist())
        test_ce = lm.cross_entropy_of_sequences(test_assignments.tolist())
        line = {"train_loss": train_ce, "dev_loss": dev_ce, "test_loss": test_ce}
        with open(f"saved_models/{seed}/metrics_trigram.json", "w") as f:
            json.dump(line, f)

if __name__ == "__main__":
    main()