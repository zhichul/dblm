
import argparse
import csv
import json
import os
from dblm.experiments.data import load_json

from dblm.utils import seeding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_json(os.path.join(args.data_dir, "ground_truth_models/config.json"))
    z_offsets = [sum(config["w_nvals"][:i]) for i in range(config["w_nvars"])]
    bos_index = sum(config["w_nvals"])
    x_offset = bos_index + 1 # + 1 to make space for for [BOS]
    with open(os.path.join(args.data_dir, "ground_truth_models/mapping.json")) as mapping_file:
        mapping = json.loads(mapping_file.read())
    u_indices = sorted([v for k, v in mapping.items() if k.startswith("u(")])
    x_indices = sorted([v for k, v in mapping.items() if k.startswith("x(")])
    with open(os.path.join(args.data_dir, "samples/sample.csv")) as input_file:
        csv_reader = csv.reader(input_file)
        with open(os.path.join(args.data_dir, "samples/extracted_samples.jsonl"), "w") as output_file:
            for line in csv_reader:
                if not len(z_offsets) == len(u_indices): 
                    raise AssertionError()
                print(json.dumps({"z": [int(line[u]) + o for o, u in zip(z_offsets, u_indices)], "x": [bos_index] + [int(line[x]) + x_offset for x in x_indices]}), file=output_file)




if __name__ == "__main__":
    main()
