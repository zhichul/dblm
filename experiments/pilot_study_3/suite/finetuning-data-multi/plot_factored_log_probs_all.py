import json


if __name__ == "__main__":
    with open("plot_factored_log_probs.sh", "w") as g:
        print("#"+"!/usr/bin/env bash\n\n", file=g)
        with open("best_models.json") as f:
            d = json.load(f)
        for arch in d:
            for data_size in d[arch]:
                print(f"python3 plot_factored_log_probs_single.py \\\n --model {d[arch][data_size]} \\\n --name {arch}-{data_size}", file=g)