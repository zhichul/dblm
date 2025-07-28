import argparse
import os
import torch

import wandb

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--name", default=None, required=True
)
argparser.add_argument(
    "--group", default="finetune-data-multi-factored"
)
argparser.add_argument(
    "--model", required=True
)
args = argparser.parse_args()
run = wandb.init(
    # Set the project where this run will be logged
    name=args.name,
    project="dblm",
    group=f"pilot_study_3/{args.group}",
    # Track hyperparameters and run metadata
    config=vars(args))

d = torch.load(os.path.join(args.model, "checkpoint-early-stopping", "dev_cross_entropy_breakdown.bin"))
var_ce = d["variable_cross_entropy"].mean(dim=0)
val_ce = d["value_cross_entropy"].mean(dim=0)
for pos, (var, val) in enumerate(zip(var_ce, val_ce)):
    wandb.log({"loss_var": var, "loss_val": val}, step=pos+1)