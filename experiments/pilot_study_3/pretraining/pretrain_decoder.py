import argparse
import json
import os
import random

import torch
import tqdm
from dblm.core.modeling import gpt2
from dblm.experiments.pilot_study_3 import distributions
from dblm.utils import seeding
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    # z
    parser.add_argument("--nvars", type=int, required=True)
    parser.add_argument("--nvals", type=int, required=True)

    # x
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--n_branches", type=int, required=True)
    parser.add_argument("--x_model_seed", type=int, required=True)

    # training
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train_steps", type=int, required=True)
    parser.add_argument("--save_steps", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--gpu_batch_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def save_checkpoint(args, checkpoint_name, model, step, n_examples):
    checkpointdir = os.path.join(args.output_dir, checkpoint_name)
    os.makedirs(checkpointdir, exist_ok=True)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(checkpointdir, "pytorch_model.bin")
    output_config_file = os.path.join(checkpointdir, "config.json")

    # save model and config
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)

    with open(os.path.join(checkpointdir, "info.json"), "w") as f:
        print(json.dumps({"step": step, "n_examples": n_examples}), file=f)


def train_data_generator(nvars, nvals, seq_len, n_branches, batch_size, x_model_seed):
    """Note that seq_len is the length WITHOUT [BOS]"""
    indices = torch.tensor(distributions.all_indices(seq_len, n_branches, list(range(nvars)), x_model_seed), dtype=torch.long)
    ids = list(range(indices.size(0)))
    while True:
        z = torch.randint(nvals, (batch_size, nvars))
        z = z + (torch.arange(nvars) * nvals)[None,...]
        x_sel = indices[random.sample(ids, batch_size)]
        x = torch.gather(z, 1, x_sel) # type:ignore
        x = torch.cat([torch.empty((batch_size, 1), dtype=torch.long).fill_(nvars * nvals), x], dim=1) # prepend BOS
        px = torch.arange(seq_len + 1).expand(x.size())
        yield x, px

def main():
    args = parse_args()
    seeding.seed(args.seed)
    with open(os.path.join(args.output_dir, "log.jsonl"), "w") as f:
        pass


    # load data
    seeding.seed(args.seed)
    train_data = train_data_generator(args.nvars, args.nvals, args.seq_len, args.n_branches, args.gpu_batch_size, args.x_model_seed)

    # load model

    model = gpt2.GPT2LMHeadModel(gpt2.GPT2Config(vocab_size=args.nvars * args.nvals + 1,
                                                    n_positions=args.seq_len + 1,
                                                    bos_token_id=args.nvars * args.nvals,
                                                    n_layer=args.n_layer,
                                                    add_cross_attention=True,
                                                    is_decoder=True))
    model.to("cuda")

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train
    step = 0

    # train
    run = wandb.init(
        # Set the project where this run will be logged
        project="dblm",
        group="pilot_study_3/pretrain_decoder",
        # Track hyperparameters and run metadata
        config=vars(args))
    bar = tqdm.tqdm(train_data, desc="loss=", total=args.train_steps)
    for gpu_batch in bar:
        x, px = (i.to("cuda") for i in gpu_batch)
        output = model(input_ids=x, position_ids=px, labels=x, return_dict=True)
        loss = output.loss
        bar.set_description(f"loss={loss.item():.2f}")
        ((args.gpu_batch_size / args.batch_size) * loss).backward() # gradient accumulation
        if (step +1) % (args.batch_size // args.gpu_batch_size) == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % args.save_steps == 0:
                save_checkpoint(args, f"checkpoint-{step}", model, step, step * args.batch_size)
            if (step == 1) or (step % args.logging_steps == 0):
                logline={
                        "train_loss": loss.item(),
                        "step": step,
                        "n_examples": step * args.batch_size
                    }
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline)
            if step >= args.train_steps:
                break
    save_checkpoint(args, "checkpoint-final", model, step, step * args.batch_size)


if __name__ == "__main__":
    main()
