import argparse
import json
import math
import os

import torch
import tqdm
from dblm.core.modeling import gpt2
from dblm.utils import seeding
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import wandb

def load_forever(iterator):
    while True:
        yield from iterator

def parse_args():
    parser = argparse.ArgumentParser()

    # training data
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--data_meta", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)

    # training
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train_steps", type=int, required=True)
    parser.add_argument("--eval_steps", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--gpu_batch_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--n_hidden", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--group", type=str, required=False)
    parser.add_argument("--name", type=str, required=False)

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

def evaluate(model, loader, data_config):
    loss = 0
    n = 0
    with torch.no_grad():
        model.eval()
        for gpu_batch in tqdm.tqdm(loader):
            _, x, _ = (i.to("cuda") for i in gpu_batch)
            px = torch.arange(x.size(1)).expand_as(x).to("cuda")
            output = model(input_ids=x, position_ids=px, labels=x)
            loss += output.loss.item() * x.size(0) # type:ignore
            n += x.size(0)
        model.train()
    return loss / n

def main():
    args = parse_args()
    seeding.seed(args.seed)
    with open(os.path.join(args.output_dir, "log.jsonl"), "w") as f:
        pass

    # load data config
    with open(args.data_config) as f:
        data_config = json.load(f)

    # load model
    model = gpt2.GPT2LMHeadModel(gpt2.GPT2Config(vocab_size=data_config["nvars"] * data_config["nvals"] + 1,
                                                    n_positions=data_config["seq_len"] + 1,
                                                    bos_token_id=data_config["nvars"] * data_config["nvals"],
                                                    n_layer=args.n_layer,
                                                    n_embd=args.n_hidden,
                                                    add_cross_attention=False,
                                                    is_decoder=True))
    model.to("cuda")

    # load data
    with open(args.data_meta) as f:
        data_meta = json.load(f)
        theoretical_minimum = (data_meta["z_entropy"] +
                               math.log(data_config["n_branches"] ** (data_config["seq_len"]-data_config["n_branches"])
                                         * math.factorial(data_config["n_branches"]))
                              ) / data_config["seq_len"]
    train_dict = torch.load(args.train_file)
    dev_dict = torch.load(args.dev_file)
    test_dict = torch.load(args.test_file)

    train_dataset = TensorDataset(train_dict["z"], torch.cat([torch.empty((train_dict["x"].size(0),1), dtype=torch.long).fill_(model.config.bos_token_id), train_dict["x"]], dim=1), train_dict["inf"])
    dev_dataset = TensorDataset(dev_dict["z"],torch.cat([torch.empty((dev_dict["x"].size(0),1), dtype=torch.long).fill_(model.config.bos_token_id), dev_dict["x"]], dim=1), dev_dict["inf"])
    test_dataset = TensorDataset(test_dict["z"], torch.cat([torch.empty((test_dict["x"].size(0),1), dtype=torch.long).fill_(model.config.bos_token_id), test_dict["x"]], dim=1), test_dict["inf"])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=args.gpu_batch_size,
                                  drop_last=True)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                sampler=SequentialSampler(dev_dataset),
                                batch_size=args.gpu_batch_size)
    test_dataloader = DataLoader(dataset=test_dataset,
                                sampler=SequentialSampler(test_dataset),
                                batch_size=args.gpu_batch_size)


    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train
    step = 0
    best_dev_loss = None
    default_group = "finetune_decoder"
    run = wandb.init(
        # Set the project where this run will be logged
        name=args.name,
        project="dblm",
        group=f"pilot_study_3/{default_group if not args.group else args.group}",
        # Track hyperparameters and run metadata
        config=vars(args))


    # evaluate once
    dev_loss = evaluate(model, dev_dataloader, data_config)
    test_loss = evaluate(model, test_dataloader, data_config)
    logline = {
            "model_size": model.num_parameters(only_trainable=True),
            "model_size_no_embedding": model.num_parameters(only_trainable=True, exclude_embeddings=True),
            "step": step,
            "n_examples": step * args.batch_size,
            "dev_loss": dev_loss,
            "test_loss": test_loss,
            "dev_loss_delta": dev_loss - theoretical_minimum,
            "test_loss_delta": test_loss - theoretical_minimum,
        }

    bar = tqdm.tqdm(load_forever(train_dataloader), desc="loss=", total=args.train_steps)
    for gpu_batch in bar:
        _, x, _ = (i.to("cuda") for i in gpu_batch)
        px = torch.arange(x.size(1)).expand_as(x).to("cuda")
        output = model(input_ids=x, position_ids=px, labels=x)
        loss = output.loss # type:ignore
        bar.set_description(f"loss={loss.item():.2f}")
        ((args.gpu_batch_size / args.batch_size) * loss).backward() # gradient accumulation
        if (step +1) % (args.batch_size // args.gpu_batch_size) == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if (step % args.logging_steps == 0) and step % args.eval_steps != 0:
                logline={
                        "model_size": model.num_parameters(only_trainable=True),
                        "model_size_no_embedding": model.num_parameters(only_trainable=True, exclude_embeddings=True),
                        "train_loss": loss.item(),
                        "step": step,
                        "n_examples": step * args.batch_size,
                        "train_loss_delta": loss.item() - theoretical_minimum,
                    }
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline)
            if (step == 1):
                logline["train_loss"] = loss.item()
                logline["train_loss_delta"] = loss.item() - theoretical_minimum
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline)
            if (step % args.eval_steps == 0):
                dev_loss = evaluate(model, dev_dataloader, data_config)
                test_loss = evaluate(model, test_dataloader, data_config)
                if best_dev_loss is None or dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    save_checkpoint(args, f"checkpoint-early-stopping", model, step, step * args.batch_size)
                logline={
                        "model_size": model.num_parameters(only_trainable=True),
                        "model_size_no_embedding": model.num_parameters(only_trainable=True, exclude_embeddings=True),
                        "train_loss": loss.item(),
                        "step": step,
                        "n_examples": step * args.batch_size,
                        "dev_loss": dev_loss,
                        "test_loss": test_loss,
                        "train_loss_delta": loss.item() - theoretical_minimum,
                        "dev_loss_delta": dev_loss - theoretical_minimum,
                        "test_loss_delta": test_loss - theoretical_minimum,
                    }
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline)
            if step >= args.train_steps:
                break
    save_checkpoint(args, "checkpoint-final", model, step, step * args.batch_size)


if __name__ == "__main__":
    main()
