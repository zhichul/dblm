
import argparse
import json
import math
import os

import torch
from torch.utils.data import dataset, dataloader
import tqdm
from dblm.core.modeling import gpt2
from dblm.experiments.data import load_json

from dblm.utils import seeding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--train_epochs", type=int, required=True)
    parser.add_argument("--eval_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--gpu_batch_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def load_jsonl_dataset(file, key=None):
    dataset = []
    with open(file, "rt") as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example[key])
    return torch.tensor(dataset)


def save_checkpoint(args, checkpoint_name, model, step, epoch):
    checkpointdir = os.path.join(args.output_dir, checkpoint_name)
    os.makedirs(checkpointdir, exist_ok=True)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(checkpointdir, "pytorch_model.bin")
    output_config_file = os.path.join(checkpointdir, "config.json")

    # save model and config
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)

    with open(os.path.join(checkpointdir, "info.json"), "w") as f:
        print(json.dumps({"step": step, "epoch": epoch}), file=f)

def main():

    args = parse_args()
    seeding.seed(args.seed)
    with open(os.path.join(args.output_dir, "log.jsonl"), "w") as f:
        pass

    train_data = load_jsonl_dataset(args.train_file, key="z"), load_jsonl_dataset(args.train_file, key="x")
    dev_data = load_jsonl_dataset(args.dev_file, key="z"), load_jsonl_dataset(args.dev_file, key="x")
    train_dataset = dataset.TensorDataset(*train_data)
    dev_dataset = dataset.TensorDataset(*dev_data)
    train_dataloader = dataloader.DataLoader(
        train_dataset,
        batch_size=args.gpu_batch_size,
        sampler=dataloader.RandomSampler(train_dataset),
    )
    dev_dataloader = dataloader.DataLoader(
        dev_dataset,
        batch_size=args.gpu_batch_size,
        sampler=dataloader.SequentialSampler(dev_dataset),
    )
    config = load_json(args.data_config)
    model = gpt2.GPT2LMHeadModel(gpt2.GPT2Config(vocab_size=sum(config["w_nvals"]) * 2 + 1, 
                                                       n_positions=config["zt_sequence_length"] + 1,
                                                         bos_token_id=sum(config["w_nvals"]), 
                                                         n_layer=args.n_layer,
                                                         add_cross_attention=True))
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    step = 0
    best_dev_loss = math.inf
    epoch = 0

    # eval once
    model.eval()
    dev_loss = 0
    dev_examples = 0
    with torch.no_grad():
        for gpu_batch in tqdm.tqdm(dev_dataloader):
            zs, xs = gpu_batch
            zs = zs.to("cuda")
            xs = xs.to("cuda")
            encoder_hidden_states = model.transformer.wte(zs)
            loss = model(input_ids=xs, encoder_hidden_states=encoder_hidden_states, labels=xs).loss
            dev_loss += loss.item() * xs.size(0)
            dev_examples += xs.size(0)
    dev_loss = dev_loss / dev_examples
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        save_checkpoint(args, "checkpoint-early-stopping", model, step, epoch)

    # train
    for epoch in tqdm.tqdm(range(1, args.train_epochs + 1)):
        model.train()
        train_loss = 0
        train_examples = 0
        bar = tqdm.tqdm(train_dataloader, desc="loss=")
        for gpu_batch in bar:
            zs, xs = gpu_batch
            zs = zs.to("cuda")
            xs = xs.to("cuda")
            encoder_hidden_states = model.transformer.wte(zs)
            loss = model(input_ids=xs, encoder_hidden_states=encoder_hidden_states, labels=xs).loss
            bar.set_description(f"loss={loss.item():.2f}")
            ((args.gpu_batch_size / args.batch_size) * loss).backward()
            if (step +1) % (args.batch_size // args.gpu_batch_size) == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item() * xs.size(0)
            train_examples += xs.size(0)
            step += 1
        train_loss = train_loss / train_examples
        dev_loss = 0
        dev_examples = 0
        if epoch % args.eval_epochs == 0:
            model.eval()
            with torch.no_grad():
                for gpu_batch in tqdm.tqdm(dev_dataloader):
                    zs, xs = gpu_batch
                    zs = zs.to("cuda")
                    xs = xs.to("cuda")
                    encoder_hidden_states = model.transformer.wte(zs)
                    loss = model(input_ids=xs, encoder_hidden_states=encoder_hidden_states, labels=xs).loss
                    dev_loss += loss.item() * xs.size(0)
                    dev_examples += xs.size(0)
            dev_loss = dev_loss / dev_examples
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                save_checkpoint(args, "checkpoint-early-stopping", model, step, epoch)
        with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
            print(json.dumps({
                "train_loss": train_loss,
                "train_examples": train_examples,
                "dev_loss": dev_loss,
                "dev_examples": dev_examples,
                "epoch": epoch,
                "step": step,
            }), file=f)

    # eval once
    model.eval()
    dev_loss = 0
    dev_examples = 0
    with torch.no_grad():
        for gpu_batch in tqdm.tqdm(dev_dataloader):
            zs, xs = gpu_batch
            zs = zs.to("cuda")
            xs = xs.to("cuda")
            encoder_hidden_states = model.transformer.wte(zs)
            loss = model(input_ids=xs, encoder_hidden_states=encoder_hidden_states, labels=xs).loss
            dev_loss += loss.item() * xs.size(0)
            dev_examples += xs.size(0)
    dev_loss = dev_loss / dev_examples
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        save_checkpoint(args, "checkpoint-early-stopping", model, step, epoch)
    save_checkpoint(args, "checkpoint-final", model, step, epoch)




if __name__ == "__main__":
    main()
