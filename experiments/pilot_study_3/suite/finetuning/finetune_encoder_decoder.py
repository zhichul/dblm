import argparse
import json
import math
import os

import torch
import tqdm
from dblm.core.modeling import gpt2
from dblm.utils import seeding
from transformers.models.encoder_decoder import EncoderDecoderModel
from dblm.experiments.pilot_study_3 import utils
from transformers.models import bert
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
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
    parser.add_argument("--uniform_noise", type=float, required=False)
    parser.add_argument("--first", type=int, required=False)

    # pretrained model
    parser.add_argument("--pretrained_model", type=str, required=True)

    # training
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train_steps", type=int, required=True)
    parser.add_argument("--eval_steps", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--gpu_batch_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--group", type=str, required=False)
    parser.add_argument("--name", type=str, required=False)
    parser.add_argument("--multitask_lambda", type=float, required=False)
    parser.add_argument("--no_pos_z", action="store_true", default=False)
    parser.add_argument("--post_scale_gate", action="store_true", default=False)
    parser.add_argument("--gating_mode", type=str, required=False, choices=["mult", "albo"],default="albo")

    # ablations
    parser.add_argument("--no_inference", action="store_true", default=False)

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


def evaluate(model, loader, data_config, no_inference=False, uniform_noise=None, no_pos_z=False, gating_mode="albo"):
    loss = 0
    n = 0
    with torch.no_grad():
        model.eval()
        for gpu_batch in tqdm.tqdm(loader):
            _, x, inf = (i.to("cuda") for i in gpu_batch)
            if uniform_noise is not None:
                inf = utils.add_uniform_noise(inf, uniform_noise)
            xinput = x[..., :-1].contiguous()
            xlabel = x[..., 1:].contiguous()
            z_all = torch.arange(data_config["nvars"] * data_config["nvals"]).expand(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
            pz_all = torch.arange(data_config["nvars"]).repeat_interleave(data_config["nvals"]).expand(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
            px = torch.arange(xinput.size(1)).expand_as(xinput).to("cuda")
            if no_pos_z:
                pz_all = pz_all.fill_(0)
            output = model(input_ids=z_all, position_ids=pz_all,
                        token_type_ids=z_all.new_zeros(z_all.size()),
                        labels=xlabel,
                        decoder_input_ids=xinput, decoder_position_ids=px,
                        decoder_encoder_log_marginals=inf[:,:-1,:] if not no_inference else None,
                        decoder_encoder_attention_mode=gating_mode if not no_inference else None)
            loss += output.loss.item() * x.size(0) # type:ignore
            n += x.size(0)
        model.train()
    return loss / n

def main():
    args = parse_args()
    seeding.seed(args.seed)
    with open(os.path.join(args.output_dir, "log.jsonl"), "w") as f:
        pass

    if args.post_scale_gate:
        args.gating_mode = f"post_scale:{args.gating_mode}"


    # load model
    with open(os.path.join(args.pretrained_model, "encoder","config.json")) as f:
        encoder_config = json.load(f)
    with open(os.path.join(args.pretrained_model, "decoder","config.json")) as f:
        decoder_config = json.load(f)
    encoder = bert.BertModel(bert.BertConfig.from_dict(encoder_config))
    decoder = gpt2.GPT2LMHeadModel(gpt2.GPT2Config.from_dict(decoder_config))
    def remove_persistent(d):
        return {k:v for k, v in d.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias") and not k.endswith("crossattention.bias") and not k.endswith("crossattention.masked_bias") }
    encoder.load_state_dict(torch.load(os.path.join(args.pretrained_model, "encoder","pytorch_model.bin")))
    decoder.load_state_dict(remove_persistent(torch.load(os.path.join(args.pretrained_model, "decoder","pytorch_model.bin"))))
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to("cuda")

    # load data
    with open(args.data_config) as f:
        data_config = json.load(f)
    with open(args.data_meta) as f:
        data_meta = json.load(f)
        theoretical_minimum = (data_meta["z_entropy"] +
                               math.log(data_config["n_branches"] ** (data_config["seq_len"]-data_config["n_branches"])
                                         * math.factorial(data_config["n_branches"]))
                              ) / data_config["seq_len"]
    train_dict = torch.load(args.train_file)
    dev_dict = torch.load(args.dev_file)
    test_dict = torch.load(args.test_file)

    train_dataset = TensorDataset(train_dict["z"], torch.cat([torch.empty((train_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), train_dict["x"]], dim=1), train_dict["inf"])
    dev_dataset = TensorDataset(dev_dict["z"],torch.cat([torch.empty((dev_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), dev_dict["x"]], dim=1), dev_dict["inf"])
    test_dataset = TensorDataset(test_dict["z"], torch.cat([torch.empty((test_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), test_dict["x"]], dim=1), test_dict["inf"])
    if args.first is not None:
        train_dataset = Subset(train_dataset, indices=list(range(args.first)))

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

    # load multitask data
    if args.multitask_lambda is not None:
        multitask_data = utils.encoder_decoder_train_data_generator(data_config["nvars"],
                                                                    data_config["nvals"],
                                                                    data_config["seq_len"],
                                                                    data_config["n_branches"],
                                                                    args.gpu_batch_size,
                                                                    data_config["x_model_seed"])
    else:
        multitask_data = None

    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # train
    step = 0
    best_dev_loss = None

    default_group = "finetune_encoder_decoder" + ("_no_inference" if args.no_inference else "_with_inference")
    run = wandb.init(
        # Set the project where this run will be logged
        name=args.name,
        project="dblm",
        group=f"pilot_study_3/{default_group if not args.group else args.group}",
        # Track hyperparameters and run metadata
        config=vars(args))
    bar = tqdm.tqdm(load_forever(train_dataloader), desc="loss=", total=args.train_steps)


    # evaluate once
    dev_loss = evaluate(model, dev_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode)
    test_loss = evaluate(model, test_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode)
    logline={
            "step": step,
            "n_examples": step * args.batch_size,
            "dev_loss": dev_loss,
            "test_loss": test_loss,
            "dev_loss_delta": dev_loss - theoretical_minimum,
            "test_loss_delta": test_loss - theoretical_minimum,
        }
    for gpu_batch in bar:
        _, x, inf = (i.to("cuda") for i in gpu_batch)
        if args.uniform_noise is not None:
            inf = utils.add_uniform_noise(inf, args.uniform_noise)
        xinput = x[..., :-1].contiguous()
        xlabel = x[..., 1:].contiguous()
        z_all = torch.arange(data_config["nvars"] * data_config["nvals"]).expand(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
        pz_all = torch.arange(data_config["nvars"]).repeat_interleave(data_config["nvals"]).expand(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
        px = torch.arange(xinput.size(1)).expand_as(xinput).to("cuda")
        if args.no_pos_z:
            pz_all = pz_all.fill_(0)
        output = model(input_ids=z_all, position_ids=pz_all,
                    token_type_ids=z_all.new_zeros(z_all.size()),
                    decoder_input_ids=xinput, decoder_position_ids=px,
                    labels=xlabel,
                    decoder_encoder_log_marginals=inf[:,:-1,:] if not args.no_inference else None,
                    decoder_encoder_attention_mode=args.gating_mode if not args.no_inference else None)
        loss = output.loss # type:ignore
        bar.set_description(f"loss={loss.item():.2f}")

        if args.multitask_lambda is not None:
            z, xinput, xlabel, pz, px = (i.to("cuda") for i in next(multitask_data)) # type:ignore
            reg_output = model(input_ids=z, decoder_input_ids=xinput, position_ids=pz, decoder_position_ids=px, labels=xlabel, return_dict=True)
            reg_loss = reg_output.loss
            actual_loss = loss + reg_loss * args.multitask_lambda
        else:
            actual_loss = loss

        ((args.gpu_batch_size / args.batch_size) * actual_loss).backward() # gradient accumulation
        if (step +1) % (args.batch_size // args.gpu_batch_size) == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if (step % args.logging_steps == 0) and step % args.eval_steps != 0:
                logline={
                        "train_loss": loss.item(),
                        "step": step,
                        "n_examples": step * args.batch_size,
                        "train_loss_delta": loss.item() - theoretical_minimum,
                    }
                if args.multitask_lambda is not None:
                    logline["reg_loss"] = reg_loss.item() # type:ignore
                    logline["actual_loss"] = actual_loss.item() # type:ignore
                    logline["actual_loss_delta"] = actual_loss.item() - theoretical_minimum # type:ignore
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline, step=step)
            if (step == 1):
                logline["train_loss"] = loss.item()
                logline["train_loss_delta"] = loss.item() - theoretical_minimum
                if args.multitask_lambda is not None:
                    logline["reg_loss"] = reg_loss.item() # type:ignore
                    logline["actual_loss"] = actual_loss.item() # type:ignore
                    logline["actual_loss_delta"] = actual_loss.item() - theoretical_minimum # type:ignore
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline, step=0)
            if (step % args.eval_steps == 0):
                dev_loss = evaluate(model, dev_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode)
                test_loss = evaluate(model, test_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode)
                if best_dev_loss is None or dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    save_checkpoint(args, f"checkpoint-early-stopping", model, step, step * args.batch_size)
                logline={
                        "train_loss": loss.item(),
                        "step": step,
                        "n_examples": step * args.batch_size,
                        "dev_loss": dev_loss,
                        "test_loss": test_loss,
                        "train_loss_delta": loss.item() - theoretical_minimum,
                        "dev_loss_delta": dev_loss - theoretical_minimum,
                        "test_loss_delta": test_loss - theoretical_minimum,
                    }
                if args.multitask_lambda is not None:
                    logline["reg_loss"] = reg_loss.item() # type:ignore
                    logline["actual_loss"] = actual_loss.item() # type:ignore
                    logline["actual_loss_delta"] = actual_loss.item() - theoretical_minimum # type:ignore
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline, step=step)
            if step >= args.train_steps:
                break
    save_checkpoint(args, "checkpoint-final", model, step, step * args.batch_size)


if __name__ == "__main__":
    main()
