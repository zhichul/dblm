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
from torch.nn import CrossEntropyLoss

def load_forever(iterator):
    while True:
        yield from iterator

def parse_args():
    parser = argparse.ArgumentParser()

    # training data
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--data_meta", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--data_group", default="pilot_study_3", required=False)
    # pretrained model
    parser.add_argument("--pretrained_model", type=str, required=True)

    # training
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu_batch_size", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--gpu_num_samples", type=int, required=True)
    parser.add_argument("--train_steps", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, required=True)
    parser.add_argument("--noise", type=float, default=None, required=False)
    parser.add_argument("--l1", action="store_true",required=False)
    parser.add_argument("--shared_outcome_space", action="store_true",required=False)
    parser.add_argument("--positions", type=int, nargs="+",required=False)

    args = parser.parse_args()
    if not args.num_samples % args.gpu_num_samples == 0:
        raise ValueError()
    return args

def evaluate(model, loader, data_config, gpu_num_samples, num_samples, noise=None, l1=False, positions=None, shared_outcome_space=False):
    print(f"Noise: {noise}")
    logits_list = []
    labels_list = []
    loss = 0
    n = 0
    seq_len = data_config["seq_len"]
    nvars = data_config["nvars"]
    nvals = data_config["nvals"]
    z_offsets = (nvals * torch.arange(nvars)).to("cuda")[None, :]
    if positions is None:
        positions = list(range(seq_len))
    with torch.no_grad():
        model.eval()
        bar = tqdm.tqdm(loader)
        for gpu_batch in bar:
            _, x, inf = (i.to("cuda") for i in gpu_batch)
            if noise is not None:
                inf = utils.add_uniform_noise(inf, noise)
            gbatch_size = x.size(0)
            # for l1 loss
            varname = x[..., 1:].contiguous() // 7
            index =  (varname[None,..., None].expand(gpu_num_samples, gbatch_size, seq_len, 1)*nvals + torch.arange(nvals).to("cuda"))

            # 
            xinput = x[..., :-1].contiguous()[None, :, None, :].expand(gpu_num_samples, gbatch_size, seq_len, seq_len).reshape(gpu_num_samples * gbatch_size * seq_len, seq_len)
            xlabel = x[..., 1:].contiguous()[None, :, None, :].expand(gpu_num_samples, gbatch_size, seq_len, seq_len).reshape(gpu_num_samples * gbatch_size * seq_len, seq_len)
            pz = torch.arange(nvars).expand(gpu_num_samples * gbatch_size * seq_len, nvars).to("cuda")
            px = torch.arange(xinput.size(-1)).expand_as(xinput).to("cuda")
            infs = inf[:, :-1, :].reshape(inf.size(0), seq_len, nvars, nvals)
            log_prob = infs.new_tensor(-math.inf)
            full_log_prob  = infs.new_tensor(-math.inf)
            for _ in tqdm.tqdm(range(num_samples // gpu_num_samples), leave=False):
                samples = torch.distributions.Categorical(logits=infs).sample((gpu_num_samples,)) # type:ignore gpu_samples, batch, seq, nvars
                samples = samples.reshape(gpu_num_samples * gbatch_size * seq_len, nvars)
                if not shared_outcome_space:
                    samples = samples + z_offsets
                output = model(input_ids=samples, position_ids=pz,
                            token_type_ids=samples.new_zeros(samples.size()),
                            labels=xlabel,
                            decoder_input_ids=xinput, decoder_position_ids=px)
                logits = output["logits"].reshape(gpu_num_samples, gbatch_size, seq_len, seq_len, -1)[..., positions, positions, :]
                sample_full_log_prob = logits.log_softmax(-1).logsumexp(0)
                if not l1:
                    loss_fct = CrossEntropyLoss(reduction="none")
                    sample_log_prob = (-loss_fct(output["logits"].reshape(-1, model.decoder.config.vocab_size), xlabel.view(-1)).reshape(gpu_num_samples, gbatch_size, seq_len, seq_len)[:, :, positions, positions]).logsumexp(0)
                else:
                    sample_log_prob = torch.gather(logits, -1, index).log_softmax(-1).logsumexp(0)
                log_prob = torch.logaddexp(log_prob, sample_log_prob)
                full_log_prob = torch.logaddexp(full_log_prob, sample_full_log_prob)
            log_prob = log_prob - math.log(num_samples)
            full_log_prob = full_log_prob - math.log(num_samples)
            if not l1:
                loss += -(log_prob).mean().item() * gbatch_size # type:ignore
            else:
                reference = torch.gather(infs, -2, varname[..., None, None].expand(gbatch_size, seq_len, 1, nvals)).squeeze(-2)
                loss += (reference.exp() - (log_prob).exp()).abs().sum(-1).mean().item() * gbatch_size
            n += gbatch_size
            logits_list.extend(full_log_prob.cpu().tolist())
            labels_list.extend(xlabel.cpu().tolist())
            bar.set_description_str(f"{loss/n:.5f}")
    model.train()
    return loss / n, logits_list, labels_list

def main():
    args = parse_args()
    seeding.seed(args.seed)
    with open(os.path.join(args.output_dir, "log.jsonl"), "w") as f:
        pass

    # load pretrained model
    try:
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
    except FileNotFoundError as e:
        with open(os.path.join(args.pretrained_model, "config.json")) as f:
            config = json.load(f)
        model = EncoderDecoderModel.from_pretrained(args.pretrained_model)
        decoder = model.decoder # type:ignore
        model.decoder = gpt2.GPT2LMHeadModel(gpt2.GPT2Config.from_dict(config["decoder"])) # type:ignore
        model.decoder.load_state_dict(decoder.state_dict()) # type:ignore
        model.to("cuda") # type:ignore
        model.eval()# type:ignore
    # load data
    with open(args.data_config) as f:
        data_config = json.load(f)
    with open(args.data_meta) as f:
        if args.data_group == "spelling_same":
            theoretical_minimum = 0
        else:
            data_meta = json.load(f)
            theoretical_minimum = (data_meta["z_entropy"] +
                                math.log(data_config["n_branches"] ** (data_config["seq_len"]-data_config["n_branches"])
                                            * math.factorial(data_config["n_branches"]))
                                ) / data_config["seq_len"]

    dev_dict = torch.load(args.dev_file)
    test_dict = torch.load(args.test_file)

    dev_dataset = TensorDataset(dev_dict["z"],torch.cat([torch.empty((dev_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), dev_dict["x"]], dim=1), dev_dict["inf"]) # type:ignore
    test_dataset = TensorDataset(test_dict["z"], torch.cat([torch.empty((test_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), test_dict["x"]], dim=1), test_dict["inf"])  # type:ignore

    dev_dataloader = DataLoader(dataset=dev_dataset,
                                sampler=SequentialSampler(dev_dataset),
                                batch_size=args.gpu_batch_size)
    test_dataloader = DataLoader(dataset=test_dataset,
                                sampler=SequentialSampler(test_dataset),
                                batch_size=args.gpu_batch_size)

    # train
    run = wandb.init(
        # Set the project where this run will be logged
        name=args.name,
        project=args.project,
        group=f"{args.data_group}/{args.group}",
        # Track hyperparameters and run metadata
        config=vars(args))

    # evaluate once
    # dev_loss = evaluate(model, dev_dataloader, data_config, args.gpu_num_samples, args.num_samples, args.noise, args.l1, args.positions, shared_outcome_space=args.shared_outcome_space)
    test_loss, test_logits, test_labels = evaluate(model, test_dataloader, data_config, args.gpu_num_samples, args.num_samples, args.noise, args.l1, args.positions, shared_outcome_space=args.shared_outcome_space)
    logline={
            "step": 0,
            "n_examples": 0,
            # "dev_loss": dev_loss,
            "test_loss": test_loss,
            # "dev_loss_delta": dev_loss - theoretical_minimum,
            "test_loss_delta": test_loss - theoretical_minimum,
        }
    torch.save(dict(
        test_preds=test_logits,
        test_labels=test_labels
    ),
        os.path.join(args.output_dir, "predictions.bin")
    )
    for step in range(0, args.train_steps, args.logging_steps):
        logline["step"] = step
        with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
            print(json.dumps(logline), file=f)
            wandb.log(logline, step=step)

if __name__ == "__main__":
    main()
