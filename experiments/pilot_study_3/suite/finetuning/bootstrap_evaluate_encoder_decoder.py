import argparse
import json
import math
import os

import torch
from dblm.utils import seeding
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from finetune_encoder_decoder import evaluate, load_model
from transformers.models import bert, encoder_decoder
from dblm.core.modeling import gpt2

def parse_args():
    parser = argparse.ArgumentParser()

    # training data
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--data_meta", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--uniform_noise", type=float, required=False)

    # pretrained model
    parser.add_argument("--finetuned_model", type=str, required=True)

    # training
    parser.add_argument("--gpu_batch_size", type=int, required=True)
    parser.add_argument("--no_pos_z", action="store_true", default=False)
    parser.add_argument("--post_scale_gate", action="store_true", default=False)
    parser.add_argument("--gating_mode", type=str, required=False, choices=["mult", "albo", "albo_full", "shared_sample", "pre_sample"],default="albo")
    parser.add_argument("--num_samples", type=int, required=False)
    parser.add_argument("--no_pdrop", action="store_true", required=False)
    # ablations
    parser.add_argument("--no_inference", action="store_true", default=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    is_full = args.gating_mode == "albo_full"
    seeding.seed(42)

    if args.post_scale_gate:
        args.gating_mode = f"post_scale:{args.gating_mode}"
    gating_mode_is_shared_sample = args.gating_mode.endswith("shared_sample")
    gating_mode_is_pre_sample = args.gating_mode.endswith("pre_sample")
    if (gating_mode_is_shared_sample or gating_mode_is_pre_sample) and is_full:
        raise NotImplementedError("don't support currently the use samples cond. marginals")
    if args.uniform_noise is not None and is_full:
        raise NotImplementedError("don't support currently the use of uniform noise with cond. marginals")


    with open(os.path.join(args.finetuned_model, "config.json")) as f:
        config = json.load(f)
    model = encoder_decoder.EncoderDecoderModel.from_pretrained(args.finetuned_model)
    decoder = model.decoder # type:ignore
    model.decoder = gpt2.GPT2LMHeadModel(gpt2.GPT2Config.from_dict(config["decoder"])) # type:ignore
    model.decoder.load_state_dict(decoder.state_dict()) # type:ignore
    model.to("cuda") # type:ignore
    model.eval()# type:ignore

    # load data
    with open(args.data_config) as f:
        data_config = json.load(f)
    with open(args.data_meta) as f:
        data_meta = json.load(f)
        theoretical_minimum = (data_meta["z_entropy"] +
                               math.log(data_config["n_branches"] ** (data_config["seq_len"]-data_config["n_branches"])
                                         * math.factorial(data_config["n_branches"]))
                              ) / data_config["seq_len"]
    dev_dict = torch.load(args.dev_file)
    test_dict = torch.load(args.test_file)
    dev_dataset = TensorDataset(*((dev_dict["z"],torch.cat([torch.empty((dev_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), dev_dict["x"]], dim=1)) # type:ignore
                                    + ((dev_dict["inf"],) if not gating_mode_is_pre_sample else tuple())
                                   + ((dev_dict["cond_inf"],) if is_full else tuple())
                                   + ((dev_dict["inf_samples"][:,:-1,:args.num_samples,:],) if gating_mode_is_pre_sample else tuple())))
    test_dataset = TensorDataset(*((test_dict["z"], torch.cat([torch.empty((test_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), test_dict["x"]], dim=1)) # type:ignore
                                    + ((test_dict["inf"],) if not gating_mode_is_pre_sample else tuple())
                                    + ((test_dict["cond_inf"],) if is_full else tuple())
                                    + ((test_dict["inf_samples"][:,:-1,:args.num_samples,:],) if gating_mode_is_pre_sample else tuple())))
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                sampler=SequentialSampler(dev_dataset),
                                batch_size=args.gpu_batch_size)
    test_dataloader = DataLoader(dataset=test_dataset,
                                sampler=SequentialSampler(test_dataset),
                                batch_size=args.gpu_batch_size)
    dev_loss, dev_bootstrap = evaluate(model, dev_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode, num_samples=args.num_samples, is_full=is_full, bootstrap=10000) # type:ignore
    test_loss, test_bootstrap = evaluate(model, test_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode, num_samples=args.num_samples, is_full=is_full, bootstrap=10000) # type:ignore
    logline = {"dev_loss": dev_loss,
               "test_loss": test_loss,
               "dev_loss_low": dev_bootstrap.confidence_interval.low,
               "dev_loss_high": dev_bootstrap.confidence_interval.high,
               "dev_loss_std": dev_bootstrap.standard_error,
               "test_loss_low": test_bootstrap.confidence_interval.low,
               "test_loss_high": test_bootstrap.confidence_interval.high,
               "test_loss_std": test_bootstrap.standard_error}
    with open(os.path.join(args.finetuned_model,"bootstrap.json"),"w") as f:
        print(json.dumps(logline, indent=4), file=f)

if __name__ == "__main__":
    with torch.no_grad():
        main()
