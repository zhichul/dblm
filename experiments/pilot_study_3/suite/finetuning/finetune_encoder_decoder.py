import argparse
import itertools
import json
import math
import os
import numpy as np
from scipy import stats

import torch
import tqdm
from dblm.core.modeling import gpt2, bert
# from transformers.models import bert
from dblm.utils import seeding
from dblm.rva.discrete import tree_belief_propagation
# from transformers.models.encoder_decoder import EncoderDecoderModel
from dblm.core.modeling.encoder_decoder import EncoderDecoderModel
from dblm.experiments.pilot_study_3 import utils as utils3
from dblm.experiments.pilot_study_4 import utils as utils4
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
import wandb
from torch.nn import CrossEntropyLoss
from marginalize_encoder_decoder import evaluate as evaluate_m


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
    parser.add_argument("--prefix_value_dropout", type=float, required=False)
    parser.add_argument("--first", type=int, required=False)

    # pretrained model
    parser.add_argument("--pretrained_model", type=str, required=False)

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
    parser.add_argument("--project", type=str, required=False)
    parser.add_argument("--name", type=str, required=False)
    parser.add_argument("--multitask_lambda", type=float, required=False)
    parser.add_argument("--no_pos_z", action="store_true", default=False)
    parser.add_argument("--post_scale_gate", action="store_true", default=False)
    parser.add_argument("--gating_mode", type=str, required=False, choices=["mult", "albo", "albo_full", "shared_sample", "pre_sample"],default="albo")
    parser.add_argument("--num_samples", type=int, required=False)
    parser.add_argument("--no_pdrop", action="store_true", required=False)
    parser.add_argument("--reencode", action="store_true", required=False)
    parser.add_argument("--train_first_k", type=int, default=None, required=False)
    parser.add_argument("--attn_only",action="store_true", required=False)
    parser.add_argument("--random_init",action="store_true", required=False)
    parser.add_argument("--vae",action="store_true", required=False)
    parser.add_argument("--resample",action="store_true", required=False)
    parser.add_argument("--shared_outcome_space",action="store_true", required=False)
    parser.add_argument("--vae_gpu_num_samples", type=int, default=None, required=False)
    parser.add_argument("--vae_num_samples", type=int, default=None, required=False)
    parser.add_argument("--inference_every_t", type=int, default=None, required=False)

    parser.add_argument("--encoder_gating_mode", type=str, choices=["mult", "albo"],default="mult")

    # ablations
    parser.add_argument("--no_inference", action="store_true", default=False)
    parser.add_argument("--pure_inference", action="store_true", default=False)
    parser.add_argument("--multitask_data_source", required=False, default="pilot_study_3")
    args = parser.parse_args()
    return args


def save_checkpoint(args, checkpoint_name, model, step, n_examples, dev_logits, test_logits, dev_labels, test_labels):
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
    torch.save(dict(
        dev_preds=dev_logits,
        dev_labels=dev_labels,
        test_preds=test_logits,
        test_labels=test_labels
    ),
        os.path.join(checkpointdir, "predictions.bin")
    )

def evaluate_enc_dec(model, loader, data_config, shared_outcome_space):
    loss = 0
    n = 0
    with torch.no_grad():
        for gpu_batch in tqdm.tqdm(loader):
            z, x = (d.to("cuda") for i, d in enumerate(gpu_batch) if i < 2)
            if not shared_outcome_space:
                z = z + torch.arange(data_config["nvars"]).to("cuda") * data_config["nvals"]
            xinput = x[..., :-1].contiguous()
            xlabel = x[..., 1:].contiguous()
            px = torch.arange(xinput.size(1)).expand_as(xinput).to("cuda")
            pz = torch.arange(z.size(1)).expand_as(z).to("cuda")
            output = model(input_ids=z, decoder_input_ids=xinput, position_ids=pz, decoder_position_ids=px, labels=xlabel, return_dict=True)
            loss += output.loss.item() * x.size(0) # type:ignore
            n += x.size(0)
    return loss / n, None, None

def evaluate(model, loader, data_config, no_inference=False, uniform_noise=None, no_pos_z=False, gating_mode="albo", num_samples=None, is_full=False, bootstrap=None, bootstrap_cl=0.95, pure_inference=False, multitask_data_source="pilot_study_3", bos_id=None, reencode=False, eval_first_k=None, encoder_gating_mode=None, vae=False, vae_gpu_num_samples=None, vae_num_samples=None, prefix_value_dropout=None, shared_outcome_space=False, inference_every_t=None):
    if vae:
        return evaluate_enc_dec(model, loader, data_config, shared_outcome_space)
    loss = 0
    loss_first = 0
    loss_last = 0
    n = 0
    gating_mode_is_shared_sample = gating_mode.endswith("shared_sample")
    gating_mode_is_pre_sample = gating_mode.endswith("pre_sample")
    losses = []
    logits = []
    labels = []
    if bootstrap is not None or eval_first_k is not None:
        loss_fct = CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        model.eval()
        for gpu_batch in tqdm.tqdm(loader):
            if is_full:
                _, x, inf, cinf = (i.to("cuda") for i in gpu_batch)
            elif gating_mode_is_pre_sample:
                _, x, pre_samples = (i.to("cuda") for i in gpu_batch)
            else:
                _, x, inf = (i.to("cuda") for i in gpu_batch)
            if uniform_noise is not None:
                inf = utils3.add_uniform_noise(inf, uniform_noise)
            xinput = x[..., :-1].contiguous()
            if prefix_value_dropout is not None and prefix_value_dropout == 1.0:
                if shared_outcome_space:
                    dropped_out_xinput = xinput.clone()

                    dropped_out_xinput[:, [i for i in range(2, data_config["seq_len"], 2)]] = xinput[:, [i for i in range(1, data_config["seq_len"] -1, 2)]]
                else:
                    dropped_out_xinput = xinput // data_config["nvals"] + data_config["nvars"] * data_config["nvals"] + 1
                dropped_out_xinput[:,0] = model.decoder.config.bos_token_id # type:ignore
                xinput = dropped_out_xinput
            xlabel = x[..., 1:].contiguous()
            if shared_outcome_space:
                z_all = torch.arange(data_config["nvals"]).expand(x.size(0), data_config["nvars"],  data_config["nvals"]).reshape(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
            else:
                z_all = torch.arange(data_config["nvars"] * data_config["nvals"]).expand(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
            pz_all = torch.arange(data_config["nvars"]).repeat_interleave(data_config["nvals"]).expand(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
            px = torch.arange(xinput.size(1)).expand_as(xinput).to("cuda")
            if pure_inference:
                if multitask_data_source == "pilot_study_3":
                    xinput = xinput // data_config["nvals"]
                elif multitask_data_source == "pilot_study_4":
                    xinput = (
                        data_config["nvars"] * ((xinput // (data_config["nvals"] * data_config["nvars"])) // data_config["nvals"])
                        + (xinput % (data_config["nvals"] * data_config["nvars"])) //  data_config["nvals"])
                else:
                    raise ValueError(f"unknown multitask_data_source {multitask_data_source}")
                xinput[..., 0] = bos_id
            if no_pos_z:
                pz_all = pz_all.fill_(0)
            if gating_mode_is_shared_sample:
                infs = inf[:, :-1, :]
                if inference_every_t is not None:
                    if infs.size(1) % inference_every_t != 0:
                        raise ValueError()
                    infs = infs[:, :-1, :][:, list(range(0, infs.size(1), inference_every_t)) ,:].repeat_interleave(inference_every_t, 1)
                infs = infs.reshape(inf.size(0), data_config["seq_len"], data_config["nvars"], data_config["nvals"])
                samples = torch.distributions.Categorical(logits=infs).sample((num_samples,)) # type:ignore
                samples_one_hot = infs.new_zeros(num_samples, infs.size(0), infs.size(1), infs.size(2), infs.size(3), dtype=torch.bool)# type:ignore
                samples_one_hot = samples_one_hot.scatter_(-1, samples.unsqueeze(-1), 1).reshape(num_samples, infs.size(0), infs.size(1), infs.size(2) * infs.size(3)).permute(1,2,0,3) # type:ignore
            if gating_mode_is_pre_sample:
                samples_one_hot = tree_belief_propagation.int_to_bin_batched(data_config["nvars"], [data_config["nvals"]] * data_config["nvars"], pre_samples)
            if no_inference or gating_mode_is_shared_sample or gating_mode_is_pre_sample:
                log_marginals = None
            elif not is_full:
                log_marginals = inf[:,:-1,:]
            else:
                log_marginals = (inf[:,:-1,:], cinf[:,:-1,:])
            if inference_every_t is not None:
                # only works with single marginals currently
                log_marginals = log_marginals[:, list(range(0, log_marginals.size(1), inference_every_t)) ,:].repeat_interleave(inference_every_t, 1) # type:ignore

            # if reencode:
            #     z_all = z_all.repeat_interleave(data_config["seq_len"], dim=0)
            #     pz_all = pz_all.repeat_interleave(data_config["seq_len"], dim=0)
            #     xlabel = xlabel.repeat_interleave(data_config["seq_len"], dim=0)
            #     xinput = xinput.repeat_interleave(data_config["seq_len"], dim=0)
            #     xlabel_mask = xlabel.new_eye(data_config["seq_len"], dtype=torch.bool).unsqueeze(0).repeat_interleave(batch_size, dim=0).reshape(-1, data_config["seq_len"])
            #     xlabel[~xlabel_mask] = -100
            output = model(input_ids=z_all, position_ids=pz_all,
                        token_type_ids=z_all.new_zeros(z_all.size()),
                        labels=xlabel,
                        decoder_input_ids=xinput, decoder_position_ids=px,
                    decoder_encoder_samples=samples_one_hot if gating_mode_is_shared_sample or gating_mode_is_pre_sample else None,
                    decoder_encoder_log_marginals=log_marginals,
                    log_conditional_marginals=None if model.encoder.config.num_hidden_layers == 0 else  cinf[:,:-1,:], # type:ignore
                    attention_mode=encoder_gating_mode if model.encoder.config.num_hidden_layers > 0 and not no_inference else None,
                    decoder_encoder_attention_mode=gating_mode if not no_inference else None,
                    reencode=model.encoder.config.num_hidden_layers > 0 and not no_inference)
            logits.extend(output.logits.cpu().tolist())
            labels.extend(xlabel.cpu().tolist())
            if bootstrap is not None:
                loss_individual = (loss_fct(output["logits"].reshape(-1, model.decoder.config.vocab_size), xlabel.view(-1)).reshape(x.size(0), data_config["seq_len"])).mean(-1)
                losses.append(loss_individual.tolist())
            if eval_first_k is not None:
                loss_individual = (loss_fct(output["logits"].reshape(-1, model.decoder.config.vocab_size), xlabel.view(-1)).reshape(x.size(0), data_config["seq_len"]))
                loss_first += loss_individual[..., :eval_first_k].mean().item() * x.size(0)
                loss_last += loss_individual[..., eval_first_k:].mean().item() * x.size(0)
            loss += output.loss.item() * x.size(0) # type:ignore
            n += x.size(0)
        model.train()
    if bootstrap is not None:
        losses = list(itertools.chain(*losses))
        br = stats.bootstrap((losses,), np.mean, n_resamples=bootstrap, confidence_level=bootstrap_cl)
        return loss / n, br
    if eval_first_k is not None:
        return loss / n, loss_first / n, loss_last / n
    return loss / n, logits, labels

def load_model(model_path, no_pdrop=False, random_init=False, args=None, data_config=None):
    if model_path is None:
        if args is None or data_config is None:
            raise ValueError()
        encoder_config = bert.BertConfig(
            num_hidden_layers=0,
            vocab_size=data_config["encoder_vocab_size"],
            max_position_embeddings=data_config["nvars"] * data_config["nvals"],
            pad_token_id=data_config["encoder_pad_token_id"]
        )
        decoder_config = gpt2.GPT2Config(
            num_hidden_layers=args.n_layer,
            vocab_size=data_config["decoder_vocab_size"],
            max_position_embeddings=data_config["seq_len"] + 1,
            bos_token_id=data_config["decoder_bos_token_id"],
            add_cross_attention=True,
            is_decoder=True,
        )
        if no_pdrop:
            encoder_config.attention_probs_dropout_prob = 0.0
            decoder_config.attn_pdrop = 0.0
        if args.prefix_value_dropout is not None:
            # assume shared outcome space so do nothing
            pass
            # raise NotImplementedError()
            # decoder_config.vocab_size = decoder_config.vocab_size + data_config["nvars"]
        encoder = bert.BertModel(encoder_config)
        decoder = gpt2.GPT2LMHeadModel(decoder_config)
        def remove_persistent(d):
            return {k:v for k, v in d.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias") and not k.endswith("crossattention.bias") and not k.endswith("crossattention.masked_bias") }
        if not random_init:
            raise NotImplementedError()
            encoder.load_state_dict(torch.load(os.path.join(model_path, "encoder","pytorch_model.bin")))
            decoder.load_state_dict(remove_persistent(torch.load(os.path.join(model_path, "decoder","pytorch_model.bin"))))
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to("cuda")
    else:
        if args is None or data_config is None:
            raise ValueError()
        with open(os.path.join(model_path, "encoder","config.json")) as f:
            encoder_config = json.load(f)
            encoder_config = bert.BertConfig.from_dict(encoder_config)
        with open(os.path.join(model_path, "decoder","config.json")) as f:
            decoder_config = json.load(f)
            decoder_config = gpt2.GPT2Config.from_dict(decoder_config)
        if no_pdrop:
            encoder_config.attention_probs_dropout_prob = 0.0
            decoder_config.attn_pdrop = 0.0
        if args.prefix_value_dropout is not None:
            decoder_config.vocab_size = decoder_config.vocab_size + data_config["nvars"]
        encoder = bert.BertModel(encoder_config)
        decoder = gpt2.GPT2LMHeadModel(decoder_config)
        def remove_persistent(d):
            return {k:v for k, v in d.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias") and not k.endswith("crossattention.bias") and not k.endswith("crossattention.masked_bias") }
        if not random_init:
            encoder.load_state_dict(torch.load(os.path.join(model_path, "encoder","pytorch_model.bin")))
            decoder.load_state_dict(remove_persistent(torch.load(os.path.join(model_path, "decoder","pytorch_model.bin"))))
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder).to("cuda")
    return model, encoder, decoder

def main():
    args = parse_args()
    is_full = args.gating_mode == "albo_full"
    seeding.seed(args.seed)
    with open(os.path.join(args.output_dir, "log.jsonl"), "w") as f:
        pass

    if args.post_scale_gate:
        args.gating_mode = f"post_scale:{args.gating_mode}"
        args.encoder_gating_mode = f"post_scale:{args.encoder_gating_mode}"
    gating_mode_is_shared_sample = args.gating_mode.endswith("shared_sample")
    gating_mode_is_pre_sample = args.gating_mode.endswith("pre_sample")
    if (gating_mode_is_shared_sample or gating_mode_is_pre_sample) and is_full:
        raise NotImplementedError("don't support currently the use samples cond. marginals")
    if args.uniform_noise is not None and is_full:
        raise NotImplementedError("don't support currently the use of uniform noise with cond. marginals")

    # load data config
    with open(args.data_config) as f:
        data_config = json.load(f)

    # load model
    model, encoder, decoder = load_model(args.pretrained_model, no_pdrop=args.no_pdrop, random_init=args.random_init, args=args, data_config=data_config)
    is_full = is_full or (not args.no_inference and encoder.config.num_hidden_layers > 0)
    
    # load data
    with open(args.data_meta) as f:
        if args.multitask_data_source.startswith("pilot_study_3"):
            data_meta = json.load(f)
            theoretical_minimum = (data_meta["z_entropy"] +
                                math.log(data_config["n_branches"] ** (data_config["seq_len"]-data_config["n_branches"])
                                            * math.factorial(data_config["n_branches"]))
                                ) / data_config["seq_len"]
        elif args.multitask_data_source.startswith("pilot_study_4"):
            data_meta = json.load(f)
            theoretical_minimum = (data_meta["z_entropy"] +
                                math.log(13182)
                                ) / data_config["seq_len"]
        elif args.multitask_data_source.startswith("spelling"):
            theoretical_minimum = 0.0
        else:
            raise ValueError(f"Unknown multitask source {args.multitask_data_source}")
    train_dict = torch.load(args.train_file)
    dev_dict = torch.load(args.dev_file)
    test_dict = torch.load(args.test_file)

    train_dataset = TensorDataset(*((train_dict["z"], torch.cat([torch.empty((train_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), train_dict["x"]], dim=1))
                                    + ((train_dict["inf"],) if not gating_mode_is_pre_sample else tuple())
                                    + ((train_dict["cond_inf"],) if is_full else tuple())
                                    + ((train_dict["inf_samples"][:,:-1,:args.num_samples,:],) if gating_mode_is_pre_sample else tuple())))
    dev_dataset = TensorDataset(*((dev_dict["z"],torch.cat([torch.empty((dev_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), dev_dict["x"]], dim=1))
                                    + ((dev_dict["inf"],) if not gating_mode_is_pre_sample else tuple())
                                   + ((dev_dict["cond_inf"],) if is_full else tuple())
                                   + ((dev_dict["inf_samples"][:,:-1,:args.num_samples,:],) if gating_mode_is_pre_sample else tuple())))
    test_dataset = TensorDataset(*((test_dict["z"], torch.cat([torch.empty((test_dict["x"].size(0),1), dtype=torch.long).fill_(decoder.config.bos_token_id), test_dict["x"]], dim=1))
                                    + ((test_dict["inf"],) if not gating_mode_is_pre_sample else tuple())
                                    + ((test_dict["cond_inf"],) if is_full else tuple())
                                    + ((test_dict["inf_samples"][:,:-1,:args.num_samples,:],) if gating_mode_is_pre_sample else tuple())))
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
        if args.multitask_data_source == "pilot_study_3":
            multitask_data = utils3.encoder_decoder_train_data_generator(data_config["nvars"],
                                                                        data_config["nvals"],
                                                                        data_config["seq_len"],
                                                                        data_config["n_branches"],
                                                                        args.gpu_batch_size,
                                                                        data_config["x_model_seed"])
        elif args.multitask_data_source == "pilot_study_4":
            multitask_data = utils4.encoder_decoder_train_data_generator(data_config["nvars"],
                                                                        data_config["nvals"],
                                                                        data_config["seq_len"],
                                                                        data_config["n_branches"],
                                                                        args.gpu_batch_size,
                                                                        data_config["x_model_seed"])
        elif args.multitask_data_source == "pilot_study_3_same" or args.multitask_data_source == "pilot_study_4_same" or args.multitask_data_source == "spelling_same":
            print("Using same data for multitask")
            multitask_data = None
        else:
            raise ValueError(f"Unknown multitask source {args.multitask_data_source}")
    else:
        multitask_data = None

    # load optimizer
    params = model.parameters() if not args.attn_only else [p for n, p in model.named_parameters() if ".crossattention." in n or ".attention." in n]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # train
    step = 0
    best_dev_loss = None

    default_group = "finetune_encoder_decoder" + ("_no_inference" if args.no_inference else "_with_inference")
    default_project = "dblm"
    run = wandb.init(
        # Set the project where this run will be logged
        name=args.name,
        project=default_project if not args.project else args.project,
        group=f"{args.multitask_data_source}/{default_group if not args.group else args.group}",
        # Track hyperparameters and run metadata
        config=vars(args))
    bar = tqdm.tqdm(load_forever(train_dataloader), desc="loss=", total=args.train_steps)


    # evaluate once
    dev_loss, dev_logits, dev_labels = evaluate(model, dev_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode, num_samples=args.num_samples, is_full=is_full, multitask_data_source=args.multitask_data_source, bos_id=decoder.config.bos_token_id, eval_first_k=args.train_first_k, encoder_gating_mode=args.encoder_gating_mode, vae=args.vae, vae_gpu_num_samples=args.vae_gpu_num_samples, vae_num_samples=args.vae_num_samples, prefix_value_dropout=args.prefix_value_dropout, shared_outcome_space=args.shared_outcome_space, inference_every_t=args.inference_every_t)
    test_loss, test_logits, test_labels = evaluate(model, test_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode, num_samples=args.num_samples, is_full=is_full, multitask_data_source=args.multitask_data_source, bos_id=decoder.config.bos_token_id, eval_first_k=args.train_first_k, encoder_gating_mode=args.encoder_gating_mode, vae=args.vae, vae_gpu_num_samples=args.vae_gpu_num_samples, vae_num_samples=args.vae_num_samples, prefix_value_dropout=args.prefix_value_dropout, shared_outcome_space=args.shared_outcome_space, inference_every_t=args.inference_every_t)
    if args.train_first_k is not None:
        dev_loss, first_dev_loss, final_dev_loss = dev_loss
        test_loss, first_test_loss, final_test_loss = test_loss
    logline={
            "step": step,
            "n_examples": step * args.batch_size,
            "dev_loss": dev_loss,
            "test_loss": test_loss,
            "dev_loss_delta": dev_loss - theoretical_minimum,
            "test_loss_delta": test_loss - theoretical_minimum,
        }
    if args.train_first_k is not None:
        logline["dev_loss_in"] = first_dev_loss
        logline["dev_loss_out"] = final_dev_loss
        logline["test_loss_in"] = first_test_loss
        logline["test_loss_out"] = final_test_loss
    for gpu_step, gpu_batch in enumerate(bar):
        if args.vae:
            loss = 0
        else:
            # VAE training skips this
            if is_full:
                _, x, inf, cinf = (i.to("cuda") for i in gpu_batch)
            elif gating_mode_is_pre_sample:
                _, x, pre_samples = (i.to("cuda") for i in gpu_batch)
            else:
                _, x, inf = (i.to("cuda") for i in gpu_batch)
            if args.uniform_noise is not None:
                inf = utils3.add_uniform_noise(inf, args.uniform_noise)
            xinput = x[..., :-1].contiguous()
            if args.prefix_value_dropout is not None:
                if args.shared_outcome_space:
                    dropped_out_xinput = xinput.clone()
                    dropped_out_xinput[:, [i for i in range(2, data_config["seq_len"], 2)]] = xinput[:, [i for i in range(1, data_config["seq_len"] -1, 2)]]
                else:
                    dropped_out_xinput = xinput // data_config["nvals"] + data_config["nvars"] * data_config["nvals"] + 1
                dropout_switch = torch.rand(xinput.size()).to(xinput.device) < args.prefix_value_dropout
                dropped_out_xinput[:,0] = model.decoder.config.bos_token_id # type:ignore
                xinput = torch.where(dropout_switch, dropped_out_xinput, xinput)
            xlabel = x[..., 1:].contiguous()
            if args.shared_outcome_space:
                z_all = torch.arange(data_config["nvals"]).expand(x.size(0), data_config["nvars"],  data_config["nvals"]).reshape(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
            else:
                z_all = torch.arange(data_config["nvars"] * data_config["nvals"]).expand(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
            pz_all = torch.arange(data_config["nvars"]).repeat_interleave(data_config["nvals"]).expand(x.size(0), data_config["nvars"] * data_config["nvals"]).to("cuda")
            px = torch.arange(xinput.size(1)).expand_as(xinput).to("cuda")
            if args.pure_inference:
                if args.multitask_data_source == "pilot_study_3":
                    xinput = xinput // data_config["nvals"]
                elif args.multitask_data_source == "pilot_study_4":
                    xinput = (
                        data_config["nvars"] * ((xinput // (data_config["nvals"] * data_config["nvars"])) // data_config["nvals"])
                        + (xinput % (data_config["nvals"] * data_config["nvars"])) //  data_config["nvals"])
                else:
                    raise ValueError(f"unknown multitask_data_source {args.multitask_data_source}")
                xinput[..., 0] = decoder.config.bos_token_id
            if args.no_pos_z:
                pz_all = pz_all.fill_(0)
            if gating_mode_is_shared_sample:
                infs = inf[:, :-1, :]
                if args.inference_every_t is not None:
                    if infs.size(1) % args.inference_every_t != 0:
                        raise ValueError()
                    infs = infs[:, :-1, :][:, list(range(0, infs.size(1), args.inference_every_t)) ,:].repeat_interleave(args.inference_every_t, 1)
                infs = infs.reshape(inf.size(0), data_config["seq_len"], data_config["nvars"], data_config["nvals"])
                samples = torch.distributions.Categorical(logits=infs).sample((args.num_samples,)) # type:ignore
                samples_one_hot = infs.new_zeros(args.num_samples, infs.size(0), infs.size(1), infs.size(2), infs.size(3), dtype=torch.bool)
                samples_one_hot = samples_one_hot.scatter_(-1, samples.unsqueeze(-1), 1).reshape(args.num_samples, infs.size(0), infs.size(1), infs.size(2) * infs.size(3)).permute(1,2,0,3) # type:ignore
            if gating_mode_is_pre_sample:
                samples_one_hot = tree_belief_propagation.int_to_bin_batched(data_config["nvars"], [data_config["nvals"]] * data_config["nvars"], pre_samples)

            if args.no_inference or gating_mode_is_shared_sample or gating_mode_is_pre_sample:
                log_marginals = None
            elif not is_full:
                log_marginals = inf[:,:-1,:]
            else:
                log_marginals = (inf[:,:-1,:], cinf[:,:-1,:])

            if args.inference_every_t is not None:
                # only works with single marginals currently
                log_marginals = log_marginals[:, list(range(0, log_marginals.size(1), args.inference_every_t)) ,:].repeat_interleave(args.inference_every_t, 1) # type:ignore
            if args.train_first_k is not None:
                xlabel[..., args.train_first_k:] = -100
            output = model(input_ids=z_all, position_ids=pz_all,
                        token_type_ids=z_all.new_zeros(z_all.size()),
                        decoder_input_ids=xinput, decoder_position_ids=px,
                        labels=xlabel,
                        decoder_encoder_samples=samples_one_hot if gating_mode_is_shared_sample or gating_mode_is_pre_sample else None,
                        decoder_encoder_log_marginals=log_marginals,
                        log_conditional_marginals=None if model.encoder.config.num_hidden_layers == 0 else  cinf[:,:-1,:], # type:ignore
                        attention_mode=args.encoder_gating_mode if model.encoder.config.num_hidden_layers > 0 and not args.no_inference else None, # type:ignore
                        decoder_encoder_attention_mode=args.gating_mode if not (args.no_inference) else None,
                        reencode=model.encoder.config.num_hidden_layers > 0 and not args.no_inference) # type:ignore
            loss = output.loss # type:ignore
            bar.set_description(f"loss={loss.item():.2f}")

        if args.multitask_lambda is not None:
            if args.resample:
                _, x, inf = (i.to("cuda") for i in gpu_batch)
                positions = list(range(data_config["seq_len"]))
                z_offsets = (data_config["nvals"] * torch.arange(data_config["nvars"])).to("cuda")[None, :]
                xinput = x[..., :-1].contiguous()[None, :, None, :].expand(1, args.gpu_batch_size, data_config["seq_len"], data_config["seq_len"]).reshape(1 * args.gpu_batch_size * data_config["seq_len"], data_config["seq_len"])
                xlabel = x[..., 1:].contiguous()[None, :, None, :].expand(1, args.gpu_batch_size, data_config["seq_len"], data_config["seq_len"]).reshape(1 * args.gpu_batch_size * data_config["seq_len"], data_config["seq_len"])
                pz = torch.arange(data_config["nvars"]).expand(1 * args.gpu_batch_size * data_config["seq_len"], data_config["nvars"]).to("cuda")
                px = torch.arange(xinput.size(-1)).expand_as(xinput).to("cuda")
                infs = inf[:, :-1, :]
                if args._get_argsinference_every_t is not None:
                    if infs.size(1) % args.inference_every_t != 0:
                        raise ValueError()
                    infs = infs[:, :-1, :][:, list(range(0, infs.size(1), args.inference_every_t)) ,:].repeat_interleave(args.inference_every_t, 1)
                infs = infs.reshape(inf.size(0), data_config["seq_len"], data_config["nvars"], data_config["nvals"])
                samples = torch.distributions.Categorical(logits=infs).sample((1,)) # type:ignore gpu_samples, batch, seq, nvars
                samples = samples.reshape(1 * args.gpu_batch_size * data_config["seq_len"], data_config["nvars"])
                if not args.shared_outcome_space:
                    samples = samples + z_offsets
                if args.prefix_value_dropout is not None:
                    if args.shared_outcome_space:
                        dropped_out_xinput = xinput.clone()
                        dropped_out_xinput[:, [i for i in range(2, data_config["seq_len"], 2)]] = xinput[:, [i for i in range(1, data_config["seq_len"] -1, 2)]]
                    else:
                        dropped_out_xinput = xinput // data_config["nvals"] + data_config["nvars"] * data_config["nvals"] + 1
                    dropout_switch = torch.rand(xinput.size()).to(xinput.device) < args.prefix_value_dropout
                    dropped_out_xinput[:,0] = model.decoder.config.bos_token_id # type:ignore
                    xinput = torch.where(dropout_switch, dropped_out_xinput, xinput)
                if args.uniform_noise is not None:
                    # mix with marginal noise
                    noise_switch = torch.rand(samples.size()).to(samples.device) < args.uniform_noise # type:ignore
                    samples_noise = (torch.randint(0, data_config["nvals"], samples.size()) + torch.arange(data_config["nvars"]) * data_config["nvals"]).to(samples.device)
                    samples = torch.where(noise_switch, samples_noise, samples)
                output = model(input_ids=samples, position_ids=pz,
                            token_type_ids=samples.new_zeros(samples.size()),
                            labels=xlabel,
                            decoder_input_ids=xinput, decoder_position_ids=px)
                loss_fct = CrossEntropyLoss(reduction="none")
                reg_loss = (loss_fct(output["logits"].reshape(-1, model.decoder.config.vocab_size), xlabel.view(-1)).reshape(1, args.gpu_batch_size, data_config["seq_len"], data_config["seq_len"])[:, :, positions, positions]).mean() #type:ignore
                actual_loss = loss + reg_loss * args.multitask_lambda
                if args.vae:
                    bar.set_description(f"loss={actual_loss.item():.2f}")
                    loss = actual_loss
            else:
                if args.multitask_data_source[-4:] == "same":
                    z, x = (d.to("cuda") for i, d in enumerate(gpu_batch) if i < 2)
                    if not args.shared_outcome_space:
                        z = z + torch.arange(data_config["nvars"]).to("cuda") * data_config["nvals"]
                    xinput = x[..., :-1].contiguous()
                    xlabel = x[..., 1:].contiguous()
                    px = torch.arange(xinput.size(1)).expand_as(xinput).to("cuda")
                    pz = torch.arange(z.size(1)).expand_as(z).to("cuda")
                else:
                    z, xinput, xlabel, pz, px = (i.to("cuda") for i in next(multitask_data)) # type:ignore
                if args.prefix_value_dropout is not None:
                    if args.shared_outcome_space:
                        dropped_out_xinput = xinput.clone()
                        dropped_out_xinput[:, [i for i in range(2, data_config["seq_len"], 2)]] = xinput[:, [i for i in range(1, data_config["seq_len"] -1, 2)]]
                    else:
                        dropped_out_xinput = xinput // data_config["nvals"] + data_config["nvars"] * data_config["nvals"] + 1
                    dropout_switch = torch.rand(xinput.size()).to(xinput.device) < args.prefix_value_dropout #type:ignore
                    dropped_out_xinput[:,0] = model.decoder.config.bos_token_id # type:ignore
                    xinput = torch.where(dropout_switch, dropped_out_xinput, xinput)
                if args.uniform_noise is not None:
                    # mix with marginal noise
                    noise_switch = torch.rand(z.size()).to(z.device) < args.uniform_noise # type:ignore
                    z_noise = torch.randint(0, data_config["nvals"], z.size())
                    if not args.shared_outcome_space:
                        z_noise = z_noise + torch.arange(data_config["nvars"]) * data_config["nvals"]
                    z_noise = z_noise.to(z.device)
                    z = torch.where(noise_switch, z_noise, z)
                reg_output = model(input_ids=z, decoder_input_ids=xinput, position_ids=pz, decoder_position_ids=px, labels=xlabel, return_dict=True)
                reg_loss = reg_output.loss
                actual_loss = loss + reg_loss * args.multitask_lambda
                if args.vae:
                    bar.set_description(f"loss={actual_loss.item():.2f}")
                    loss = actual_loss
        else:
            if args.vae:
                raise ValueError("VAE models must specify a nonzero multitask_lambda, should be 1.0 by default")
            actual_loss = loss

        ((args.gpu_batch_size / args.batch_size) * actual_loss).backward() # gradient accumulation
        if (gpu_step +1) % (args.batch_size // args.gpu_batch_size) == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if (step % args.logging_steps == 0) and step % args.eval_steps != 0:
                logline={
                        "train_loss": loss.item(), #type:ignore
                        "step": step,
                        "n_examples": step * args.batch_size,
                        "train_loss_delta": loss.item() - theoretical_minimum, #type:ignore
                    }
                if args.multitask_lambda is not None:
                    logline["reg_loss"] = reg_loss.item() # type:ignore
                    logline["actual_loss"] = actual_loss.item() # type:ignore
                    logline["actual_loss_delta"] = actual_loss.item() - theoretical_minimum # type:ignore
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline, step=step)
            if (step == 1):
                logline["train_loss"] = loss.item() #type:ignore
                logline["train_loss_delta"] = loss.item() - theoretical_minimum #type:ignore
                if args.multitask_lambda is not None:
                    logline["reg_loss"] = reg_loss.item() # type:ignore
                    logline["actual_loss"] = actual_loss.item() # type:ignore
                    logline["actual_loss_delta"] = actual_loss.item() - theoretical_minimum # type:ignore
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline, step=0)
            if (step % args.eval_steps == 0):
                dev_loss, dev_logits, dev_labels = evaluate(model, dev_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode, num_samples=args.num_samples, is_full=is_full, multitask_data_source=args.multitask_data_source, bos_id=decoder.config.bos_token_id, eval_first_k=args.train_first_k, encoder_gating_mode=args.encoder_gating_mode, vae=args.vae, vae_gpu_num_samples=args.vae_gpu_num_samples, vae_num_samples=args.vae_num_samples, prefix_value_dropout=args.prefix_value_dropout, shared_outcome_space=args.shared_outcome_space, inference_every_t=args.inference_every_t)
                test_loss, test_logits, test_labels = evaluate(model, test_dataloader, data_config, args.no_inference, uniform_noise=args.uniform_noise, no_pos_z=args.no_pos_z, gating_mode=args.gating_mode, num_samples=args.num_samples, is_full=is_full, multitask_data_source=args.multitask_data_source, bos_id=decoder.config.bos_token_id, eval_first_k=args.train_first_k, encoder_gating_mode=args.encoder_gating_mode, vae=args.vae, vae_gpu_num_samples=args.vae_gpu_num_samples, vae_num_samples=args.vae_num_samples, prefix_value_dropout=args.prefix_value_dropout, shared_outcome_space=args.shared_outcome_space, inference_every_t=args.inference_every_t)
                if args.train_first_k is not None:
                    dev_loss, first_dev_loss, final_dev_loss = dev_loss
                    test_loss, first_test_loss, final_test_loss = test_loss
                if best_dev_loss is None or dev_loss < best_dev_loss:# type:ignore
                    best_dev_loss = dev_loss
                    save_checkpoint(args, f"checkpoint-early-stopping", model, step, step * args.batch_size, dev_logits=dev_logits, test_logits=test_logits, dev_labels=dev_labels, test_labels=test_labels)
                logline={
                        "train_loss": loss.item(), #type:ignore
                        "step": step,
                        "n_examples": step * args.batch_size,
                        "dev_loss": dev_loss,
                        "test_loss": test_loss,
                        "train_loss_delta": loss.item() - theoretical_minimum, #type:ignore
                        "dev_loss_delta": dev_loss - theoretical_minimum,
                        "test_loss_delta": test_loss - theoretical_minimum,
                    }
                if args.train_first_k is not None:
                    logline["dev_loss_in"] = first_dev_loss
                    logline["dev_loss_out"] = final_dev_loss
                    logline["test_loss_in"] = first_test_loss
                    logline["test_loss_out"] = final_test_loss
                if args.multitask_lambda is not None:
                    logline["reg_loss"] = reg_loss.item() # type:ignore
                    logline["actual_loss"] = actual_loss.item() # type:ignore
                    logline["actual_loss_delta"] = actual_loss.item() - theoretical_minimum # type:ignore
                with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                    print(json.dumps(logline), file=f)
                wandb.log(logline, step=step)
            if step >= args.train_steps:
                break
    # save_checkpoint(args, "checkpoint-final", model, step, step * args.batch_size)


if __name__ == "__main__":
    main()
