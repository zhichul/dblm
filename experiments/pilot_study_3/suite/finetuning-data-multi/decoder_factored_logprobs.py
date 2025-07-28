import json
import os
import sys
import torch
import tqdm
from transformers.models import bert, encoder_decoder
from dblm.core.modeling import gpt2
from dblm.experiments.pilot_study_3 import utils
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

NVARS=10
NVALS=7

def evaluate(model, loader):
    model.eval()
    ce_vars = []
    ce_vals = []
    with torch.no_grad():
        for gpu_batch in tqdm.tqdm(loader, leave=False):
            _, x, _ = (i.to("cuda") for i in gpu_batch)
            px = torch.arange(x.size(1)).expand_as(x).to("cuda")
            output = model(input_ids=x, position_ids=px, labels=x)
            output.logits = output.logits[:,:-1,:]
            xlabel = x[:,1:].contiguous()
            x_index = (xlabel // NVALS * NVALS)[..., None] + torch.arange(NVALS, dtype=torch.long, device="cuda")
            Z = output.logits.logsumexp(dim=-1)
            group = output.logits.gather(dim=-1, index=x_index)
            ce_var = -(group.logsumexp(dim=-1) - Z)
            ce_val = -group.log_softmax(dim=-1).gather(dim=-1, index=xlabel[..., None] % NVALS).squeeze(-1)
            ce_vars.append(ce_var)
            ce_vals.append(ce_val)
    return torch.cat(ce_vars, dim=0), torch.cat(ce_vals, dim=0)

def compute(dir):
    # load model
    model = gpt2.GPT2LMHeadModel.from_pretrained(os.path.join(dir, "checkpoint-early-stopping"))
    dev = torch.load("../../data/nvars=10-nvals=7-zseed=42-seq_len=10-nbranches=3-xseed=42-mean=0.0-std=1.0-sseed=42-N=100000/dev.bin")

    dataset = TensorDataset(dev["z"], torch.cat([torch.empty((dev["x"].size(0),1), dtype=torch.long).fill_(model.config.bos_token_id), dev["x"]], dim=1), dev["inf"]) # type:ignore
    dataloader = DataLoader(dataset,
                            sampler=SequentialSampler(dataset),
                            batch_size=64)
    model.to("cuda") # type:ignore
    ce_vars, ce_vals = evaluate(model, dataloader)
    torch.save({"variable_cross_entropy": ce_vars.detach().to("cpu"), "value_cross_entropy": ce_vals.detach().to("cpu")},
               os.path.join(dir, "checkpoint-early-stopping", "dev_cross_entropy_breakdown.bin"))
    print((ce_vars.mean()+ce_vals.mean()).item())

if __name__ == "__main__":
    for dir in sys.argv[1:]:
        compute(dir)
