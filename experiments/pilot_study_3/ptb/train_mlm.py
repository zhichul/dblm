import math
import os
import random
import torch
import tqdm
from transformers import BertForMaskedLM, BertConfig
import wandb
from dblm.utils import seeding
from dblm.experiments.filter_ptb import filter_fn
import numpy as np

seeding.seed(139584)

vocab_size = 30 # empty is 0, mask is 27, pad is 28 (never used), 29 is extra token
pad_id = 28
mask_id = 27
c2i = {c:i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
i2c = {i+1:c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
i2c[0]="#"
i2c[27]="*"
i2c[28]="_"
recompile_dataset = True
context_length = 12

# training args
max_iters = 100000
eval_interval = 100
eval_batchs = 100
batch_size = 4096
lr=6e-4

def decode(lis):
    return "".join([i2c[i] for i in lis])

def compile(file, name):
    xs = []
    with open(file) as f:
        for line in f:
            for token in filter(filter_fn, line.strip().split(" ")):
                x = [0] * context_length
                for i, char in enumerate(token[:context_length]):
                    x[i] = c2i[char]
                xs.append(x)
    xs = np.array(xs)
    dtype = np.uint8
    arr = np.memmap(name, dtype=dtype, mode='w+', shape=(xs.shape[0], xs.shape[1]))
    arr[:] = xs[:]
    print(f"integerized and compiled {file} to {name}, {xs.shape[0]} x {xs.shape[1]}, {dtype}")

if recompile_dataset or not os.path.exists("train.bin"):
    compile("ptb.train.txt", "train.bin")
train_data = np.memmap('train.bin', dtype=np.uint8, mode='r').reshape(-1, context_length)
if recompile_dataset or not os.path.exists("val.bin"):
    compile("ptb.valid.txt", "val.bin")
val_data = np.memmap('val.bin', dtype=np.uint8, mode='r').reshape(-1, context_length)
if recompile_dataset or not os.path.exists("test.bin"):
    compile("ptb.test.txt", "test.bin")
test_data = np.memmap('test.bin', dtype=np.uint8, mode='r').reshape(-1, context_length)

config = BertConfig(vocab_size=vocab_size,
                    hidden_size=768//2,
                    num_hidden_layers=12//2,
                    intermediate_size=3072//2,
                    num_attention_heads=12//2,
                    max_position_embeddings=context_length,
                    pad_token_id=pad_id,)
model = BertForMaskedLM(config)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def load_once(data, randomize=True, drop_last=True, first=None):
    inds = list(range(data.shape[0]))
    if randomize:
        random.shuffle(inds)
    for i, off in enumerate(range(0, len(inds), batch_size)):
        if first is not None and i >= first:
            break
        if drop_last and len(inds) - off < batch_size:
            break
        yield (data[inds[off:off+batch_size]],)

def load_forever(data):
    while True:
        yield from load_once(data)

run = wandb.init(
    project="char_mlm",
    config=dict(
        vocab_size = vocab_size,
        pad_id = pad_id,
        mask_id = mask_id,
        c2i = c2i,
        recompile_dataset = recompile_dataset,
        context_length = context_length,

        # training args
        max_iters = max_iters,
        eval_interval = eval_interval,
        eval_batchs = eval_batchs,
        batch_size = batch_size,
        lr=lr,
    ))
eval_data = test_data
eval_best = math.inf
model.to("cuda")
bar = tqdm.tqdm(zip(range(max_iters), load_forever(train_data)), total=max_iters)
for step, batch in bar:
    x, = batch
    x = torch.from_numpy(x).to("cuda").to(torch.long)
    num_masks = torch.randint(0, context_length, (x.size(0),))
    is_mask = torch.stack([torch.randperm(context_length) for _ in range(batch_size)]) < num_masks[:,None]
    labels = x.clone()
    #labels[~is_mask] = -100
    input_ids = x.clone()
    input_ids[is_mask] = mask_id
    output = model(input_ids=input_ids, labels=labels)

    output.loss.backward()
    optimizer.step()
    model.zero_grad()
    state = {"train_loss": output.loss.item()}
    bar.set_description_str(f"{output.loss.item():.2f}")
    if step % eval_interval == 0:
        model.eval()
        eval_loss = 0
        eval_n = 0
        for _, batch in tqdm.tqdm(zip(range(eval_batchs), load_forever(eval_data)), total=eval_batchs):
            x, = batch
            x = torch.from_numpy(x).to("cuda").to(torch.long)
            num_masks = torch.randint(0, context_length, (x.size(0),))
            is_mask = torch.stack([torch.randperm(context_length) for _ in range(batch_size)]) < num_masks[:,None]
            labels = x.clone()
            #labels[~is_mask] = -100
            input_ids = x.clone()
            input_ids[is_mask] = mask_id
            output = model(input_ids=input_ids, labels=labels)
            eval_n += 1
            eval_loss += output.loss.item()
        eval_loss /= eval_n
        model.train()
        if eval_best > eval_loss:
            eval_best = eval_loss
            os.makedirs("out", exist_ok=True)
            model_file = "out/pytorch_model.bin"
            config_file = "out/config.json"

            # save model and config
            torch.save(model.state_dict(), model_file)
            model.config.to_json_file(config_file)
            print("saving checkpoint to out/")
        print(f"eval loss: {eval_loss}, eval best: {eval_best}")
        state["val_loss"] = eval_loss
    wandb.log(state, step=step)


