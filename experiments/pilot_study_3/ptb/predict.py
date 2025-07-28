import torch
from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("out/")

c2i = {c:i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
i2c = {i+1:c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
i2c[0]="#"
i2c[27]="*"
i2c[28]="_"
c2i["#"]=0
c2i["*"]=27
c2i["_"]=28

def decode(lis):
    return "".join([i2c[i] for i in lis])
def encode(s):
    l = [0] * 12
    for i, c in enumerate(s[:12]):
        l[i] = c2i[c]
    return l

while True:
    items = input("Enter a word, will be padded to 12 or truncated...\n").split()
    if len(items) == 1:
        s = items[0]
        temp = 1.0
    elif len(items) == 2:
        temp = float(items[1])
        s = items[0]
    else:
        print("bad format, continuing")
        continue
    input_ids = torch.tensor(encode(s)).unsqueeze(0)
    logits = model(input_ids).logits[0] # type:ignore
    sample = torch.distributions.Categorical(logits=logits / temp).sample((1,))[0] # type:ignore
    is_mask = input_ids == 27
    output = torch.where(is_mask, sample, input_ids)
    print(decode(output.view(-1).tolist()))
