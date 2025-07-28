from typing import List, Optional
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HF_HOME"]="/export/a02/huggingface"
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
import os
import torch
import tqdm
import numpy as np
def serialize(tokenizer, chunks: List[List[str]], weights: List[Optional[List[float]]], log_marginals_dtype=torch.float):
    if tokenizer.add_bos_token == True:
        raise ValueError()
    tokens = tokenizer(sum(chunks, []), return_tensors="pt", padding=True).to("cuda")
    ser_input_ids = tokens["input_ids"]
    ser_attention_mask = tokens["attention_mask"]
    length = ser_attention_mask.sum()
    input_ids = ser_input_ids.new_zeros(length + 1)
    min = torch.finfo(log_marginals_dtype).min
    log_marginals = ser_input_ids.new_zeros((length + 1, length + 1), dtype=log_marginals_dtype).fill_(min).triu(diagonal=1)
    attention_mask = torch.tril(ser_input_ids.new_ones((length + 1, length + 1)))
    serial_index = 0
    chunk_offset = 1
    alt_offset = 1
    input_ids[0] = tokenizer.bos_token_id
    for alts, ws in zip(chunks, weights):
        ws = ws if ws is not None else [0.0]
        if (len(alts) != len(ws)):
            raise ValueError(f"chunks length and weight length must match. chunk"
                            f" {serial_index} erred.\n{chunks}\n{weights}")
        for alt, w in zip(alts, log_marginals.new_tensor(ws).log_softmax(-1)):
            l = ser_attention_mask[serial_index].sum()
            input_ids[alt_offset:alt_offset+l] = ser_input_ids[serial_index][:l]
            attention_mask[alt_offset:alt_offset+l, chunk_offset:alt_offset] = 0
            log_marginals[alt_offset:alt_offset+l, chunk_offset:alt_offset] = min
            log_marginals[alt_offset+l:, alt_offset:alt_offset+l] = w
            alt_offset += l
            serial_index += 1
        chunk_offset = alt_offset
    return input_ids, attention_mask, log_marginals

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", add_bos_token=False, pad_token="<unk>")
N = 10

# [30, 30, 30, 35, 35, 30, 30, 30, 35, 35, 25, 35, 35, 35, 35, 35, 35, 35, 45, 55, 0, 25, 25, 35, 35, 35, 35, 45, 50]
weights = [0.0, 0.0]
# [18, 25, 25, 25, 25, 25, 25, 25, 25, 25]
# weights = [0.0, 1.0]
# [25, 35, 45, 55, 55, 55, 60, 65, 65, 65]
# weights = [1.0, 0.0]

# [20, 25, 25, 25, 25, 25, 25, 25, 32]
# weights = [0.0, 2.0]
# [18, 40, 45, 55, 65, 65, 65]
# weights = [2.0, 0.0]

input_ids, attention_mask, log_marginals = serialize(tokenizer, [["<s>[INST] <<SYS>>\nYou are a helpful assistant that adheres to the format required by the user.\n<</SYS>> Please answer the following question about an imagined uncertain situation: "], 
["Alice is a teenager. ", "Alice died of old age. ", ],["How old is Alice? Please answer in a single number.[/INST] Given the provided information, one possibility of Alice's age is"]], [None, weights, None], log_marginals_dtype=torch.float16)
# print(tokenizer.decode(input_ids))
# print(attention_mask)
# print(log_marginals.exp())
print(input_ids.size())
ages = []
for i in tqdm.tqdm(range(N)):
    print(f"### Run {i} ###")
    generation_config = GenerationConfig(
        max_length=128,
        max_new_tokens=128,
        temperature=1.0,
        do_sample=True,
        num_return_sequences=1,
        use_cache=True
    )
    outputs = model.generate(input_ids=input_ids[None, ...], generation_config=generation_config, attention_mode="albo-b", log_marginals=log_marginals[None, ...])
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    print(generated)
    try:
        ages.append(int(re.search(r"[^\d]+(\d+)[^\d]+", generated).group(1)))
    except:
        pass
    print(f"### End ###")
    # print()
print(sorted(ages))
print(np.mean(ages))
print(np.median(ages))
print(np.std(ages))
# inputs = tokenizer(["Alice is a widow.", "Alice is a new bride."], return_tensors="pt", padding=True).to("cuda")
# print(tokenizer.decode(inputs["input_ids"][0]))

# <s>[INST] <<SYS>>
# {{ system_prompt }}
# <</SYS>>

# {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]