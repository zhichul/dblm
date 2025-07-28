

import torch
from dblm.core.modeling import gpt2


model = gpt2.GPT2LMHeadModel.from_pretrained(
    "/export/a02/artifacts/dblm/experiments/pilot_study_2/pretrained_seq2seq_models/11/1/default/checkpoint-early-stopping"
)
output = model(
    input_ids=torch.tensor([[16, 23, 23, 23, 26, 29],
                            [16, 31, 25, 31, 17, 17]]),
    encoder_hidden_states=model.transformer.wte(torch.tensor([[0, 6, 9, 12],  #type:ignore
                                                              [0, 5, 8, 14]])),
    output_attentions=True,
) #type:ignore

breakpoint()
xx = 1
