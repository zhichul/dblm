import json
import encoder_factored_logprobs
import decoder_factored_logprobs

with open("best_models.json") as f:
    d = json.load(f)

for arch in ["Nested-Oracle", "No-Inference"]:
    for size in d[arch]:
        location = d[arch][size]
        print(arch, size, location)
        encoder_factored_logprobs.compute(location)

for arch in ["Regular-l24-h888"]:
    for size in d[arch]:
        location = d[arch][size]
        print(arch, size, location)
        decoder_factored_logprobs.compute(location)
