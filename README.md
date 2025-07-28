# Database Language Models

Codebase for langauge model architectures that condition on externally provided probability distributions, as well as joint architectures that have a graphical model component and a neural lm component. (Thinking of probability distributions as "probabilistic databses" about the variables.

# Core Code
Core interfaces and model definitions are in `src/dblm/core`. Attention mechanisms for encoding distributions is in `src/dblm/rva`.

# Interleaved Language Modeling and Probabilistic Inference
`pilot_study_1` and `pilot_study_2` study interleaved language modeling on data synthesized from an interleaved model.

# Conditioning on Distributions
`pilot_study_3` studies training a language model to condition on an entire joint distribution as input. Possible applications include conditioning on simulation tools that output joint distributions over variables of interest.
