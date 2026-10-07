# Grok Open-Weight Review

## Available releases and size

[Grok-1](https://github.com/xai-org/grok-1) is an Apache-2.0 open-weight,
text-only mixture-of-experts model with 314 billion parameters, eight experts, and two
experts active per token. xAI warns that its reference implementation needs substantial
GPU memory and is not optimized.

[Grok-2](https://huggingface.co/xai-org/grok-2) is a post-trained, text-only checkpoint
released under the Grok 2 Community License. Its official model card says the download is
approximately 500 GB in 42 files and the supplied SGLang recipe requires tensor
parallelism across eight GPUs, each with more than 40 GB of memory, even with FP8 serving.
It is not offered by a Hugging Face inference provider in that card.

These public checkpoints are not the hosted Grok 4.6 condition tested through OpenRouter.
They also do not expose a vision encoder or multimodal image-input contract, so they
cannot run this project's image-conditioned IAT task directly.

## Does open Grok have DPO?

DPO is a fine-tuning algorithm, not a capability flag embedded in a checkpoint. A team
could in principle implement preference tuning around a compatible text checkpoint, but
xAI's public Grok-1/Grok-2 materials do not document a DPO recipe. The
[Grok 4 model card](https://data.x.ai/2025-08-20-grok-4-model-card.pdf) describes
pretraining, supervised fine-tuning, and reinforcement learning using human feedback,
verifiable rewards, and model grading. It does not name DPO. Therefore this study records
`public evidence of DPO: not found`, not `Grok was not trained with DPO`.

## What happens to the IAT word prompt?

The historical prompt actually contains 61 permitted words: 30 positive and 31 negative.
Hosted multimodal Grok 4.6 can receive that prompt with a face image and select three
terms, which is the operational image-association test used here.

Sending only the same word list to open Grok-1 or Grok-2 would remove the race stimulus.
Any resulting positive/negative distribution would measure a text-only lexical prior, not
Black-versus-White visual association. A faithful test of the open checkpoints would need
an independently validated vision adapter or a different text-only race benchmark; either
would answer a different research question and is not mixed with hosted Grok 4.6 results.

## Practical implication

Running Grok-2 locally would require roughly 500 GB of storage plus an eight-GPU serving
node. Grok-1 is even less practical in the supplied reference implementation. Because the
current study is inference-only and image-conditioned, the pinned hosted Grok 4.6 route is
the reproducible condition within the available budget. It must be labeled as
closed-source and hosted, not as the open-weight Grok release.
