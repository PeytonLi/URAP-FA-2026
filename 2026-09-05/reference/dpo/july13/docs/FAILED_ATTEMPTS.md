# Failed Attempts and Lessons

This is the maintained record of configurations that failed technically or did not support a debiasing claim. It is intended to prevent repeated experiments and to keep smoke-test outcomes separate from research results.

## Summary

| Attempt | Outcome | Cause or limitation | Current correction |
| --- | --- | --- | --- |
| April 29 pilot adapter | Ran, but did not demonstrate debiasing | Only one unique DPO pair survived; it was positive-vs-positive and did not encode the intended fairness target | Require a direct reduction in negative terms or identity leakage; reject positive-vs-positive pairs by default |
| Global eager attention with generic LoRA targets | H100 80 GB CUDA OOM | Generic projection names also matched the SigLIP `vision_tower`, so DPO retained large trainable vision-attention graphs for both chosen and rejected forwards | Exclude `.*vision_tower.*`, freeze the visual encoder, and assert that no vision parameter is trainable |
| Tiny model-observed pair audit | Zero observed pairs | Of 128 candidates, almost all valid outputs had zero negative terms; no image had both a suitable chosen and rejected output | Run the larger staged pair audit and review pair diagnostics before training |
| Direct PEFT wrapper through the image-text pipeline | Compatibility warning during adapter evaluation | The pipeline did not officially list the PEFT wrapper class as a supported model type | Merge LoRA weights into the supported Gemma evaluation model before adapter inference |
| Eight-image smoke evaluation | Zero negatives in both conditions | One response per image was far too small to estimate a low-frequency negative-term rate | Use 100 held-out images per group and 10 generations per image; do not call the smoke result an efficacy result |
| July 13 final 500-update checkpoint | Negative terms nearly vanished, but response diversity collapsed | Fifty-eight unique pairs were cycled for 500 optimizer updates after DPO loss had already saturated near step 60 | Evaluate earlier checkpoints and retrain for fewer updates with a lower learning rate and diversity-aware early stopping |

## H100 OOM: Exact Interpretation

The failed trace reported:

- GPU capacity: 79.18 GiB
- total GPU memory in use: approximately 78.79 GiB
- memory allocated by PyTorch: approximately 77.73 GiB
- free memory: approximately 390 MiB
- failed allocation request: 512 MiB
- failure location: SigLIP vision attention while constructing `query_states @ key_states.transpose(...)`

The defensible conclusion is **not** that Gemma eager attention always requires 78.8 GiB. The failed configuration combined all of the following:

1. `google/gemma-3-12b-it`, including its SigLIP image encoder;
2. one global `attn_implementation="eager"` setting, applied to both the language and vision backbones;
3. generic LoRA targets such as `q_proj`, `k_proj`, `v_proj`, and `o_proj`, which also matched modules inside `vision_tower`;
4. trainable vision attention, which required autograd to retain its intermediate activations; and
5. DPO chosen and rejected policy forwards before one backward pass, leaving both computation graphs live at the peak.

Eager attention computes the standard scaled dot-product operation with ordinary framework operations. A basic implementation materializes the attention-score matrix, whose memory grows quadratically with sequence length. Vision encoders can have many image tokens, so making those attention projections trainable made this particularly expensive. Optimized SDPA or FlashAttention backends rearrange the same mathematics to reduce memory traffic and avoid retaining the full matrix in the same way.

Eager attention is not a new attention algorithm. Scaled dot-product attention was introduced with the Transformer in 2017. `eager` is a newer Hugging Face backend label for the direct/manual implementation, in contrast to backends such as PyTorch SDPA and FlashAttention. The pinned Transformers 4.50 Gemma 3 implementation emitted a training warning recommending eager attention, which is why the smoke test selected it. That warning did not imply that the vision tower should also receive LoRA adapters.

## Corrected Memory Result

The corrected smoke kept eager attention but added:

```python
exclude_modules = r".*vision_tower.*"
```

It also checks trainable parameter names and raises an error if any contain `vision_tower`. The frozen SigLIP encoder still produces visual features; only language-side LoRA projections are updated.

Measured on the same H100 class:

- trainable language-side LoRA parameters: 65,470,464
- peak PyTorch allocation: 42.59 GiB
- peak PyTorch reservation: 43.96 GiB
- device capacity: 79.18 GiB

This successful comparison isolates the important factor: eager attention was compatible with the smoke run after the vision tower was excluded. The accidental trainable vision path, combined with DPO's two policy graphs, caused the earlier peak.

## Would More GPUs Fix It?

Merely attaching a second H100 would not provide a validated fix for the failed configuration.

- The validated workflow is single-process and single-GPU and does not launch a distributed trainer. Two 80 GB devices do not automatically become one 160 GB memory pool.
- Standard data parallelism or DDP replicates the model and each sample's activations on every GPU. It improves throughput but normally does not lower the per-GPU peak that caused this OOM.
- `device_map="auto"` is a big-model inference placement mechanism, not a substitute for distributed training. It should not be used as evidence that training is safely sharded.
- FSDP or DeepSpeed ZeRO can shard parameters, gradients, and optimizer state. That can reduce model-state memory, but a large attention activation can still remain on the GPU that owns that layer.
- Tensor, sequence/context, or carefully designed pipeline parallelism could split the offending computation, but that requires explicit distributed-training changes and validation.

A 141 GB H200 would probably have enough raw capacity for the observed 78.8 GiB peak. That would only mask the unintended vision-tower training and would spend more money on a less controlled intervention. The corrected language-side LoRA path already has roughly 35 GiB of headroom on one 80 GB H100, so a larger GPU is not currently justified for memory safety. Two GPUs could later reduce wall-clock time only after implementing explicit data or generation parallelism.

## July 13 Overtraining Result

The completed 100-per-group run was technically successful and strongly affected the intended output rate, but the final training schedule was too aggressive for the available pair diversity.

- DPO loss fell below 0.001 by optimizer step 44 and below `1e-6` by step 59.
- Training continued through optimizer step 500.
- Negative terms fell from 373 to 2.
- Response entropy fell 38% and unique responses fell 72%.
- The five most common words accounted for 91.47% of adapter terms.
- The paired 95% interval for race-gap change included zero.

This configuration should not be repeated unchanged. The next run should shorten training to roughly 20-40 optimizer updates, reduce the learning rate, save checkpoints every 10 updates, and stop when preference loss or held-out diversity indicates saturation.

## Reporting Language

Use the following wording in a research update:

> An initial Gemma 3 12B DPO smoke test used global eager attention and generic LoRA projection targets. The targets unintentionally included SigLIP vision-tower projections. Because DPO retained trainable chosen and rejected vision graphs before backward, H100 memory reached approximately 78.8 GiB and a 512 MiB allocation failed. Excluding and freezing the vision tower while retaining eager attention reduced the measured peak to 42.59 GiB allocated and 43.96 GiB reserved. Therefore, the failure should not be attributed to eager attention alone.

## Checks Before Future Runs

1. Confirm the manifest records `exclude_modules` as `.*vision_tower.*`.
2. Confirm the trainable-parameter assertion reports no `vision_tower` names.
3. Run a two-microstep smoke after dependency, model-revision, image-size, or attention-backend changes.
4. Record both `max_memory_allocated` and `max_memory_reserved`.
5. Generate and review the pair CSV before the research training stage.
6. Do not interpret DPO loss, an adapter activation check, or the tiny smoke evaluation as debiasing evidence.

## Technical References

- [Hugging Face attention backends](https://huggingface.co/docs/transformers/attention_interface)
- [Transformers 4.50 Gemma 3 eager-training warning](https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/gemma3/modeling_gemma3.py)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [PyTorch 2.0 accelerated Transformer backends (2023)](https://pytorch.org/blog/accelerated-pytorch-2/)
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [FlashAttention (2022)](https://arxiv.org/abs/2205.14135)
- [Hugging Face big-model inference and device maps](https://huggingface.co/docs/accelerate/v0.29.0/en/concept_guides/big_model_inference)
- [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
