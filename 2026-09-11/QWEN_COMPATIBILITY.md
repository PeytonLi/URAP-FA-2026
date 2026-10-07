# Qwen2.5-VL-3B Compatibility

Model: `Qwen/Qwen2.5-VL-3B-Instruct`; frozen revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.

Saved probe result: `{"attention": "sdpa", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3", "rows": 12, "status": "complete", "training_performed": false, "valid": 7}`

The probe uses the same 12 face bytes and 61-word request, but Qwen's own image/chat template, SDPA, BF16, and a 64-token sampled output budget. It is a format/inference compatibility test, not a full bias estimate. The saved probe has 7 parser-valid answers among 12 responses (status: complete). By recorded image group: Black 2/6 valid, White 5/6 valid. Parser-invalid visible answers are retained in `results/local/qwen_probe.csv`; the current failures use out-of-list positive synonyms. No retry or parsing fallback was used for this probe. A small, selectively valid subset cannot establish that the model is unbiased or that DPO will improve it; format compliance needs work before a full IAT comparison.

The [official model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) documents image-and-text inference. [TRL's DPOTrainer documentation](https://huggingface.co/docs/trl/dpo_trainer#training-vision-language-models) includes this model in its vision-language DPO example and supports PEFT adapters. Thus multimodal DPO/LoRA is an available training route; no Qwen training or gradient test was performed here. Future training needs image-grounded chosen/rejected pairs, correctly masked visual/prompt tokens, a frozen reference, and held-out evaluation.

A 3B model generally needs less weight/optimizer memory than Gemma 12B, but actual dollar savings depend on GPU rental and throughput. An H100 still costs the same per hour when running the smaller model. The probe download used container-local storage, not replacement of the existing network-volume models.
