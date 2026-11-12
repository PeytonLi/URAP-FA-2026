# August 10 DPO/LoRA Mechanistic Diagnostic

**Decision: MECHANISTIC PASS; NOT A RESEARCH OR DEPLOYMENT CANDIDATE.**

The immutable base is `google/gemma-3-12b-it` revision
`96b6f1eccf38110c56df3a15bffe176da04bfd80`. This run trained a fresh rank-8
LoRA only. No prior adapter or base-weight update was used. No Claude or ChatGPT
API calls were used.

## Why this diagnostic was needed

The first 12-image neutral panel produced zero negative terms for either race,
so it could not measure a reduction. A separate screen of 80 prior training
images found 15 negative terms for Black images and 15 for White images. Six
matched blocks containing that signal were frozen into this explicitly
non-confirmatory exact-image canary.

## Main result

| Condition | Black negative rate | White negative rate | Gap | Valid | Entropy | Unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 16.67% | 16.67% | 0.00% | 100.00% | 4.158 | 64.17% |
| Update 8 | 0.00% | 0.00% | 0.00% | 100.00% | 3.747 | 55.00% |
| Update 16 | 0.00% | 0.00% | 0.00% | 100.00% | 1.465 | 10.00% |

Update 8 proves that DPO plus LoRA can causally change the sampled word metric:
both race-group negative rates fell from 16.67% to 0%, the gap stayed at zero,
validity remained 100%, entropy retention was
90.1%, and
unique-response retention was
85.7%.

Update 16 added no metric benefit and collapsed the unique-response rate to
10.0%. Later checkpoints were not opened
for generation after this early-stop failure.

## Accuracy warning

Base negative-term rates were
27.5% on negative-expression
images, 22.5% on neutral images,
and 0.0% on positive images.
Update 8 drove every expression class to 0%. Therefore, the current all-positive
preference construction removes expression sensitivity as well as unjustified
negativity. It must not be described as a successfully debiased adapter.

## Next experiment

Use expression-conditioned preference pairs. Neutral images should prefer
low-negativity, image-grounded outputs. Negative-expression images should retain
calibrated terms such as sadness when visible while rejecting unrelated or
extreme terms such as evil, hate, ugly, or criminality. Evaluate on separate
images, preserve the exact fixed base responses, and promote only checkpoints
that reduce neutral unjustified negativity without erasing expression signal or
response diversity.

The 419-image protected evaluation manifest remained closed and disjoint.
