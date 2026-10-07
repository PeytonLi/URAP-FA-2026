# August 10 Research Log

## 2026-08-10 - First pilot LoRAs evaluated on neutral benchmark

The five final adapters from the first August 10 negative-expression pilot
were retrospectively evaluated on the original 12 neutral IAT images. The
frozen base CSV hash was verified and base inference was not rerun. Every
adapter used the pinned Gemma revision, eager attention, seed `7192026`, ten
repetitions per image, and the historical sampling protocol. All five produced
120 valid responses with zero identity leakage.

| Condition | Black negative | White negative | Absolute gap | Entropy retention | Unique retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sum `2e-6` | 41.67% | 3.33% | 38.33% | 101.8% | 100.0% |
| Sum `5e-6` | 43.33% | 1.67% | 41.67% | 98.3% | 95.3% |
| Sum `1e-5` | 45.00% | 0.00% | 45.00% | 95.4% | 87.5% |
| Mean/token `5e-6` | 38.33% | 3.33% | 35.00% | 92.6% | 84.4% |
| Mean/token `2e-5` | 16.67% | 0.00% | 16.67% | 91.1% | 81.2% |

The frozen base was 35.00% Black, 3.33% White, and a 31.67-point absolute gap.
Mean-token `2e-5` was the only directional adapter: its Black rate fell 18.33
points and its gap fell 15 points. A 100,000-sample paired image bootstrap gave
a Black change interval of [-36.67, -3.33] points and a gap-change interval of
[-33.33, 0.00] points. Black `Pain`, `Sadness`, `Grief`, and `Hurtful` counts
all fell, although one new `Poison` selection appeared.

This adapter is not a research candidate. Entropy retention was 91.1% and
unique-response retention was 81.2% (64 to 52 unique responses), below the
conservative 95% reporting rule. The legacy evaluator's 75% entropy guardrail
passes, but that historical threshold does not establish base-distribution
preservation. Four of five adapters moved in the wrong direction, demonstrating
strong objective and learning-rate sensitivity.

The comparison is explicitly retrospective: the benchmark was already known,
five adapters were compared, and the independent visual sample is six images
per race. Complete plots, response CSVs, bootstraps, manifests, and checksums
are under `results/august10_first_run_neutral12_retrospective_v1/`.

## 2026-08-10 - DPO/LoRA mechanistic diagnostic completed

The prior pilot's exact generation ties left two competing explanations: the
training objective might be unable to alter the sampled metric, or the pilot
might not contain enough aligned signal. A fail-closed overfit diagnostic was
run to distinguish those explanations without opening the protected 419-image
evaluation panel.

The first balanced 12-image neutral panel was preserved as a null diagnostic:
the pinned base generated zero negative terms for both races, so reduction was
mathematically impossible. A separate 80-image screen of the prior training
split found 15 base negative terms for Black images and 15 for White images.
Six expression-matched Black/White blocks containing those events became an
explicitly non-confirmatory exact-image canary. It contains 204 rows: 192 train
and 12 diagnostic rows. Template distributions are identical by race, chosen
responses contain zero negative words, and rejected responses contain three.

A fresh rank-8 LoRA was trained from immutable Gemma revision
`96b6f1eccf38110c56df3a15bffe176da04bfd80` with mean-token DPO, learning rate
`5e-5`, beta `0.1`, and SDPA. No previous adapter was loaded. Training completed
48 optimizer updates with a final DPO loss of about `0.000981`, 100% development
preference accuracy for both races, and peak reserved GPU memory of 25.199 GiB.

| Condition | Black negative rate | White negative rate | Entropy | Unique |
| --- | ---: | ---: | ---: | ---: |
| Pinned base | 16.67% | 16.67% | 4.158 | 64.17% |
| Update 8 | 0.00% | 0.00% | 3.747 | 55.00% |
| Update 16 | 0.00% | 0.00% | 1.465 | 10.00% |

Update 8 proves that DPO plus LoRA can causally lower this generated-word
metric. It retained 90.1% of entropy and 85.7% of unique-response rate. Update
16 added no primary benefit and failed the diversity safeguard, so later
checkpoints were not generation-evaluated.

This is not a fairness result. Base negative rates were 27.5% on
negative-expression images, 22.5% on neutral images, and 0% on positive images;
update 8 drove every expression class to 0%. The all-positive construction is
therefore too blunt and removes grounded expression sensitivity. The next
dataset must use expression-conditioned preferences, retain calibrated terms
such as sadness when visually supported, reject unrelated or extreme terms,
and evaluate on different images with fixed base responses.

Results and plots are isolated under
`results/august10_dpo_lora_overfit_diagnostic_v2/`. The exact training and
evaluation outputs are separately preserved under
`data/august10/diagnostic_overfit_v2_informative/` and
`outputs/august10/diagnostic_overfit_v2_informative/`. No Claude or ChatGPT API
calls were used.

## 2026-08-10 - Negative-expression pilot completed

The prespecified FairFace expansion added 18 matched images with negative
expressions. Three pinned-base Gemma profiles and three independent GPT teacher
profiles were available per image, producing a 162-comparison profile pool.
Claude recovery audits used four blinded presentations and supplied enough
compatible rows to freeze 100 examples.

The negative-expression expansion used 219 Claude calls: 57 Opus 5 and 162
Sonnet 5. Returned usage was 549,355 input and 141,080 output tokens with an
estimated cost of $3.367927, below the authorized $10 session cap. Cumulative
August 10 auditing reached 955 calls and an estimated $14.639742. No credential
was written to the repository.

Frozen v4 contains 100 unique FairFace images, 50 rows per race, 80 train and
20 development rows, 30 IAT and 70 profile rows, and seven negative-expression
blocks. The historical v1 dataset gate passed. The pinned base development
challenge also passed its informativeness gate with 12 Black and 15 White
negative terms, 100% validity, zero identity leakage, and complete profile
coverage.

Five fresh rank-8 DPO LoRAs were trained from the same immutable Gemma revision:

| Reduction | Learning rate | Dev preference gate | Generated result |
| --- | ---: | --- | --- |
| Sum | `2e-6` | Pass | Exact primary-metric tie |
| Sum | `5e-6` | Fail | Exact primary-metric tie |
| Sum | `1e-5` | Fail | Tie plus one identity-leakage response |
| Mean/token | `5e-6` | Pass | Exact primary-metric tie |
| Mean/token | `2e-5` | Pass | Exact primary-metric tie |

Eight checkpoints were screened. Black stayed at 26.67%, White at 33.33%,
overall at 30.00%, and the absolute gap at 6.67 points. The corrected v2
checkpoint gate rejects exact ties, so no seed replication or confirmatory
evaluation was opened.

The summed objective produced very large profile margins while IAT margins
remained near zero. Mean-token scoring removed most of that length-related
imbalance and is now an explicit, recorded trainer option, but it still did not
move generation rates.

Post-run split inspection identified the decisive data-coverage flaw: all three
negative-expression IAT blocks were assigned to development, leaving zero in
train. Dataset gate v2 now requires at least three training negative-IAT blocks
and intentionally rejects frozen v4. Future work must expand this category
before another GPU run rather than continue the learning-rate sweep.

The complete public bundle is under
`results/august10_negative_expression_pilot_v4/`; five final adapters are
preserved locally under ignored `outputs/august10/` paths and are identified by
SHA-256 in `adapter_inventory.json`.

## 2026-08-10 - Iteration opened

The August 3 six-row adapter is retained as a negative directional result and
will not be continued or used as a warm start.

The prior evidence was consolidated into a fail-closed protocol. Retrospective
analysis found that the August 3 exact matcher retained five blocks from 26
Claude-eligible rows, while original-pair structural matching retains ten. The
same six-row training CSV fails the new dataset gate, and its generated
development summary fails because it has zero base negative events and no
profile coverage.

No August 10 model training has started. The next authorized work is selector
correction, additional candidate judging, and construction of a 100-row frozen
pilot with an informative held-out development challenge.

## 2026-08-09 - Claude audit expanded

The 480-row August 3 Gemma/GPT candidate pool was mechanically audited before
additional provider calls. One row failed deterministic prefiltering; 479 were
mechanically valid. Existing Claude judgments were reused by `pair_id` and
provider checkpoint rather than repeated.

IAT candidates were judged with Claude Opus 5. Profile candidates were judged
with Claude Sonnet 5. Each accepted candidate passed four presentations:
image normal, image reversed, text-only normal, and text-only reversed. Race
metadata was not included in the judge prompt. Calls were staged so an
irrecoverable early failure did not incur the remaining presentations.

Across August 10 cycles 1 through 13:

| Model | Calls | Input tokens | Output tokens | Estimated cost |
| --- | ---: | ---: | ---: | ---: |
| Claude Opus 5 | 203 | 469,553 | 109,068 | $5.074465 |
| Claude Sonnet 5 | 533 | 1,368,645 | 346,006 | $6.197350 |
| **Total** | **736** | **1,838,198** | **455,074** | **$11.271815** |

The provider API returned token usage but not the Anthropic Console balance.
Therefore exact spend is tracked, while remaining credits and the 10% reserve
are explicitly marked unknown. No API key is stored in this repository.

The audit now contains 163 unique eligible rows when the 26 prior eligible rows
are included. Original `sampling_match_id` selection reaches 15 IAT blocks and
20 profile blocks. Additional profile judging had a low complete-block yield
because eligible Black and White rows often received different objective
labels despite independently passing.

## 2026-08-09 - Profile matching amendment

Paid calls were paused and the already eligible rows were evaluated under a
prespecified covariate-rematching rule. The final deterministic matcher keeps
the same objective, gender, expression class, and fine expression; permits at
most one adjacent age bucket and one expression-intensity level; requires at
least 0.70 expression confidence; and retains the prior word-count, grounding,
and quality compatibility limits. Source match IDs remain in the frozen rows.

This produces 35 profile blocks with 70 unique images:

- 30 direct mechanical harm-reduction blocks;
- 5 grounding-improvement blocks;
- 25 exact-age and 10 adjacent-age matches;
- 33 exact-intensity and 2 adjacent-intensity matches;
- 11 male/neutral, 9 male/positive, 9 female/positive, and 6 female/neutral
  blocks.

Together with 15 IAT blocks, the freeze has 100 rows, 100 unique images, 50
Black and 50 White rows, 80 train and 20 development rows, and the required
12/28 train plus 3/7 development task split.

## 2026-08-09 - Dataset decision

The candidate freeze passes every dataset check except negative-expression
coverage. It has zero complete negative-expression blocks; the gate requires
four. The entire 480-row source pool contains only two negative-expression
matched image blocks, making the requirement impossible even before Claude
acceptance losses.

Decision: `do_not_train`. Preserve the candidate CSV for inspection, expand the
prespecified FairFace negative-expression pool, generate fresh Gemma/GPT
candidates, run the same Claude protocol, and freeze again. The next pool
should add at least eight negative-expression blocks so four can survive
candidate and judge rejection. After the dataset gate passes, freeze and run
the base development challenge before starting DPO/LoRA.
