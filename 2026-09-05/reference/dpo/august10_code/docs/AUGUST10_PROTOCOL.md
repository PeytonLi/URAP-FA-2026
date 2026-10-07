# August 10 Directional Pilot Protocol

## Objective

Train a fresh DPO LoRA on pinned Gemma 3 12B IT so unsupported negative-term
selection decreases without forcing positive descriptions, increasing another
group's negativity, leaking identity labels, or collapsing response diversity.

## Fixed boundary

- Model revision: `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- Fresh LoRA for every candidate; never continue a failed adapter.
- SDPA attention on one H100 80 GB. Eager attention is excluded after an OOM.
- Train/development images are disjoint from 419 frozen evaluation hashes.
- Base and adapter use identical prompts, images, generation seeds, and decoding.
- The 400-image FairFace, 12-face neutral, and confirmatory profile panels stay
  closed until a configuration replicates on development.

## Pair construction

Every retained row must pass mechanical policy and a race-blind Claude audit.
Claude receives image normal/reversed and text-only normal/reversed
presentations. The pair is eligible only when order reversal does not reverse
the judgment and image/text judgments satisfy the task policy.

IAT blocks require the same chosen and rejected three-word tuples for the
matched Black and White images. Chosen must contain fewer negative terms than
rejected; positive-versus-positive pairs are forbidden.

Profile blocks require matched task, objective, gender, expression class and
fine expression, compatible age/intensity, confidence at least 0.70, compatible
length/grounding/quality, unique images, and unique chosen profiles. Chosen
profiles must contain zero measured identity leakage or unsupported content.

## Dataset gate v2

The target remains 50 complete blocks and 100 rows:

| Split/task | Blocks | Rows |
| --- | ---: | ---: |
| Train IAT | 12 | 24 |
| Train profiles | 28 | 56 |
| Development IAT | 3 | 6 |
| Development profiles | 7 | 14 |
| **Total** | **50** | **100** |

Additional requirements are machine-readable in `config/pilot_gate.json`.
The post-run v2 amendment adds a minimum of three **training**
negative-expression IAT blocks. This prevents all measurable target examples
from being reserved for development, as happened in the historical v1 freeze.

The historical v1 CSV passed before training. It now fails v2 only on
`train_negative_iat_blocks` and must not authorize another run.

## Development challenge

The base challenge must contain at least six negative terms and ten profile
responses per race. The August 10 base challenge passed with:

- Black: 12/45 negative terms (26.67%);
- White: 15/45 negative terms (33.33%);
- overall: 27/90 (30.00%);
- signed Black-minus-White gap: -6.67 points;
- validity: 100%; identity leakage: 0%;
- 14 profile responses per race with 100% format, subject reference, and
  uniqueness.

This intentionally difficult negative-expression panel is development-only.
It is not interchangeable with the neutral 12-face benchmark.

## Training and promotion

Initial summed DPO used rank 8, alpha 16, dropout 0.05, beta 0.05, one pass,
80 gradient steps, accumulation 8, and ten optimizer updates. Learning rates
`2e-6`, `5e-6`, and `1e-5` were screened. Mean-token DPO was then added as an
explicit exploratory objective to reduce the response-length imbalance between
three-word IAT answers and 80-140 word profiles; `5e-6` and `2e-5` were screened.

A checkpoint is promotable only when:

- Black, overall, and absolute-gap rates do not increase;
- at least one primary rate strictly improves;
- validity and identity leakage do not regress;
- IAT entropy/uniqueness and profile compliance/diversity meet retention gates;
- both task/race groups have adequate held-out preference behavior.

Eight evaluated checkpoints produced zero primary-rate movement. One also
increased identity leakage. No checkpoint was promoted.

## Required next iteration

1. Add at least six new negative-expression IAT blocks so three can enter train
   and three can remain in development.
2. Keep the dataset race-balanced and image-disjoint; do not duplicate rows or
   increase White negativity to manufacture equality.
3. Prespecify mean-token scoring plus either task-balanced weights or a matched
   accumulation schedule. Change one objective factor at a time.
4. Require strict development movement before three-seed replication.
5. Open confirmatory panels once only after data, configuration, seed rule, and
   adapter hash are frozen.
