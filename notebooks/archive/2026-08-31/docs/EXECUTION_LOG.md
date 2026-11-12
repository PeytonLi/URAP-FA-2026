# August 31 Study Execution Log

This log distinguishes compatibility failures from final analysis. Detailed failed-probe
artifacts remain in the full local workspace and are intentionally omitted from this
compact GitHub package; they were never pooled into final estimates.

## Preparation

1. Created a standalone August 31 workspace from the partial August 24 branch without
   modifying the historical August 17/24/26 folders.
2. Rebuilt the Python 3.11 environment, passed lint/unit tests/dry runs, and copied the
   minimal row-level August 24 IAT, DecodingTrust, and BOLD inputs now published as
   `data/raw/archived_*`.
3. Recomputed the archived significance CSVs/figures. No BH-adjusted q was below 0.05.
4. Pinned all eight OpenRouter aliases and exact ZDR routes. Provider fallback remained
   disabled for every face-image call.

## UTKFace freeze

1. Downloaded the pinned UTKFace mirror archive and verified SHA-256.
2. Annotated 11,483 eligible adult Black/White files with the pinned FER model.
3. An initial pool build found byte-identical source duplicates. Selection was changed to
   deduplicate all eligible files by content hash before matching; a regression test now
   enforces this behavior.
4. Tightened gates to FER confidence >=0.60 and quality >=0.50 after visual audit.
5. Froze 400 evaluation and 400 future-training images, exactly 100 per race/gender cell,
   with no cross-pool hash overlap. Future DPO rows remain prompt-only.

## Route and payload compatibility

- MiniMax's original DeepInfra FP8 route returned repeated 429s; AtlasCloud was not an
  active ZDR route. The study pinned Venice FP8, which passed low-cost probes but later
  required one-request throttling.
- Kimi's original DeepInfra BF16 route failed two-image calls and spent reasoning tokens
  without visible output. The active ZDR DigitalOcean route passed. Setting
  `reasoning_effort=none` reduced its one-image IAT response cost by about tenfold in the
  compatibility probe.
- GPT-5.6 Luna's pinned Azure EU route repeatedly returned `502 no healthy upstream` for
  both one- and two-image calls. No fallback route was used; final GPT cells remain
  missing. A final one-image retry at 10:12 UTC failed identically. The final failure
  status is retained here and in `config/study_manifest.json`; verbose probe output
  remains local.
- VisBias source images varied from small portraits to multi-megapixel files. All selected
  images were normalized to a 512-pixel maximum edge at JPEG quality 90. This reduced a
  Kimi two-image prompt from roughly 16,075 to 686 input tokens and made image-token cost
  comparable across source cells.

## Fail-closed corrections during paid generation

### Seed-range failure

The first staged VisBias run used a stable SHA-256-derived model offset but did not bound
it to a signed seed range. Google Vertex returned systematic HTTP 400 errors. The run was
stopped and excluded from final results. The final seed contract uses the same hash
offset modulo one billion and is tested to stay below `2^31`; final generation started
clean. Verbose failed-run transport artifacts remain local.

### Worker starvation

Model-major submission let Kimi semaphore waiters occupy the global worker pool. Task
submission was changed to round-robin task-major order so every selected model route can
make progress. Staged resumes merge model contracts, task hashes, seed contracts, and
account baselines.

### MiniMax shared-pool throttling

Venice returned sustained upstream-shared-pool 429s with a 30-second retry hint. The
stage stopped, then resumed at one concurrent request, 0.75-second minimum spacing, and
four client retries. Remaining invalid cells are retried once more after cooldown; any
persistent cells are reported rather than rerouted.

### Step reasoning budget

Step 3.7 Flash consumed the 220-token description budget in hidden reasoning and returned
no visible paragraph. A 1,024-token retry also ended at the token limit. The previously
observed 2,048-token Step contract passed a fresh compatibility probe and is used for its
final VisBias cells. It completed 228/252 cells at that stage; 24 still exhausted 2,048
tokens without visible output. One bounded, race-blind retry at 4,096 tokens recovered
22/24. The two persistent length failures are retained, and no further Step resampling is
performed. The effective requested cap is recorded per response.

### UTKFace generation

The seven healthy routes completed all 2,800 planned model/image cells. Seven provider
errors were retried once as a technical class and recovered; eleven two-word/empty
format failures were not selectively resampled. The frozen row-level result therefore
contains 2,789 valid three-word outputs from 2,800 distinct model/image cells. GPT is
absent from this new run because its pinned route never passed the compatibility gate.

### Identity-proxy correction

The first local VisBias scoring pass counted bare `black` and `white` tokens as identity
mentions. A context audit showed that ordinary clothing/background phrases dominated
the result. Before report generation, the parser was narrowed to unambiguous or
person-linked identity phrases and regression-tested against `black suit`/`Black man`
examples. Raw model responses were not changed; the superseded local aggregates were
regenerated and are not used.

### IAT inferential-validity correction

The initial offline UTKFace reanalysis correctly excluded invalid outputs from reported
rates, but allowed partial two-word outputs into the image-cluster bootstrap and
permutation inputs. Before artifact freeze, both inferential paths were changed to use
only `valid_three=true` cells, and regression tests were added. Raw API responses were
not changed; confidence intervals, raw p-values, BH-adjusted q-values, figures, and the
report were regenerated from the corrected analysis.

## Stopping and interpretation rules

- Sampling was fixed before effect directions were inspected. Nulls and missing cells
  remain visible.
- No full 10X expansion was run.
- Calls stop on systematic provider errors, alias/route drift, $12 additional-session
  usage, or a projected balance below the $10 reserve.
- Operational retries change concurrency/backoff only. Model alias, canonical revision,
  provider route, prompt, image hashes, seed, temperature/top-p, and analysis contract do
  not change within a final model cell.
