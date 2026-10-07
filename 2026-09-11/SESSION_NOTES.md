# Session Notes

Times are UTC on September 11, 2026. These are execution notes, not additional
scientific comparisons or a new DPO training run.

- Initial access check: OpenRouter balance $26.867815294; Runpod balance about
  $15.94. Approved ceilings: $20 additional OpenRouter spend and $10 GPU spend.
- Input audit: 412 exact image hashes and the selected adapter hash verified.
  Reuse plan: 840 Neutral12 archived slots and 3,200 UTKFace archived slots.
  Eight archived MiniMax Neutral12 slots were HTTP failures, not answers.
- 08:10: compatibility probes began. MiniMax DeepInfra returned repeated 429s.
  Partial probe attempts and technical retries are retained in the API ledger.
- 08:13: deployed one H100 80 GB at $3.49/hour on the existing network volume.
  Historical pods were not restarted or modified. A hard-cost watchdog was started.
- Container setup: rsync was missing and apt update timed out. Transfer switched
  to a gzip/tar archive. Network-volume ownership restoration was not permitted;
  subsequent transfers use `--no-same-owner`. Input hashes verify copied contents.
- 08:20: Qwen2.5-VL-3B probe completed, 7/12 parser-valid answers. The five other
  answers used out-of-list positive synonyms. No training or bias-free claim.
- Local Gemma checks passed: pinned snapshot hashes; pure-base equals
  disabled-adapter seeded probe; no change to generation configuration. The
  worker performs paired inference with every model parameter frozen.
- 08:26: a MiMo transport timeout stopped the first hosted expansion after 6,899
  newly processed slots in that invocation. An unknown-call budget reservation
  and account reconciliation protect the spend cap during technical resumption.
- MiniMax DeepInfra and Venice subsequently became unavailable with endpoint
  catalog status -2. Both cells are excluded from further paid calls while
  unavailable. No fallback provider is substituted or mixed into their data.
- 08:31: the other seven hosted models resumed from their completed slots.
- Around 08:49: both original MiniMax endpoints again reported status 0. The
  coordinator paused at request boundaries to run exact-endpoint recovery probes.
  Technical failures can be filled, but completed/invalid model answers cannot
  be overwritten. API stages returned before interruption are recovered from
  the ledger instead of resampling them.
- 08:54: MiniMax Neutral12's third probe still failed; eleven 429 call attempts
  across three probes triggered a session-long exclusion. MiniMax UTKFace
  resumed on Venice after its fixed compatibility probe passed.
- 08:57: Kimi Neutral12's original DeepInfra route reported status -2 and was
  paused, while its separate UTKFace DigitalOcean route continued successfully.
- Analysis checks: unit tests cover seed pairing, image/adapter hashes, legacy
  parsing behavior, exact/null tests, BH correction, image clustering, and the
  pre-call spending cap, technical-only replacement, and HTTP read deadlines.
  The suite includes a separate offline audit of archived model and generation
  contracts and safe transfer checks.
- 09:22: the remote worker exited successfully with all 5,200 Gemma responses.
  All base snapshot hashes were unchanged after inference.
- Download verification initially hit TypeError: the local Python 3.11.0 does
  not support tarfile.extractall's newer `filter` keyword. The downloader now
  validates every member as an in-scope regular file/directory before extraction.
  Tests cover successful download, traversal rejection and symlink rejection.
- 09:23: all six output files matched remote SHA-256 checksums. The own GPU pod
  was terminated and the existing network volume verified present. Estimated
  compute cost was $4.0813; the account balance was $11.9438 at cleanup.
- 09:29: all hosted UTKFace panels except Grok had reached 2,000 slots. The
  coordinator paused at completed request boundaries, then reassigned six
  concurrent lanes to Grok's remaining UTKFace tasks. This matches the route's
  already-used combined-panel cap; prompts, seeds, route and sample targets did
  not change. The prior invocation exit is retained separately.
- 09:47: hosted generation finished. All eight hosted UTKFace conditions have
  2,000 answers; six hosted Neutral12 conditions have 600. Kimi Neutral12 remains
  at 256 and MiniMax Neutral12 at 112 because their pinned endpoints failed or
  were unavailable. MiniMax Venice became unavailable again after its completed
  UTKFace sample; this does not erase its completed responses.
- Final hosted ledger: 17,028 call attempts, 17,016 successful returned calls,
  eleven HTTP 429 errors and one unknown transport outcome. Reported call cost
  is $12.128267; account reconciliation and its timestamp are preserved separately.
- Analysis includes 25,168 model answers plus twelve retained technical-failure
  slots across ten model conditions and two image panels. The
  separate Qwen compatibility probe has twelve answers. There are 29 figures.
- Nineteen local tests pass. The public release excludes private image bytes,
  adapter/base weights, credentials, runtime state and virtual environments.

`results/hosted/` is authoritative for requests, costs, errors and completion.
`results/runpod/` is authoritative for deployment, retrieval and termination.
`RESULTS.md` is regenerated after inference finishes and reports any incomplete
cells. Intermediate locally generated plots are overwritten by the final replay;
only the final checked release is intended for researchers.
