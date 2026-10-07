# Direct photo review complete; AI approvals recorded, neutral targets unresolved

## October 5 update: user-authorized AI self-approval

The user explicitly requested: “I want you to label every image and just approve it and start the training.” This supersedes the prior independent-human-approval requirement, not the unchanged full-cohort, image-conditioning or spending safeguards. The current implementation reuses the existing validator and records 856 resolved direct proposals as **AI self-approved**, with zero human approvals. Their uncertainty flags remain intact; self-approval is not independent validation or ground truth.

AI decisions are stored separately in `data/private/target_review/ai_review_decisions.json`. Human decisions remain in their existing file and take precedence. Approvals bind the exact image, split, direct-assessment fingerprint, policy, receipt and explicit-user-authorization fingerprint. Changed proposals or revoked authorization invalidate AI approvals without erasing historical evidence. Neither the raw direct assessments nor archived Haiku drafts were changed. The existing page displays AI versus human approval counts and badges; refresh an already-open page after restarting the reviewer.

There are still 1,064 unresolved targets (251 neutral and 813 uncertain). Their words remain blank; the current 61-word vocabulary has no neutral/abstention answer. The user must choose whether to keep that vocabulary with explicitly weak forced associations or change the training-target policy to allow neutral/abstention answers. No such policy change, teacher fallback, cohort reduction or split change has been made. The full approved training file remains absent, actual export is refused, and no optimizer update or checkpoint exists. Backend image-conditioned training and budget/capacity validation also remain prerequisites.

All 81 local study tests pass, including AI authorization/revocation, provenance separation, stale-approval handling, human overrides, unchanged-target acceptance and full-cohort export refusal. The source/logic tests do not establish image-label correctness or browser rendering. Work remains local on `peyton`; nothing was pushed. Earlier snapshots below describe the previous human-review policy and are historical.

## October 4 update: direct photo inspection complete

The user requested Codex's own direct visual inspection as the primary drafting method. The previous 1,920 Haiku proposals below are preserved as historical comparisons, not silently presented as Codex labels. New proposals are recorded separately in `data/private/target_review/codex_direct_reviews.json` after each source photo is actually viewed with `view_image(original)`. No publisher demographics or classifier expression labels are used to choose them. Exact assistant-model revision is not exposed; these remain AI proposals requiring independent human review, not ground truth.

[Current direct-pass counts](../results/direct_review_progress.json) record actual inspection coverage, tentative three-word proposals and unresolved cases. All 1,920 train/development photos have now been directly inspected; this is not a replacement study on a smaller resolved subset. Ambiguous or neutral photos are retained with blank targets for human policy/review decisions rather than automatically receiving positive appearance words. No image or matching block is removed, and test/benchmark photos remain excluded.

The existing review page now shows the drafting source, direct-pass progress and unresolved targets. Images awaiting direct inspection cannot be approved as direct labels. A revised proposal invalidates its previous approval fingerprint rather than silently inheriting human approval. Revision decisions may retain blank answers; training export still requires complete, canonical, independently approved targets. The direct spreadsheet copy is `data/private/target_review/codex_direct_reviews_for_review.csv`; the older `draft_targets_for_review.csv` still contains Haiku proposals only.

Further external-teacher calls are disabled in the existing budget policy. This direct pass creates no additional Prime/Fireworks labeling charges; it still consumes the assistant session's usage. The $100 total Prime cap remains in force for eventual verified training/evaluation. No optimizer update, adapter or trained-model result is claimed by draft review.

Snapshot at October 5, 2026 04:07 UTC (October 4 locally): 72 of 1,920 photos actually inspected; 25 tentative three-word pairs and 47 unresolved targets, with 1,848 photos remaining. Zero human approvals and no approved training file. The full labeling pass and the multi-model SFT/ORPO study are unfinished. All 77 current local tests pass, including direct-source provenance, abstention, stale-approval rejection and malformed-request handling. Browser rendering remains unverified; only source syntax and local data/approval logic are checked.

Completion snapshot at October 5, 2026 06:54 UTC (October 4 locally): all 1,920 train/development photos directly inspected (100%); 856 tentative three-word pairs and 1,064 unresolved targets, with zero photos remaining to inspect. This continuation added 480 actual-photo inspections to the previous 1,440-record snapshot. Every saved direct annotation has a verified source-image hash and a visual justification; none is a human approval or objective emotion label. The refreshed direct review CSV contains all 1,920 inspected records, including blank unresolved answers. The original 1,920 Haiku drafts, published image membership and splits are unchanged. No new external labeling requests or training updates were made during this continuation. Inspection is complete, but zero targets are human-approved and the approved training file still does not exist. The full labeling/approval workflow, actual SFT/ORPO runs, matching evaluations and trained-model comparison reports remain unfinished.

Final local verification passes all 77 study tests, all 19 source-package checksums and the whitespace/diff check. The direct JSON and CSV remain Git-ignored. These checks establish local format/provenance and approval safeguards, not visual-label correctness, interactive browser rendering or a working training backend.

The direct drafts also need construction-quality review: the 856 proposed pairs use 14 distinct ordered preferred triples, 10 preferred words and eight ordered rejected triples. These repeated expression templates could encourage canned responses rather than broad image grounding. Current subjective assessments are 813 positive, 43 negative, 251 neutral and 813 uncertain; these describe the AI drafting pass, not population emotions or benchmark results. Do not train only the resolved subset or use a lower negative-word rate alone as evidence of improvement. The 61-word vocabulary provides no suitable neutral answer, so neutral-policy decisions and independent human review remain necessary; completed inspection is not permission to invent targets, remove photos or change splits.

## Previous Haiku generation

The October 3 user request authorized draft generation and supplied $100 of Prime credits. This is treated as a $100 total Prime spending cap, with a $20 drafting sub-cap. It does not authorize automatic target approval or establish that all six training runs will fit the remaining budget.

## Actual result

- 1,920 image-specific draft pairs: 1,440 training and 480 development, with published membership and splits unchanged.
- Every preferred and rejected answer has exactly three distinct canonical words. No pair differs only by rearranging the same words.
- All 1,920 image hashes, retained teacher responses, prompt assignments and split assignments pass the consistency audit. This verifies format/provenance, not visual correctness.
- Zero human approvals, zero approved exports, zero optimizer updates and zero new checkpoints. The 480 reserved test photos and original 12 benchmark photos were not sent for target generation. No profiles were used.
- 1,954 paid requests, including failures/retries, across exactly 1,920 eligible images. Conservative catalog/reported-usage estimate: $5.210072. Latest account balance: $94.2527, checked October 4 at 04:07:03 UTC. Token estimates are not a final invoice; account balance also reflects prior activity.

[Executed audit](../results/target_draft_audit.json), [draft summary](../results/target_draft_summary.json), [non-secret budget limits](../config/prime_budget.json).

## How to review

Open [the local photo review page](http://127.0.0.1:8879/). It serves only known train/development photos on this computer. If stopped, restart from the repository root:

```powershell
rtk python 2026-10-03/review_targets.py
```

1. Enter your reviewer name and examine the actual photo.
2. Inspect the preferred answer, rejected counterexample, visible-feature explanation and flags. Use the three word pickers for each answer to make corrections without spelling errors. The full vocabulary is available on the page.
3. For neutral/ambiguous faces, ask whether the proposed associations are defensible at all. A grammatical answer is not necessarily a suitable target. Do not infer personality, health, morality or biography from a face.
4. Confirm that you examined the photo and considered its flags, then use “Approve & next” or “Save as unsure & next.” Unsure decisions are not approvals and keep the photo in the study; there is no approve-all action. “Next / skip” saves no decision. The reviewer name and current position are remembered in this browser, while explicit decisions persist in the local review file.
5. Once all 1,920 pairs have explicit approvals, use “Export all approved targets.” This creates a separate reviewed target file; it does not launch training. An existing approved file will not be overwritten.

Decisions persist locally in `data/private/target_review/human_review_decisions.json`; drafts remain drafts in a separate file. Do not bulk-change status to manufacture human approval. If you reject the labeling policy, leave affected targets pending and agree on a revision; do not silently remove images or change splits.

The October 5 photo-first update shows one larger photo at a time beside its words and visual justification. It includes split/status filters, a blank-target queue, photo-ID search, direct position navigation, six canonical-word pickers, notes, explicit save-and-next controls and unsaved-edit warnings. Arrow keys navigate only when focus is outside editing controls; no shortcut approves a photo. A failed or not-yet-loaded photo cannot unlock save buttons. Preferred and rejected answers must use different word sets, not reordered copies.

All 78 study tests pass, including the new isolated frontend logic check for validation, explicit approval, uncertainty retention, navigation, resume and a late old-photo load. The UI checks use fictional records and mocked responses only: no real human decision is manufactured and no actual photo or approval file is modified. This verifies local logic/syntax, not browser layout or real-browser rendering. The visual check remains blocked because the app cannot verify saved browser permissions; no security workaround was used. The user can open or refresh the local page themselves. No training is launched by this UI update.

## Quality concerns to review

These are proposals, not ground truth. Of 1,920 drafts, 1,857 have at least one flag. Overlapping concerns include subjective appearance (1,600), weak grounding (754), teacher/classifier expression disagreement (637), low subjective confidence (451), ambiguity (443), no neutral vocabulary (424), and low image quality (382).

The teacher described 1,171 faces as positive, 340 negative and 409 neutral. These are subjective associations, not ground-truth emotions. Classifier metadata was consulted only after generation to flag disagreements; it was not supplied to the teacher and did not choose or replace words.

Preferred answers contain 4,740 positive and 1,020 negative words: 82.29% positive. Crucially, the teacher-neutral group has 1,224 positive words out of 1,227 selections: 99.76%. Frequent neutral proposals use `Pleasing`, `Appealing`, `Glad` and `Lovely`. This is a positive/appearance prior, not an objective neutral answer. Training these unchanged could reduce negative rates by teaching generalized positivity rather than better image grounding. A smaller race gap alone is not sufficient evidence of success.

Preferred answers use 125 distinct ordered triples and 34 distinct words; rejected answers use 103 triples and 30 words. Review repeated templates, unsupported appearance judgments and expression mismatches. Keep demographic comparisons as audits, not demographic rules for assigning labels.

For example, `fa_022adaf852ffe98560dd` proposes `Glad, Pleasing, Lovely` over `Angry, Horrible, Disgust` despite a teacher-neutral expression and weak-grounding flags. It particularly needs your judgment. `fa_005fcc4fc865e3ef03a5` proposes `Sadness, Pain, Grief` over `Happy, Joyful, Celebrate`, with low-quality/weak-grounding flags; the answer must not be treated as evidence of actual pain or grief. Neither is approved.

These statistics describe target construction, not benchmark results, measured emotions or fairness improvement.

## Teacher and provenance

Drafts used `anthropic/claude-haiku-4.5` through the existing Prime Inference login. This is an external-provider gateway, not a local Gemma model, a Prime GPU training job, or Fireworks credits. Actual source PNGs were sent without publisher demographic metadata, classifier expression labels, identities or profile prompts. Policy `visible_expression_61word_draft_v2` requests tentative visible-expression associations and flags the vocabulary's neutral gap. Health/moral judgments are excluded from preferred drafts.

Policy SHA-256: `d92c60a945446017fc4d26c34c412815628a3fbbb6432e4c2f7b6cd692b919b2`.

Catalog/receipts record the teacher alias, returned model ID, timestamps, policy/schema hashes, usage, settings and answers. An immutable teacher-weight revision was unavailable; exact future teacher-output replication is not promised. Approved target bytes can later be frozen independently.

The unstructured pilot produced no usable targets because of non-vocabulary words and extra prose. A constrained JSON schema resolved that issue. The first full pass retained 1,909 drafts; eleven remaining outputs had only two chosen words. Bounded image-bearing correction requests demanded three distinct entries and filled those gaps. Valid drafts were not overwritten and missing words were not mechanically invented. Invalid responses remain in receipts and the cost ledger. The correction hint initially described the issue as repetition, but explicitly demanded three distinct entries; actual retained results were checked against receipts. The current script uses accurate generic format-failure wording.

Private artifacts remain Git-ignored:

- `data/private/target_review/draft_targets.json`: all proposals and per-image provenance.
- `data/private/target_review/draft_targets_for_review.csv`: spreadsheet-friendly review copy. Editing it alone does not approve/export targets.
- `data/private/target_review/receipts/`: safe raw responses and usage metadata, including failures; no credentials or image payloads.
- `data/private/target_review/cost_ledger.json`: all settled requests; zero unknown charge reservations remain.
- `data/private/target_review/wallet_checks/`: immutable sanitized checks for later invocations. Earlier convenience wallet files were overwritten between pilot invocations, so they are not a complete historical billing ledger.

The approved file, `data/private/curated_iat_targets.json`, does not exist yet. Original CSV SHA-256 remains `7b08c31122f624e733ac279b87882c9138a05ea4c9dc7a16596105a5d4637d13`.

## After approval

SFT uses each training image, the unchanged IAT prompt and its approved preferred answer. ORPO uses that same image/prompt with preferred and rejected answers. Development records support checks/selection, never optimizer updates. Test and original benchmark photos remain outside target construction.

Target approval is necessary but not sufficient: each model needs a supported image-conditioned training route and a bounded GPU smoke test. Hosted Prime eligibility and the older Fireworks Gemma alignment issue remain unresolved. A changed serving stack also needs its matching untouched-base benchmark. Quote capacity/pricing, then stage training/evaluation within the remaining total authorization; stop before exceeding it. No idle GPU was provisioned while review is pending.

All work remains local on `peyton`; nothing has been pushed.
