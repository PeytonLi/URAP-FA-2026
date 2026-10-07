# URAP-FA-2026 working instructions

- All further changes and commits belong to the private repository
  `https://github.com/PeytonLi/URAP-FA-2026` (`origin`).
- `upstream` is `https://github.com/eliaswuberkeley/URAP.git`, a read-only research
  reference; its push URL is disabled. Never push there. Never mirror or push the old `peyton`/`main`
  history: an original notebook contains a hard-coded credential.
- Work on the active `codex/fa-2026` branch unless the user requests another.
  Use Conventional Commits. Preserve historical research results and attribution.
- New training and evaluations must use **Gemma 3 12B-IT only**
  (`google/gemma-3-12b-it`), with no comparison models. Existing Gemma 4/Qwen
  runs and scripts are archived evidence, not authorization to rerun them.
- The intended next image pool is
  `C:/Users/lipey/Downloads/FairFace 1400/`. Audit its manifest and image integrity
  before activating it. Preserve declared train/development/test splits; keep the
  12-image benchmark out of training. Do not treat an image-only folder as reviewed
  three-word targets or treat proposed emotion labels as ground truth.
- Retain mandatory private Hugging Face backups, one repo per fine-tuning run,
  with local checkpoint saving first and verified upload checksums.
- Never commit credentials, raw face images, model/optimizer weights, downloaded
  archives, runtime environments or cloud authentication files. Private GitHub
  visibility does not make these suitable Git artifacts. Keep them in ignored
  controlled storage. Do not paste tokens into chat or command arguments.
- Always prefix shell commands with `rtk`; use `rtk proxy` for commands without
  a filter or PowerShell cmdlets (via `powershell.exe -NoProfile -Command`).
- For library, SDK, API, CLI or cloud-service documentation, use Context7:
  resolve the library ID first, then query current documentation. Do not use it
  for general programming concepts or business-logic diagnosis.
