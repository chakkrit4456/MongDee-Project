# Third-Party Model / License Audit

Produced during the multi-camera AI accuracy upgrade (2026-09-19). Covers
the candidate models named in the upgrade brief. **No code from any of
these repos has been cloned or integrated** — this document records the
research/license-audit step only. Per the upgrade brief's own rule
("if licensing is incompatible, do not integrate"; "a candidate model is
accepted only if it provides measurable improvement"), nothing here is
wired into MongDee until a specific, evidenced need justifies it.

Sources: GitHub's own license metadata (`api.github.com/repos/...`) and
each project's README/LICENSE file, fetched directly — not asserted from
memory. Flagged where GitHub's auto-detection returned `null` (does not
mean "unlicensed", usually means a non-standard or split license).

| Project | Repo license (code) | Pretrained weights license | Commercial use for MongDee? | Notes |
|---|---|---|---|---|
| **FairFace** (dataset) | GitHub metadata: `null` for `dchen236/FairFace` repo code | **Dataset: CC BY 4.0** (from fairface.ai) | **OK** — CC BY permits commercial use with attribution | MongDee already trains its own model on this dataset (see `FAIRFACE_TRAINING_REPORT.md`) rather than using a third-party checkpoint — this is the *lowest-risk* path of everything evaluated, already in production, and this audit confirms it was the right call license-wise. |
| **ByteTrack** (FoundationVision) | **MIT** | N/A — a tracking algorithm, not a pretrained weight distribution; runs on top of any detector's output | OK if used | **Superseded by a better option**: `ultralytics` (already a MongDee dependency, v8.4.153) ships its own maintained ByteTrack implementation (`ultralytics/cfg/trackers/bytetrack.yaml`) — same algorithm family, zero new dependencies, no separate license to track, no Python-3.14-compatibility risk. Cloning FoundationVision's original repo would add risk (unmaintained-relative-to-ultralytics, its own PyTorch/YOLOX version pins) for no corresponding benefit. **Recommendation: do not clone; use `ultralytics`'s built-in tracker if/when a tracker upgrade is justified by evidence.** |
| **BoT-SORT** (NirAharon) | **MIT** | N/A (same reasoning as ByteTrack) | OK if used | Same conclusion as ByteTrack: `ultralytics` ships `botsort.yaml` natively. **Recommendation: do not clone.** |
| **FastReID** (JDAI-CV) | **Apache-2.0** | **Mixed — dataset-dependent.** Model Zoo (`MODEL_ZOO.md`) offers checkpoints trained on Market-1501, **DukeMTMC-reID**, MSMT17, and vehicle Re-ID sets. | **Conditional — avoid the DukeMTMC-trained checkpoints specifically.** | DukeMTMC(-reID) was withdrawn by Duke University in 2019 after it came to light the underlying video was collected without subject consent; continued use/redistribution of models trained on it is widely treated as an ethical and legal liability in the CV community, independent of FastReID's own Apache-2.0 code license. If FastReID is ever integrated, only Market-1501 or MSMT17-trained checkpoints should be used, never the DukeMTMC ones — and this should be re-confirmed at integration time, not assumed from this audit. Not cloned this session (no evidence yet that MongDee's existing from-scratch MobileNetV3-Small Re-ID embedder, `core/reid.py`, is insufficient — see Architecture Audit). |
| **InsightFace** (deepinsight) | **MIT** (code) | **BLOCKED for MongDee — "available for non-commercial research purposes only."** Exact quote from the project's own README: *"The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only."* Applies to buffalo/antelope recognition packs and the face-swap models alike; some packs (e.g. `buffalo_l`) require contacting InsightFace directly for a commercial license. | **BLOCKED** | MongDee is a commercial retail/event-booth analytics product, not non-commercial research — this rules out InsightFace's pretrained SCRFD face detector and recognition models entirely unless a separate commercial license is obtained directly from InsightFace. The code itself (MIT) has wheels for Python 3.14 (`insightface==2.0`, universal wheel) and its `onnxruntime`/`onnxruntime-gpu` dependency also has Python-3.14 wheels (1.30.0) — so this is a **license blocker, not a technical one**. **Recommendation: do not integrate unless MongDee obtains a commercial license from InsightFace directly; this is a business decision, not an engineering one.** |
| **MiVOLO** (WildChlamydia) | **Apache-2.0** (root `LICENSE` file, fetched and confirmed directly) | Best available evidence: no separate/different license found for the pretrained checkpoints — the README's "see license" link resolves to the same Apache-2.0 file. No explicit statement either way beyond that (less airtight than InsightFace's explicit non-commercial notice, but nothing found to the contrary). | **Provisionally OK**, re-verify before integrating | **Not a PyPI package** (`pip index` shows no published `mivolo` release) — would require cloning + running from source, unlike the others. Its accuracy claims (face+body age/gender) are unverified on MongDee's actual camera conditions per this brief's own rule against trusting README benchmarks. Not cloned this session: MongDee's current FairFace-based gender/age (95.18% measured gender accuracy on FairFace's own validation set, see `FAIRFACE_TRAINING_REPORT.md`) already covers the face-visible case; MiVOLO's specific value-add (body-based age/gender when no face is visible, for distant customers) is real per the architecture, but no real long-range/body-only failure data has been collected yet to justify the integration effort — see Architecture Audit's gap list. |

## Compatibility notes (Python 3.14, this machine's only installed interpreter)

Checked directly against PyPI's published wheel list (not assumed):

| Package | Python 3.14 (win_amd64) wheel available? |
|---|---|
| `insightface` 2.0 | Yes (universal `py3-none-any`) |
| `onnxruntime` 1.30.0 | Yes (`cp314-cp314-win_amd64`) |
| `onnxruntime-gpu` 1.30.0 | Yes (`cp314-cp314-win_amd64`) — GTX 1050 (Pascal, compute capability 6.1) CUDA-kernel compatibility with this specific wheel **not yet empirically tested**, same category of risk `torch` had (see main report: `torch` needed the specific `cu126` build tag for Pascal; a generic "wheel exists" check does not confirm Pascal kernels are included) |
| `timm` 1.0.29 (MiVOLO's model-loading dependency) | Yes (universal `py3-none-any`) |
| `opencv-python` 5.0.0.93 | Yes (`cp37-abi3` stable-ABI wheel — already installed and working in the main `.venv`, confirmed) |
| `mivolo`, `fast-reid` | **Not on PyPI at all** — both require cloning from source; no wheel-level compatibility signal available, would need an isolated venv + a real install attempt to know if their pinned dependencies (setup.py/requirements.txt) resolve on Python 3.14 |

## Bottom line

Of the six named candidates, only **FairFace** (already in production) is
both license-clear and already integrated. For the other five: two
(ByteTrack, BoT-SORT) are superseded by functionality MongDee's existing
`ultralytics` dependency already ships; one (InsightFace) is licensing-
blocked outright for commercial use; two (FastReID, MiVOLO) are
conditionally viable but unintegrated pending real evidence that current
in-house components (`core/reid.py`'s Re-ID embedder, FairFace's face-only
age/gender) have a measurable gap severe enough to justify the added
dependency/compatibility risk on Python 3.14 and the GTX 1050's 3GB budget.
