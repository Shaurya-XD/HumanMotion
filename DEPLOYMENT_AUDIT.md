# Free deployment audit — 2026-10-16

**Local release preparation: PASS. Actual public deployment: NOT VERIFIED.**

No app, checkpoint, repository or account was published. No public URL exists from this work. Public model permissions/downloads and actual free-host capacity require operator acceptance after publishing. Local performance does not establish hosted reliability.

## Requirement checklist

| Requirement | Status | Evidence / scope |
|---|---|---|
| Preserve minimal interface and default MViT Standard mode | PASS | No CSS, theme, main interface or inference changes; AppTest and browser startup verified |
| Selected real checkpoint loads on CPU | PASS | Strict real-weight tests; CPU Torch 2.7.1+cpu / TorchVision 0.22.1+cpu; CUDA unavailable |
| Correct five labels and logits | PASS | walk=0, pushup=1, sit=2, drink=3, clap=4; head has five outputs; finite normalized scores |
| Exact trained preprocessing | PASS | 16 RGB frames, centered two-second window, float conversion before resize, short side 256, crop 224, mean .45/std .225; notebook-equivalence tests |
| Standard and Enhanced | PASS | Actual deterministic Standard forwards and exact Enhanced mean-softmax tests |
| Efficient inference/cache | PASS | inference_mode, eval, no gradients, meta construction, mmap/assign loading, one-entry shared model cache and lifecycle lock |
| All four local models | PASS | Actual sequential Standard forwards, own metadata/preprocessing, cache eviction, weak-reference release checks, partial failure recovery |
| Hosted comparison disabled before resource acceptance | PASS | Explicit cloud configuration requires both enable=true and hosted-verification=true; remote repository alone never enables it |
| Genuine experimental metrics | PASS | Public CSV matches supplied experiment data; separate from video predictions; no historical latency presented as CPU latency |
| Python/dependency compatibility | PASS locally | Python 3.11.9, pinned CPU pair, headless OpenCV; pip check has no conflicts; no reinstall needed |
| Actual Debian/Community Cloud build | NOT VERIFIED | Paths use pathlib/forward slashes and requirements use supported CPU builds; only Windows runtime was executed |
| Bounded upload and cleanup | PASS | 30 MiB, 30 seconds, resolution/FPS/frame caps; invalid metadata, corrupt input and temp-file cleanup tests |
| Upload/reset and responsive UI | PASS | Existing pipeline/UI tests; prior full upload browser checks recorded in UI_REFINEMENT_VERIFICATION.md; fresh cloud startup/Clear/disabled comparison and desktop/mobile checks below |
| Public HF configuration and caching implementation | PASS locally | Exact repo/filename/revision passed to hf_hub_download; real selected checkpoint loaded and checksummed after substituted transport; second call reuses same cached resource |
| Actual public HF download | NOT VERIFIED | No actual public repository ID supplied; local transport substitution does not test network/permissions |
| Git exclusion and already tracked private files | PASS | git check-ignore confirms representative formats; git ls-files is empty (repository has no tracked files yet); publishable untracked inventory inspected |
| Real human-action video acceptance | NOT VERIFIED | Only codec-fixture.avi and corrupt.mp4 present; no suitable real action video supplied |
| Free-host Standard/Enhanced memory, concurrency and cold start | NOT VERIFIED | Local measurements below; actual public URL required |
| Free-host four-model live comparison | NOT VERIFIED / disabled | Keep local comparison and public CSV metrics; no claim of hosted capacity |
| Paid services required | PASS | CPU inference, public HF file download, GitHub and Community Cloud workflow; no inference API, Docker or separate backend |

No unresolved local test failures were found. NOT VERIFIED items are release acceptance work, not completed deployment claims.

## Minimal changes

- `src/motionmirror/comparison.py`: explicit local/cloud policy and conservative auto mode. `DEPLOYMENT_MODE` accepts auto/local/cloud. Auto recognizes a complete local checkpoint bundle; otherwise hosted policy applies. Cloud comparison requires `ENABLE_MODEL_COMPARISON=true` **and** `HOSTED_COMPARISON_VERIFIED=true`, which must reflect recorded hosted evidence.
- `src/motionmirror/comparison_ui.py`: replace the old enable-flag-only help text with accurate configuration guidance; preserve layout, metrics and all controls.
- `.env.example`: document deployment mode and default hosted verification=false.
- `.gitignore`: exclude checkpoint formats anywhere, private notebooks, raw archives, environments, uploads, credentials, temporary files and caches; retain `.env.example`.
- `tests/test_comparison.py`: existing remote-source auto test now explicitly selects local execution.
- `tests/test_deployment.py`: four deployment boundary/configuration tests.
- `README.md`: exact HF download test, Cloud secrets, Python selection, Git review, hosted policy, live acceptance and troubleshooting.
- This audit report. Requirements, selected checkpoint, manifest, preprocessing, inference, main UI, CSS, theme and experimental CSV are unchanged.

## Verification and measurements

Targeted deployment/enablement tests: **5 passed**, 24 deselected, 24.19 s.
Full suite: **45 passed**, 79.23 s (41 existing tests plus four deployment tests). Final direct verification: **45 passed in 62.19 s, pytest_exit=0, process exit 0**. The direct run confirmed the exit status after the redirected run reported passing tests but its shell returned 1.
`python -m pip check`: **No broken requirements found**.

Installed: Python 3.11.9; torch 2.7.1+cpu; torchvision 0.22.1+cpu; streamlit 1.50.0; opencv-python-headless 4.11.0.86 (cv2 4.11.0); numpy 2.2.6; huggingface-hub 0.35.3; psutil 7.1.0. No packages changed.

Fresh-process local CPU runs with the unchanged actual weights and **generated codec fixture** (50 frames, 10 FPS, 80×64; not a human action):

| Run | Load (s) | Processing / total (s) | Sampled peak RSS (MiB) |
|---|---:|---:|---:|
| MViT Standard, one centered clip | 0.257 | 8.312 processing | 760.8 |
| MViT Enhanced, three clips | 0.288 | 9.651 processing | 799.8 |
| Four models sequential Standard | See below | 16.959 total | 776.0 |

| Comparison architecture | Load (s) | Forward (s) | Clip processing (s) |
|---|---:|---:|---:|
| R(2+1)D-18 | .447 | 3.826 | 3.837 |
| R3D-18 | .247 | 7.714 | 7.725 |
| MC3-18 | .228 | 1.976 | 1.986 |
| MViT-V2-S | .243 | 1.506 | 1.523 |

Four actual results, zero errors, CLI exit 0; comparison starting RSS 281.1 MiB. Timings vary with host load and are not a controlled architecture speed ranking. RAM includes imported runtime and allocator retention, sampled every 20 ms for selected-video verification / 10 ms for comparison; brief peaks can be missed. It excludes separate server processes and does not simulate concurrent visitors. Checkpoint parameters alone occupy 130.592 MiB; this is much less than process peak usage. Some local peaks exceed the published minimum host memory range, so even selected-model capacity is not guaranteed.

Local-only raw records: `test-results/deployment-standard.json`, `deployment-enhanced.json`, `deployment-four-model.json`, and `deployment-tests.txt`. These artifacts remain ignored by Git. Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_video.py test-results\codec-fixture.avi
.\.venv\Scripts\python.exe scripts\verify_video.py test-results\codec-fixture.avi --mode Enhanced
.\.venv\Scripts\python.exe scripts\compare_models.py test-results\codec-fixture.avi --output test-results\deployment-four-model.json
```

Cloud-configured browser preview: `http://127.0.0.1:8503/` (loopback only). At 1280 pixels, panels share one row and document width is 1280. At 390 pixels, panels stack and document width is 390; the experimental table scrolls within a 308-pixel region (398-pixel contents). Standard is selected, Clear works, live comparison is disabled, and four genuine metric rows appear. Screenshots: `test-results/deployment-desktop.jpg` and `deployment-mobile.jpg`. No redesign performed. Earlier real browser upload/invalid-file/Standard/Enhanced/full-four-model interaction evidence is in `UI_REFINEMENT_VERIFICATION.md`; today's fixture CLI and automated tests recheck the unchanged inference path.

Selected checkpoint SHA256 remains `f8666134bbe97f0ce5dd292e0a9f9c983fccaa7a659aa3d9529b056fca678ccf`.

## Current official hosting guidance

Checked 2026-10-16:

- [Streamlit deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy): choose repo/branch/entrypoint; configure secrets and explicitly choose tested Python **3.11** (current default is 3.12).
- [Resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#resource-limits): approximate published ranges, dated February 2024, are 0.078–2 CPU cores, 690 MB–2.7 GB memory, up to 50 GB storage; can change without notice. No guaranteed allocation inferred. Idle apps can sleep after 12 hours.
- [Platform limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status): Debian Linux, forward-slash paths, one package-manager configuration. Actual hosted build remains unverified.
- [Dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) and [PyTorch supported version pairs](https://pytorch.org/get-started/previous-versions/): retain requirements.txt and the CPU index; 2.7.1/0.22.1 is an official Linux/Windows pair. Headless OpenCV avoids GUI library dependencies.
- [HF repository uploads](https://huggingface.co/docs/hub/en/repositories-getting-started): manually create a public Model file repository and upload only the selected inference checkpoint for initial deployment.

## Manual release acceptance

Follow the complete copyable Windows/Git/HF/Cloud commands and TOML in `README.md` from the **inner project directory containing streamlit_app.py**.

1. Review the Git staged inventory, create/push the public source repository. No checkpoint, notebook, credentials, dataset or video should be staged.
2. Create the free public HF Model repository, upload unchanged `models/model.pth` as `model.pth`, record repo ID/full revision and verify SHA256. Run README's forced-download check on a real action video.
3. In Community Cloud: Create app → GitHub repo → main → streamlit_app.py → Advanced settings → Python 3.11 → paste README secrets with real repo ID. Keep deployment=cloud, comparison=false, hosted verification=false. Deploy.
4. Record the actual public URL only after it loads. Run README's live acceptance checks, inspect build/download/runtime logs, measure cold/warm Standard memory/time, then test Enhanced and expected visitor load. Keep comparison disabled unless separately validated on the actual free host.
5. If resource limits prevent reliable selected MViT inference, record the failure and leave readiness unverified/failed; preserve the trained model and free-only requirement. Do not claim success, silently substitute another model or require a paid plan.

Public URL: **not deployed**. Hosted load/prediction/RSS: **not measured**. Real action video: **not supplied**.
