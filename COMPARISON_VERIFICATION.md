# Four-model live comparison verification — 13 October 2026

## Outcome

All four supplied trained checkpoints were safely inspected, strictly reconstructed on CPU, and executed sequentially on the same video. Local comparison enables automatically when the four files exist. The exact order is R(2+1)D-18 → R3D-18 → MC3-18 → MViT-V2-S. No application inference timeout, model substitution, ensemble or quantization was introduced.

## Checkpoints and preprocessing

| File | Architecture | Head | Frames / crop |
|---|---|---|---|
| models/r2plus1d_18.pth | R(2+1)D-18 | Dropout(0.5), Linear(512,5) | 16 / 112×112 |
| models/r3d_18.pth | R3D-18 | Dropout(0.5), Linear(512,5) | 16 / 112×112 |
| models/mc3_18.pth | MC3-18 | Dropout(0.5), Linear(512,5) | 16 / 112×112 |
| models/model.pth | MViT-V2-S | Dropout(0.5), Linear(768,5) | 16 / 224×224 |

All checkpoints save walk=0, pushup=1, sit=2, drink=3, clap=4 and a two-second temporal window. CNN preprocessing uses resize 128×171, crop 112, mean (0.43216,0.394666,0.37645) and std (0.22803,0.22145,0.216989). MViT uses short-side resize 256, crop 224, mean 0.45 and std 0.225. Evaluation sampling, float-before-resize, RGB conversion and center cropping are unchanged. The R(2+1)D checkpoint omits architecture/normalization metadata; these fields use the matching notebook-verified preset. The three CNN files include optimizer/scaler state, which is not used for inference. Originals were not rewritten.

## Tests actually executed

- Initial actual checkpoint checks: 5 passed, including four independent predictions and a full comparison.
- Comparison suite after progressive failure handling: 23 passed.
- Final complete suite: **41 passed in 57.38 seconds, no skips, native pytest exit code 0**. Includes all original 16 tests.
- Actual CPU inference verifies five logits, finite five-class softmax vectors summing to one, exact class mapping, architecture-specific preprocessing, and Standard-only comparison.
- Full-sequence lifecycle test primes the selected MViT cache, verifies eviction before the first CNN, and uses weak references to check each actual network is released before the next loads. Ordinary MViT cache/inference works afterward.
- Progress snapshots contain 1, 2, 3 and 4 completed predictions. Tested the exact four Running messages and Comparison complete.
- An incompatible R3D checkpoint and a missing R3D file were tested: preceding predictions survive, later real models execute, and architecture-specific errors are shown. No model outputs are mocked.
- Actual-result renderer tested summary columns, score matrix, probability bars and partial errors. Hosted transport configuration tested using the actual locally cached MViT checkpoint; no claim is made that a remote user repository was contacted.
- pip check: no broken requirements; Python compilation completed. No new dependency installation was needed.

## Measured local performance

The benchmark used the supplied trained weights with a generated 50-frame, 10 FPS, 80×64 MJPEG codec fixture, float32 CPU, batch size one and two Torch threads. It is implementation/performance evidence, **not a real-action accuracy evaluation**. No real human-action sample was supplied.

| Model | Forward inference | Loading | Clip processing |
|---|---:|---:|---:|
| r2plus1d_18 | 0.763 s | 0.082 s | 0.774 s |
| r3d_18 | 0.516 s | 0.042 s | 0.525 s |
| mc3_18 | 0.561 s | 0.046 s | 0.570 s |
| mvit_v2_s | 1.613 s | 0.241 s | 1.642 s |

CLI total: **4.694 seconds**; sampled peak process RSS: **765.7 MiB**; starting RSS: 280.0 MiB. Full scalar output: `test-results/four-model-benchmark.json` (ignored by Git).

Live Streamlit browser comparison also succeeded: **4.54 seconds**, **787 MiB** sampled peak RAM, 292 MiB starting RAM. Browser forward times were 0.707 / 0.511 / 0.531 / 1.505 seconds in the required model order. Workload and UI overhead differ from the CLI benchmark; these are observations, not controlled performance comparisons. RSS includes runtime/session memory and was sampled every 10 ms, so brief peaks may be missed. CPU allocators can retain freed pages after network release.

### Actual fixture outputs (not human-action recognition evidence)

| Model | Predicted label | Top score | Walk | Pushup | Sit | Drink | Clap |
|---|---|---:|---:|---:|---:|---:|---:|
| r2plus1d_18 | walk | 31.25% | 31.25% | 17.15% | 26.73% | 14.01% | 10.86% |
| r3d_18 | clap | 36.14% | 23.78% | 35.98% | 2.84% | 1.26% | 36.14% |
| mc3_18 | clap | 27.61% | 16.60% | 18.16% | 18.34% | 19.29% | 27.61% |
| mvit_v2_s | clap | 30.32% | 18.64% | 24.05% | 10.44% | 16.54% | 30.32% |

These are actual independent trained-model outputs on generated colored frames. Disagreement is preserved, and no ensemble is applied. They must not be interpreted as evidence that a person performed the reported action.

## Live UI and cleanup

Browser verified the automatically enabled comparison button after upload, live Running R(2+1)D-18 (1/4) progress, 4/4 completion, summary table, action matrix and 20 probability bars. Results appear progressively and the existing design is preserved. Desktop/mobile evidence: `test-results/four-model-live-desktop.jpg` and `test-results/four-model-live-mobile.jpg`. At mobile width 390 the document width remains 390; score tables scroll inside their own containers. Uploaded MJPEG AVI decoded correctly but is not playable by the browser; real H.264 MP4 playback still needs a supplied sample.

Windows preview exposed a watchdog cross-drive path error (Python on C:, project on D:) that left imported modules stale. Switching Streamlit fileWatcherType to poll and restarting fixed the live preview.

After live comparison, the existing Analyze Video button completed ordinary MViT Standard analysis in 1.78 seconds. Browser Clear removed the upload and all comparison results; Analyze disabled again and only the historical experiment table remained. Original automated tests also verify Enhanced averaging/determinism. Temporary-file success/failure cleanup remains covered by the original suite.

## Files changed in this completion

`src/motionmirror/comparison.py`, `src/motionmirror/comparison_ui.py`, `scripts/compare_models.py`, `tests/test_comparison.py`, `.env.example`, `.streamlit/config.toml`, `README.md`, `COMPARISON_VERIFICATION.md`, and the verification index. The trained checkpoints, original 16 tests, model manifest, model reconstruction/preprocessing and main analysis logic were preserved.

## Run and deploy

Local: from the folder containing streamlit_app.py run `.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py`. Upload a real short action video, expand Compare All Models, and select its button. All four detected files enable comparison without another flag. If an old environment/secrets value explicitly sets ENABLE_MODEL_COMPARISON=false, change it to auto or true and restart. Clear resets input and comparison state.

Free hosting: live comparison is allowed through ENABLE_MODEL_COMPARISON=auto/true and COMPARISON_HF_REPO pointing to a manually published free public model file repository with all four correctly named checkpoints. Missing local files download/cache sequentially; a failed model download is reported individually. Primary Standard/Enhanced analysis retains its separate selected-model source/cache. Prefer inference-only CNN exports for publishing, without overwriting local originals. No files were uploaded externally.

**Streamlit Community Cloud feasibility is not tested.** Local full comparison succeeds around 766–787 MiB sampled RAM and about five seconds on this machine, making it a candidate for free CPU hosting; this is not a guarantee of Linux host RAM, resource quotas, cold-start downloads or concurrent-session capacity. Test the deployed site, record any platform failures, and explicitly disable live comparison only if needed. The actual experimental CSV table and fully functional local comparison remain available as fallback. No paid endpoint, hosting subscription, managed GPU or model replacement was introduced.
