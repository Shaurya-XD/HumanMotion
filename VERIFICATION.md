# Verification — 12 October 2026

The subsequent optional four-model extension is recorded separately in
`COMPARISON_VERIFICATION.md`. Its final suite retains all checks below and reports
41 passed with no skips after all four trained checkpoints were supplied and
live sequential comparison was enabled. The older single-model checks below
remain valid; current four-model measurements are in that separate report.

## Executed successfully

- Installed `requirements-dev.txt` in a newly created Windows Python 3.11 virtual environment. CPU Torch 2.7.1 / TorchVision 0.22.1, Streamlit 1.50.0 and headless OpenCV 4.11.0.86. `pip check`: no broken requirements.
- Safely inspected the actual checkpoint: `mvit_v2_s`, five-class mapping, 16 frames, 224 crop, two-second window, normalization 0.45 / 0.225. Head tensors: `(5,768)` and `(5,)`. Strict CPU reconstruction succeeded.
- Final automated suite: **16 passed in 26.84 seconds**, exit code 0. Includes actual checkpoint five-logit/finite-softmax checks, class order, model cache identity, deterministic repeated Standard prediction, exact Enhanced softmax averaging, exact notebook preprocessing equality, short/portrait/landscape fixtures, corrupt-file errors, upload size and metadata limits, cleanup, configuration priority, missing weights, bad checksum, and Streamlit AppTest startup/Clear.
- Python compilation of app, source and scripts completed.
- Started actual server with `python -m streamlit run streamlit_app.py --server.address 127.0.0.1`; inspected the live website in the Codex browser.
- Browser Standard analysis on a generated 50-frame, 10 FPS MJPEG codec fixture displayed five real checkpoint scores, descending bars and processing details. Tested invalid MP4 → readable error → Clear → valid upload recovery. Tested mode interaction and Enhanced analysis.
- Desktop 1280-pixel viewport and mobile 390×844 viewport checked. Input/output columns stack on mobile. Document width equals viewport width (390), with no horizontal overflow. Clear removes uploaded file, error and result; Analyze disables without input.
- Initialized local Git repository without committing or publishing. `git check-ignore` confirms checkpoint, private notebooks, virtual environment and Streamlit secrets are ignored; eligible source listing contains none of them.

## CPU benchmark (local, not hosted)

`python scripts/benchmark_cpu.py`: supplied checkpoint, zero input `[1,3,16,224,224]`, float32, two threads, one warmup and three measured forwards. Intel64 Family 6 Model 165 Stepping 3.

| Measurement | Observed value |
|---|---:|
| Load/reconstruction | 0.249 s |
| Measured forwards | 1.378 / 1.317 / 1.414 s |
| Median model forward | 1.378 s |
| Parameter storage | 130.592 MiB |
| Sampled peak process RSS | 776.473 MiB |

`scripts/verify_video.py` on the codec fixture: processing 2.005 seconds, sampled peak process RSS 770.230 MiB. Browser Standard processing was 1.68 seconds. These are workload checks on generated imagery, **not real-action recognition results**. Test processes and browser conditions differ; do not compare them as controlled performance trials. RSS includes Python/Torch imports and allocations and was sampled, not an exact allocator peak. Free Linux host memory/latency can differ; the live multi-session server's peak was not measured.

## Unverified / remaining work

- No real action sample was provided anywhere in the project. Real-video semantic correctness and evaluation accuracy were not tested. Run `scripts/verify_video.py` with a real action video and use the uploader for acceptance testing. Do not interpret fixture predictions as accuracy evidence.
- Browser MJPEG AVI playback is unsupported; OpenCV decoding and inference passed. Browser playback of a real H.264 MP4 still needs a supplied sample.
- No external publication was authorized. Hugging Face/release download against the user's eventual public model repository and deployed Community Cloud inference remain untested. Follow README to publish and verify.
- Free-host capacity is not guaranteed. Approximately 776 MiB local process RSS leaves limited hosting headroom. Test Standard on the actual host before enabling heavier usage; multi-session uploads can add RAM despite serialized inference. No model substitution was made.
- Clean installation was verified on Windows Python 3.11; Linux Community Cloud installation remains a deployment check.

Checkpoint SHA256: `f8666134bbe97f0ce5dd292e0a9f9c983fccaa7a659aa3d9529b056fca678ccf`.
