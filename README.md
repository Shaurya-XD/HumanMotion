# MotionMirror

A Streamlit workspace using the supplied **MViT-V2-S** five-action checkpoint by default, with optional local comparison of all four trained models. Video predictions come from actual checkpoint inference; the separate experimental table uses the supplied evaluation CSV. No training, generic Kinetics predictions, paid inference service, database, or GPU is required.

Latest release audit: [DEPLOYMENT_AUDIT.md](DEPLOYMENT_AUDIT.md). Local verification passes; actual public deployment and free-host capacity remain unverified.

## Local setup (Windows PowerShell)

Open PowerShell in the project directory containing `streamlit_app.py`:

```powershell
cd D:\ML_Lab\MotionMirror-Web-Codex\MotionMirror-Web-Codex
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Use Python **3.11** locally and on the host. The pinned PyTorch wheels are CPU builds; Python 3.14 is not supported by these pins. Activation is optional because the commands explicitly select the environment. Visit `http://localhost:8501`. The provided local `models/model.pth` works without configuration. `.env.example` is a configuration reference; `.env` files are **not automatically loaded**. Use environment variables or `.streamlit/secrets.toml`.

Upload a real short action video, choose Standard, and select **Analyze Video**. Clear resets the uploader and result; changing the video or mode clears stale scores. Enhanced averages softmax scores across beginning/middle/end clips, deduplicating identical windows. It has not been separately evaluated for accuracy.

## Verified model contract

- Mapping: `walk=0`, `pushup=1`, `sit=2`, `drink=3`, `clap=4`.
- HMDB51 Official Split 1; architecture selected by validation macro F1.
- Kinetics-pretrained backbone, transferred and fine-tuned for five actions; head is `Dropout(0.5) → Linear(768, 5)` for MViT.
- `weights=None` reconstruction, safe `torch.load(weights_only=True)`, strict state loading, evaluation mode, float32 CPU inference.
- 16 uniformly sampled RGB frames from a centered window of `round(2 × FPS)` frames, using PyTorch linspace/round exactly as in the notebooks.
- Convert float32 before bilinear antialiased resize to short side 256, center crop 224×224, normalize mean `(0.45,0.45,0.45)` / std `(0.225,0.225,0.225)`, tensor `[1,3,16,224,224]`.
- The notebook comparison CSVs are historical experiment records. Their latency figures are **not** measurements of this CPU deployment. They are not presented as hosted performance.

This is a closed-set classifier: unsupported actions still receive one of the five labels. Scores are softmax outputs, not calibrated probabilities of correctness.

## Resource and privacy behavior

Upload cap: 30 MB; duration: 30 seconds; up to 120 FPS / 3,600 decoded frames; resolution: at most 2,073,600 pixels, minimum 16 pixels per side, maximum side 4,096. Portrait and landscape are supported. H.264 MP4 is recommended for browser playback. WebM/MOV/AVI/MKV decoding depends on their codecs.

Decoding uses two sequential passes: one to establish the actual readable frame count, one to collect the requested indices. This intentionally avoids inaccurate OpenCV seeking and retains only sampled, resized frames instead of an entire float video. A truncated video can use its readable prefix with a warning. Invalid FPS is rejected rather than inventing timing.

`st.cache_resource` loads one model per configuration per process. A model lock serializes decoding/inference across visitors to bound model working memory. CPU threads default to two (`TORCH_NUM_THREADS=1..4`). Model construction uses meta tensors and assigns memory-mapped checkpoint tensors to avoid a second parameter allocation. Uploaded video bytes remain in the active Streamlit session for preview; temporary disk files are removed on success and failure. No permanent video storage is implemented.

Free hosting has shared CPU/RAM, cold starts and resource limits; local success does not guarantee host capacity. Test the deployed site with Standard first and one visitor. Avoid concurrent large uploads. If the host reports memory exhaustion, retain this checkpoint and document the failure; do not substitute another model or require a paid upgrade. Community Cloud's current limitations are documented [here](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).

## Compare All Models (sequential CPU inference)

The selected MViT Standard analysis remains the default. Expand **Compare All Models** for a separate comparison workspace. Live comparison always uses **one centered Standard clip per model**, regardless of the main workspace's Standard/Enhanced selector. It displays each model's predicted action, top softmax score, model-forward time, clip-processing time, all five action scores, and compact per-model bars. The historical experiment table is separately labeled and is available even when live comparison is disabled.

Add the **actual trained five-class checkpoints**, not generic pretrained weights, at these locations:

```text
models/r2plus1d_18.pth
models/r3d_18.pth
models/mc3_18.pth
models/model.pth          # Existing MViT file; preserve it.
```

If your training export calls a file `best.pth`, copy that architecture's inference checkpoint to its name above. Do not rename another architecture's file to satisfy a missing-file check. Use inference-only exports without optimizer state to reduce disk/RAM load. `COMPARISON_MODELS_DIR` can point to another local directory for the three CNN files; `MODEL_PATH` continues to identify the existing MViT checkpoint.

All four supplied checkpoints are present and verified. `DEPLOYMENT_MODE=auto` (default) recognizes the complete local bundle and enables local comparison when a valid video is uploaded. Use `DEPLOYMENT_MODE=local` to explicitly select local execution. `ENABLE_MODEL_COMPARISON` accepts `auto` (default), `true`, or `false`; explicit `false` disables live comparison. Ordinary Standard/Enhanced analysis remains available. No predictions are inserted from the experimental CSV.

Run locally in PowerShell:

```powershell
cd D:\ML_Lab\MotionMirror-Web-Codex\MotionMirror-Web-Codex
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

If you previously set `ENABLE_MODEL_COMPARISON = "false"` in environment variables or `.streamlit/secrets.toml`, change it to `auto` or `true` and restart the server. `.env.example` is documentation, not automatically loaded. Upload a video, expand **Compare All Models**, and select its button. It runs **R(2+1)D-18 → R3D-18 → MC3-18 → MViT-V2-S**. Progress names the current architecture and results appear as each finishes. A model failure is shown for that architecture, completed results are retained, and remaining models are attempted. The summary table, action-score matrix and probability bars show independent results; no ensemble is computed. Clear and a changed input reset comparison results/errors. A mode change also clears results, although comparison itself is always Standard.

There is **no application comparison timeout**, and slow models are not skipped. The existing 30-second cap applies to the **uploaded video's duration**, not inference execution. This build uses float32 CPU inference, batch size one, and CPU-only Torch wheels; no CUDA allocation/cache is used. Local GPU inference is not enabled by these pinned CPU builds.

For CLI comparison on a real video:

```powershell
.\.venv\Scripts\python.exe scripts\compare_models.py C:\path\to\real-action.mp4
.\.venv\Scripts\python.exe scripts\compare_models.py C:\path\to\real-action.mp4 --output test-results\my-comparison.json
```

The CLI reports actual per-model predictions, individual errors and measured total duration / sampled peak process RSS. With `--output` it saves the full JSON and prints compact timings. Exit code 0 means all four succeeded; exit 1 with a partial JSON means at least one failed, while other results are preserved. If every source is missing or the video is invalid, it gives a setup error before inference. CLI operation is explicit usage and does not require the web UI enable flag.

Each architecture is reconstructed with its own safe checkpoint load and a strict five-output head. Each saved class order is respected and scores are aligned by action name in the table. Saved frame count, crop resolution, temporal window and normalization are validated. The original R(2+1)D notebook omitted architecture/normalization metadata; only those missing fields use its configured architecture and verified Kinetics preprocessing preset. Missing temporal/spatial/class settings, six-class checkpoints, wrong architectures, or incompatible normalization are rejected.

CNN preprocessing uses the TorchVision preset's 128×171 resize, 112×112 crop, RGB mean `(0.43216, 0.394666, 0.37645)` and std `(0.22803, 0.22145, 0.216989)`. MViT retains its 256 short-side resize, 224 crop, mean 0.45 and std 0.225. All use float conversion before antialiased bilinear resizing and deterministic saved temporal sampling. MViT requires 16 frames; CNNs accept saved positive frame counts up to 64. None uses training-time augmentation.

**Memory lifecycle:** a process-wide lock covers Standard model acquisition/inference and the full comparison. The selected resource cache is limited to one entry. Before comparison, `get_model.clear()` evicts that resource and garbage collection releases it; UI sessions retain only scalar results, never networks. Comparison loads one uncached network, predicts, deletes it, and collects before loading the next. No list/cache stores all four models. Subsequent selected-model analysis lazily restores its normal cache. Other visitors' selected analysis waits while comparison runs. CPU allocators may retain freed pages, so process RSS may remain elevated despite network release.

**Measurements:** model-forward time excludes decoding/loading; processing includes temporal-frame extraction and preprocessing. Total comparison includes cache cleanup, video validation, checkpoint loading and all four attempts (queue waiting excluded). The 10 ms process-RSS sampler spans validation/loading/inference/cleanup, includes Streamlit/runtime/session memory, and reports starting RAM as well as peak RAM. Sampling can miss brief peaks; it is not a host allocator limit test. A local four-model CLI run on a generated codec fixture completed in **4.694 seconds**, with **765.7 MiB sampled peak RSS**. This is pipeline/performance evidence, not a real-action accuracy evaluation. See `COMPARISON_VERIFICATION.md` for executed tests and measured scope.

**Free deployment:** deploy with `DEPLOYMENT_MODE="cloud"`, `ENABLE_MODEL_COMPARISON="false"`, and `HOSTED_COMPARISON_VERIFIED="false"`. Hosted four-model feasibility is **NOT VERIFIED**. The public site uses selected MViT Standard/Enhanced analysis and the genuine experimental CSV table; live four-model comparison remains available locally. A public comparison repository alone never enables hosted comparison. In auto deployment mode, an incomplete local checkpoint bundle uses this conservative hosted policy.

Only after a separate, controlled test on the actual free host records reliable cold/warm runs, peak RSS, total duration and expected visitor load should its operator set both `ENABLE_MODEL_COMPARISON="true"` and `HOSTED_COMPARISON_VERIFIED="true"`. This explicit attestation records an operator decision; the app cannot establish host reliability automatically. Until that evidence exists, leave both false. No such hosted test has been performed for this project.

The optional remote transport remains available for controlled tests or explicit local use: `COMPARISON_HF_REPO` names a public file repository containing `r2plus1d_18.pth`, `r3d_18.pth`, `mc3_18.pth`, and `model.pth`; `COMPARISON_HF_REVISION` defaults to `main` (prefer a full commit hash). Selected-model `HF_MODEL_REPO`/`HF_MODEL_FILENAME` remain separate. Files download sequentially and failed downloads preserve completed model results. No files have been published externally by this implementation. Do not publish optimizer/scaler state from the supplied CNN checkpoints; use separately verified inference-only exports if future hosted tests are authorized.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_video.py C:\path\to\real-action.mp4
.\.venv\Scripts\python.exe scripts\verify_video.py C:\path\to\real-action.mp4 --mode Enhanced
```

Tests use the **actual checkpoint**, including forward shape, finite scores, mapping, cache identity, deterministic repeated inference, and exact multi-clip softmax averaging. Generated codec fixtures test preprocessing/decoding only; they are not evidence of real-action accuracy. The preprocessing test compares the bounded decoder against the complete notebook transform with exact tensor equality. Streamlit AppTest covers startup and Clear; browser checks cover uploader, real rendered layout and responsive widths. See `VERIFICATION.md` for checks actually executed and measured limitations.

Manual acceptance: upload a real supported action, analyze twice, inspect scores, try Enhanced, change the video, Clear, upload a corrupt file, recover with a valid file, and check desktop/mobile keyboard navigation. The CLI reports model load time, processing time and a sampled process RSS peak. RSS includes the Python runtime; the sampling interval can miss brief peaks.

## Publish code to free GitHub

No external account has been created or published by this implementation. In GitHub, create a **public** repository named `motionmirror` without initializing a README. Install Git if needed, then run in this project folder:

```powershell
git init
git add .
git status --short
git check-ignore models/model.pth references/MotionMirror_5Class_Training.ipynb
git diff --cached --name-only
# Confirm no .pth, private notebooks, secrets, uploads, or .venv files are listed.
git commit -m "Build MotionMirror CPU video recognition app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/motionmirror.git
git push -u origin main
```

`.gitignore` excludes checkpoint formats anywhere, private notebooks/references, raw archives, videos/uploads, virtual environments, `.env*` (except `.env.example`), credentials and caches. Keep `model_manifest.json` in the repository. The repository root must contain `streamlit_app.py` and `requirements.txt`; do not upload the outer wrapper folder as the repository root. Never use `git add -f` for excluded assets. Ignore rules do not remove previously tracked files: inspect `git ls-files` before publishing. If a private file is already tracked, remove it from the index and review history before pushing; rotate any previously published credentials.

## Publish weights to a free public Hugging Face model repository

1. Create a free account at [Hugging Face](https://huggingface.co/join).
2. Select your profile → **New Model** (or [new model repository](https://huggingface.co/new)). Name it `motionmirror-5class`, set visibility **Public**, and create it. This is a file repository, not a Space or inference endpoint.
3. Open **Files and versions** → **Add file** → **Upload files**. Upload the supplied inference-only `models/model.pth` as `model.pth`. Do not upload notebooks or training data. Add a model card describing the architecture, mapping, preprocessing and intended limitations; check your rights to publish the trained weights.
4. Record the repository ID `YOUR_USERNAME/motionmirror-5class` and optionally the full commit revision from Files and versions. Public downloads require no token or paid API.
5. Compute the checksum locally:

```powershell
(Get-FileHash models\model.pth -Algorithm SHA256).Hash.ToLower()
```

The supplied checkpoint checksum is `f8666134bbe97f0ce5dd292e0a9f9c983fccaa7a659aa3d9529b056fca678ccf`. Confirm your uploaded copy matches it. See the official [Hugging Face repository upload guide](https://huggingface.co/docs/hub/en/repositories-getting-started).

Alternatively, create a GitHub Release and attach `model.pth` as a release asset (not a tracked source file). Set `MODEL_URL` to its HTTPS download URL. Resolution priority is existing `MODEL_PATH`, then Hugging Face, then the release URL. If a configured HF download fails, its error is reported; an unrelated model is never substituted. HF downloads use the HF disk cache, while release downloads use an atomic temp-directory cache. Both are behind the process model cache. A new hosting process may need to download again.

## Deploy on Streamlit Community Cloud (free)

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with your free GitHub account. Choose **Community Cloud**, the free service; no paid hosting or inference endpoint is part of this setup.
2. Grant the required repository access and select **Create app** → deploy from the GitHub repository.
3. Repository: `YOUR_USERNAME/motionmirror`; branch: `main`; main file: `streamlit_app.py`.
4. In **Advanced settings**, select Python **3.11**. Add this TOML in **Secrets**, replacing the example account. Keep the checksum below for the unchanged supplied checkpoint:

```toml
MODEL_PATH = "models/model.pth"
HF_MODEL_REPO = "YOUR_USERNAME/motionmirror-5class"
HF_MODEL_FILENAME = "model.pth"
HF_MODEL_REVISION = "main" # Prefer a recorded commit hash for reproducibility.
MODEL_SHA256 = "f8666134bbe97f0ce5dd292e0a9f9c983fccaa7a659aa3d9529b056fca678ccf"
TORCH_NUM_THREADS = "2"
DEPLOYMENT_MODE = "cloud"
ENABLE_MODEL_COMPARISON = "false"
HOSTED_COMPARISON_VERIFIED = "false"
```

5. Select **Deploy**. `requirements.txt` installs CPU-only Torch/TorchVision and headless OpenCV. There is no separate backend to configure.
6. Open the resulting `*.streamlit.app` URL. Upload a real supported-action MP4 and run Standard; check the five scores, processing time and Clear. Check host logs for memory/resource failures. Record hosted observations separately from local benchmarks. Test Enhanced only after Standard succeeds.

7. On the live URL, repeat Standard on the same video, verify finite scores sum to approximately 100%, change the upload, Clear, try a corrupt file, then recover with a valid file. Verify the experimental comparison table is visible and live comparison is disabled. Check a phone-width viewport and keyboard controls. Record the actual URL, Python version, logs, cold-start/download time and resource measurements in `DEPLOYMENT_AUDIT.md`; only then mark hosted acceptance PASS.

The Cloud UI currently defaults to Python 3.12; explicitly select **3.11** for the tested configuration. The official resource page currently lists approximate ranges (dated February 2024): 0.078–2 CPU cores, 690 MB–2.7 GB memory, and up to 50 GB storage. These are changeable limits, not guaranteed allocations; even selected MViT needs real hosted testing. See [deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#resource-limits), checked 2026-10-16. Headless OpenCV needs no GUI Linux packages; retain the existing requirements and CPU wheel index.

After creating your public HF repository, test the **download** locally using a nonexistent `MODEL_PATH` (an existing local file otherwise takes priority):

```powershell
$env:MODEL_PATH = "models/download-check-only.pth"
$env:HF_MODEL_REPO = "YOUR_USERNAME/motionmirror-5class"
$env:HF_MODEL_FILENAME = "model.pth"
$env:HF_MODEL_REVISION = "main" # Or the uploaded commit hash.
$env:MODEL_SHA256 = "f8666134bbe97f0ce5dd292e0a9f9c983fccaa7a659aa3d9529b056fca678ccf"
.\.venv\Scripts\python.exe scripts\verify_video.py C:\path\to\real-action.mp4
Remove-Item Env:MODEL_PATH, Env:HF_MODEL_REPO, Env:HF_MODEL_FILENAME, Env:HF_MODEL_REVISION, Env:MODEL_SHA256
```

Configuration is read from nonempty environment variables first, then top-level keys in `.streamlit/secrets.toml` locally or Cloud **Settings → Secrets**. `.env.example` is a reference, not an automatically loaded file. The download test requires your actual public repository; local transport tests cannot prove its permissions or network availability.

For a Release asset, omit `HF_MODEL_REPO` and set `MODEL_URL` in Secrets instead. A public HF repository is preferred. App code does not expose secrets or raw exceptions to visitors. Public model files can be downloaded by anyone. Never enter account tokens in tracked files.

Streamlit documents its [free Community Cloud service](https://docs.streamlit.io/deploy/streamlit-community-cloud) and [deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). Follow the free Community Cloud flow; external account UI and policies can change.

## Troubleshooting

- **Missing model:** check that the local path exists or that the public HF repo ID/filename/revision are exact. Restart after changing configuration.
- **Checksum/strict state mismatch:** use the supplied inference-only checkpoint and matching manifest. Do not turn off strict loading or load an unknown pickle.
- **Dependency failure:** use Python 3.11, ensure the CPU index is reachable, and inspect build logs. Avoid installing GUI OpenCV or CUDA Torch. Run `pip check` locally.
- **Decoder failure:** re-export as H.264 MP4 with valid FPS, fixed dimensions and a short duration. The UI recovers after invalid input.
- **Slow inference:** start with Standard, a two-to-five-second clip and two CPU threads. Compare one thread if the host is throttled. Do not enable compilation or mixed precision without validating outputs.
- **Memory exhaustion:** retry with one short upload and Standard, reboot to release old resources, then document the observed host limit and process memory. This implementation does not claim guaranteed capacity on every free host.
- **Preview cannot play:** the browser may lack the container's codec. Re-export to H.264 MP4; decodability and browser playback are separate checks.
- **Code changes do not appear on Windows:** restart Streamlit once after updating imported modules. Polling file watching is configured to avoid watchdog cross-drive errors when Python is installed on C: and the project is on D:.

## Project layout

`streamlit_app.py` owns the interface; `src/motionmirror/config.py` validates settings; `model.py` reconstructs/caches/downloads weights; `preprocessing.py` decodes and transforms video; `inference.py` computes results. `assets/styles.css` and `.streamlit/config.toml` define styling. `tests/` contains automated checks and `scripts/verify_video.py` measures a real video. Local-only `models/` and `references/` remain ignored.
