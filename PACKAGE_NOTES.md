# MotionMirror website handoff

Selected checkpoint architecture: `mvit_v2_s`.

The `models/model.pth` checkpoint is inference-only (training optimizer state removed).
Class order and frame settings are recorded in `model_manifest.json`.
Read BOTH notebooks in `references/` before implementing inference.
Do not train a model, invent predictions, or change checkpoint label order.

Deployment must be free without subscriptions: Streamlit Community Cloud CPU.
If the selected model cannot fit free CPU/RAM limits, measure and document it; do not silently switch models or use paid infrastructure.
Do not upload `models/model.pth` or private notebooks to GitHub.
For deployment, publish the inference-only checkpoint to a free public HF Model repository or GitHub Release and download it once with caching.
