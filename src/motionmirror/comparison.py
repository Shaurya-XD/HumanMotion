"""Optional local sequential comparison; no collection of resident networks."""
import csv
from dataclasses import dataclass
import gc
import logging
from pathlib import Path
import threading
import time

import psutil

from .config import ROOT, setting
from .inference import Prediction, predict
from .model import MODEL_LIFECYCLE_LOCK, get_model, load_checkpoint, resolve_checkpoint
from .preprocessing import inspect_video

MODEL_NAMES = {
    "r2plus1d_18": "R(2+1)D-18",
    "r3d_18": "R3D-18",
    "mc3_18": "MC3-18",
    "mvit_v2_s": "MViT-V2-S",
}


class ComparisonError(ValueError):
    """A readable comparison setup failure."""


def comparison_enabled():
    enabled = setting("ENABLE_MODEL_COMPARISON", "auto").strip().lower()
    if enabled not in {"auto", "true", "false"}:
        raise ComparisonError("ENABLE_MODEL_COMPARISON must be auto, true, or false.")
    deployment = setting("DEPLOYMENT_MODE", "auto").strip().lower()
    if deployment not in {"auto", "local", "cloud"}:
        raise ComparisonError("DEPLOYMENT_MODE must be auto, local, or cloud.")
    local_files = not missing_checkpoints()
    # A public file source alone is not evidence of hosted resource capacity.
    # Auto recognizes the supplied local bundle; cloud deployments set cloud explicitly.
    hosted = deployment == "cloud" or (deployment == "auto" and not local_files)
    if hosted:
        return (enabled == "true" and
                setting("HOSTED_COMPARISON_VERIFIED", "false").strip().lower() == "true")
    return enabled == "true" or (enabled == "auto" and (local_files or remote_comparison_available()))


def remote_comparison_available():
    """Configured public files are fetched lazily; presence is verified on download."""
    return bool(setting("COMPARISON_HF_REPO"))


def comparison_paths():
    directory = Path(setting("COMPARISON_MODELS_DIR", "models")).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    paths = {name: directory / f"{name}.pth" for name in MODEL_NAMES}
    # Preserve the existing selected checkpoint; do not rename or overwrite it.
    selected = Path(setting("MODEL_PATH", "models/model.pth")).expanduser()
    paths["mvit_v2_s"] = selected if selected.is_absolute() else ROOT / selected
    return paths


def missing_checkpoints(paths=None):
    paths = comparison_paths() if paths is None else paths
    return tuple(name for name in MODEL_NAMES if name not in paths or not Path(paths[name]).is_file())


@dataclass(frozen=True)
class ModelComparison:
    architecture: str
    prediction: Prediction
    load_seconds: float


@dataclass(frozen=True)
class ModelFailure:
    architecture: str
    message: str
    elapsed_seconds: float


@dataclass(frozen=True)
class ComparisonResult:
    models: tuple[ModelComparison, ...]
    total_seconds: float
    peak_rss_mb: float
    baseline_rss_mb: float
    errors: tuple[ModelFailure, ...] = ()


class MemorySampler:
    """Process RSS sampled during validation, model loading and all forwards.

    Includes Streamlit/runtime/session memory. CPU allocators may retain freed
    pages. It is not an exact allocator peak and cannot guarantee host capacity.
    """
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(.01):
            self.observe()

    def observe(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.observe()


def _compare_one(path, checkpoint, architecture, info):
    """Uncached network is released before the next architecture is loaded."""
    model = None
    try:
        started = time.perf_counter()
        if not Path(checkpoint).is_file() and remote_comparison_available():
            filename = (setting("HF_MODEL_FILENAME", "model.pth") if architecture == "mvit_v2_s"
                        else f"{architecture}.pth")
            checkpoint = resolve_checkpoint(checkpoint, hf_repo=setting("COMPARISON_HF_REPO"),
                hf_filename=filename, revision=setting("COMPARISON_HF_REVISION", "main"))
        model = load_checkpoint(checkpoint, expected_architecture=architecture)
        load_seconds = time.perf_counter() - started
        result = predict(path, model, "Standard", video_info=info)
        return ModelComparison(architecture, result, load_seconds)
    finally:
        del model
        gc.collect()


def _run_sequence(path, paths, progress=None, on_result=None):
    """Shared measured engine; accepts subsets for diagnostics/tests only.

    The public Compare All Models entry point requires all four checkpoints.
    This function never puts LoadedModel or Torch tensors in result storage.
    """
    with MODEL_LIFECYCLE_LOCK:
        started = time.perf_counter()
        with MemorySampler() as memory:
            # Clearing resources alone does not release caller-held references.
            # All UI inference is inside this same lifecycle lock, and sessions
            # store only scalar Prediction records, never model objects.
            get_model.clear()
            gc.collect()
            info = inspect_video(path)
            results = []
            errors = []
            for index, (architecture, checkpoint) in enumerate(paths.items()):
                if progress:
                    progress(index, len(paths), MODEL_NAMES[architecture])
                attempt_started = time.perf_counter()
                try:
                    results.append(_compare_one(path, checkpoint, architecture, info))
                except Exception as exc:
                    logging.error("Comparison failed for %s: %s", architecture, str(exc)[:500])
                    detail = ("Checkpoint file was not found." if isinstance(exc, FileNotFoundError)
                              else "Check its trained checkpoint, preprocessing metadata, and local logs.")
                    errors.append(ModelFailure(architecture,
                        f"{MODEL_NAMES[architecture]} failed. {detail}",
                        time.perf_counter() - attempt_started))
                    # Do not carry exception tracebacks (which own model/tensor references)
                    # into the next architecture. Only the readable scalar record survives.
                    exc.__traceback__ = None
                gc.collect()
                memory.observe()
                if on_result:
                    on_result(ComparisonResult(tuple(results), time.perf_counter() - started,
                              memory.peak / 1024**2, memory.baseline / 1024**2, tuple(errors)))
            if progress:
                progress(len(paths), len(paths), "Complete")
        return ComparisonResult(tuple(results), time.perf_counter() - started,
                                memory.peak / 1024**2, memory.baseline / 1024**2, tuple(errors))


def compare_all_models(path, paths=None, progress=None, on_result=None):
    """Attempt four Standard predictions; retain successes and individual failures."""
    paths = comparison_paths() if paths is None else paths
    if set(paths) != set(MODEL_NAMES):
        raise ComparisonError("Configure exactly the four supported comparison architectures.")
    missing = missing_checkpoints(paths)
    if len(missing) == len(MODEL_NAMES) and not remote_comparison_available():
        files = ", ".join(Path(paths[name]).name for name in missing)
        raise ComparisonError(f"Comparison requires the missing trained checkpoints: {files}.")
    ordered = {name: Path(paths[name]) for name in MODEL_NAMES}
    return _run_sequence(path, ordered, progress, on_result)


def experimental_metrics(path=ROOT / "data/experimental_model_comparison.csv"):
    """Public-safe source CSV, never use historical metrics as upload scores."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or {row["model"] for row in rows} != set(MODEL_NAMES):
        raise ValueError("Experimental comparison must contain the four verified architectures.")
    by_model = {row["model"]: row for row in rows}
    output = []
    for name in MODEL_NAMES:
        row = by_model[name]
        metrics = {key: float(row[key]) for key in ("test_accuracy", "test_macro_f1", "val_macro_f1")}
        if not all(0 <= value <= 1 for value in metrics.values()):
            raise ValueError("Experimental scores must be finite values between zero and one.")
        output.append({"Model": MODEL_NAMES[name], "Test accuracy": metrics["test_accuracy"],
                       "Test macro F1": metrics["test_macro_f1"],
                       "Validation macro F1": metrics["val_macro_f1"]})
    return output
