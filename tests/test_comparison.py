"""Comparison uses genuine weights when available; metadata fixtures are not models."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
import threading
import weakref

import cv2
import numpy as np
import pytest
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from src.motionmirror import comparison
from src.motionmirror.comparison import (MODEL_NAMES, ComparisonError, MemorySampler,
    compare_all_models, comparison_paths, comparison_enabled, experimental_metrics,
    missing_checkpoints, _run_sequence)
from src.motionmirror.comparison_ui import comparison_rows, progress_message, table_html
from src.motionmirror.config import ROOT, CLASS_MAPPING
from src.motionmirror.model import (REGISTRY, MODEL_LIFECYCLE_LOCK, get_model,
    configured_model, load_checkpoint, spec_from_checkpoint)
from src.motionmirror.inference import predict_selected
from src.motionmirror.preprocessing import inspect_video, prepare_clips


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "comparison-fixture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(24):
        writer.write(np.full((48, 64, 3), (i * 8, 80, 160), np.uint8))
    writer.release()
    return path


def metadata(name):
    preset = REGISTRY[name][1].KINETICS400_V1.transforms()
    return {"model_name": name, "class_to_index": dict(CLASS_MAPPING),
            "num_frames": 16, "frame_size": preset.crop_size[0],
            "temporal_window_seconds": 2.0,
            "normalization_mean": preset.mean, "normalization_std": preset.std}


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_each_architectures_preprocessing_matches_notebook(video, name):
    spec = spec_from_checkpoint(metadata(name), name)
    preset = REGISTRY[name][1].KINETICS400_V1.transforms()
    assert spec.frame_size == (224 if name == "mvit_v2_s" else 112)
    assert spec.normalization_mean == tuple(preset.mean)
    assert spec.normalization_std == tuple(preset.std)
    info = inspect_video(video)
    actual = next(prepare_clips(video, info, spec, "Standard"))
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
    # Notebook centered two-second window and rounding, independent of production sampling.
    window = min(round(info.fps * 2), len(frames))
    start = (len(frames) - window) // 2
    indices = torch.linspace(start, start + window - 1, 16).round().long()
    frames = TF.resize(frames[indices], preset.resize_size, InterpolationMode.BILINEAR, antialias=True)
    frames = TF.center_crop(frames, preset.crop_size)
    frames = TF.normalize(frames, preset.mean, preset.std)
    torch.testing.assert_close(actual, frames.permute(1, 0, 2, 3).unsqueeze(0), rtol=0, atol=0)


def test_checkpoint_class_order_and_baseline_metadata():
    saved = metadata("r2plus1d_18")
    saved.pop("model_name")
    saved.pop("normalization_mean")
    saved.pop("normalization_std")
    saved["class_to_index"] = {"clap": 0, "drink": 1, "sit": 2, "pushup": 3, "walk": 4}
    spec = spec_from_checkpoint(saved, "r2plus1d_18")
    assert spec.labels == ("clap", "drink", "sit", "pushup", "walk")
    assert spec.normalization_mean == tuple(REGISTRY["r2plus1d_18"][1].KINETICS400_V1.transforms().mean)
    saved["num_frames"] = 24
    saved["temporal_window_seconds"] = 1.5
    assert spec_from_checkpoint(saved, "r2plus1d_18").num_frames == 24
    assert spec_from_checkpoint(saved, "r2plus1d_18").temporal_window_seconds == 1.5


@pytest.mark.parametrize("change", [
    {"model_name": "r3d_18"}, {"class_to_index": {"walk": 0}},
    {"class_to_index": {**CLASS_MAPPING, "run": 5}},
    {"class_to_index": {**CLASS_MAPPING, "clap": 3}},
])
def test_incompatible_checkpoint_metadata_rejected(change):
    saved = metadata("mvit_v2_s")
    saved.update(change)
    with pytest.raises(ValueError):
        spec_from_checkpoint(saved, "mvit_v2_s")
    saved = metadata("mvit_v2_s")
    saved.pop("num_frames")
    with pytest.raises(ValueError, match="missing"):
        spec_from_checkpoint(saved, "mvit_v2_s")


def test_missing_checkpoint_preflight_does_not_evict_cache(monkeypatch, video, tmp_path):
    paths = {name: tmp_path / f"{name}.pth" for name in MODEL_NAMES}
    def should_not_run(*_):
        pytest.fail("Missing checkpoint preflight must not start loading or evict resources")
    monkeypatch.setattr(comparison, "_run_sequence", should_not_run)
    with pytest.raises(ComparisonError, match="missing trained checkpoints"):
        compare_all_models(video, paths)
    with pytest.raises(ComparisonError, match="exactly"):
        compare_all_models(video, {"mvit_v2_s": ROOT / "models/model.pth"})


def test_local_auto_mode_and_paths_preserve_mvit(monkeypatch):
    monkeypatch.setenv("ENABLE_MODEL_COMPARISON", "false")
    assert not comparison_enabled()
    monkeypatch.setenv("ENABLE_MODEL_COMPARISON", "true")
    assert comparison_enabled()
    monkeypatch.setenv("MODEL_PATH", "models/model.pth")
    assert comparison_paths()["mvit_v2_s"] == ROOT / "models/model.pth"
    assert comparison_paths()["r3d_18"].name == "r3d_18.pth"
    monkeypatch.setenv("ENABLE_MODEL_COMPARISON", "auto")
    assert comparison_enabled() == (not missing_checkpoints())
    monkeypatch.setenv("ENABLE_MODEL_COMPARISON", "invalid")
    with pytest.raises(ComparisonError):
        comparison_enabled()


def test_real_mvit_lifecycle_metrics_and_standard_mode(video, monkeypatch):
    get_model.clear()
    cached = get_model(str(ROOT / "models/model.pth"))
    cached_ref = weakref.ref(cached)
    del cached
    loaded_refs = []
    def track_load(path, **kwargs):
        assert cached_ref() is None, "The selected model must be evicted before loading comparison"
        assert all(ref() is None for ref in loaded_refs)
        loaded = load_checkpoint(path, **kwargs)
        loaded_refs.extend([weakref.ref(loaded), weakref.ref(loaded.network)])
        return loaded
    monkeypatch.setattr(comparison, "load_checkpoint", track_load)
    busy_checks = []
    def progress(*_):
        def check_lock():
            acquired = MODEL_LIFECYCLE_LOCK.acquire(blocking=False)
            busy_checks.append(not acquired)
            if acquired:
                MODEL_LIFECYCLE_LOCK.release()
        thread = threading.Thread(target=check_lock)
        thread.start()
        thread.join()
    result = _run_sequence(video, {"mvit_v2_s": ROOT / "models/model.pth"}, progress)
    assert all(ref() is None for ref in loaded_refs)
    assert all(busy_checks)
    prediction = result.models[0].prediction
    assert prediction.clips == 1
    assert sum(prediction.scores) == pytest.approx(1, abs=1e-6)
    assert prediction.forward_seconds > 0
    assert prediction.elapsed_seconds >= prediction.forward_seconds
    assert result.total_seconds >= prediction.elapsed_seconds
    assert result.peak_rss_mb >= result.baseline_rss_mb > 0
    row = comparison_rows(result)[0]
    for label, score in zip(prediction.labels, prediction.scores):
        assert row[label.title() + " (%)"] == pytest.approx(score * 100)
    # Save actual measurements, clearly identified as a ONE-model diagnostic.
    (ROOT / "test-results").mkdir(exist_ok=True)
    (ROOT / "test-results/comparison-diagnostic.json").write_text(
        json.dumps({"scope": "one-model diagnostic, generated codec fixture, NOT four-model comparison",
                    **asdict(result)}, indent=2), encoding="utf-8")
    # Standard remains available after comparison and is cached again normally.
    restored = configured_model()
    assert restored is configured_model()
    selected = predict_selected(video)
    assert selected.scores == prediction.scores
    get_model.clear()
    del restored
    gc.collect()


def test_model_released_before_next_load_and_after_failure(video, monkeypatch):
    references = []
    def track_load(path, **kwargs):
        assert all(ref() is None for ref in references)
        model = load_checkpoint(path, **kwargs)
        references.extend([weakref.ref(model), weakref.ref(model.network)])
        return model
    monkeypatch.setattr(comparison, "load_checkpoint", track_load)
    # Real first prediction, then intentionally incompatible architecture; no invented outputs.
    snapshots = []
    result = _run_sequence(video, {"mvit_v2_s": ROOT / "models/model.pth",
                                  "r3d_18": ROOT / "models/model.pth",
                                  "mc3_18": ROOT / "models/mc3_18.pth"}, on_result=snapshots.append)
    assert [item.architecture for item in result.models] == ["mvit_v2_s", "mc3_18"]
    assert len(result.errors) == 1 and result.errors[0].architecture == "r3d_18"
    assert "R3D-18 failed" in result.errors[0].message
    assert len(snapshots[0].models) == 1
    assert snapshots[1].models == snapshots[0].models
    assert len(snapshots[1].errors) == 1
    assert len(snapshots[2].models) == 2
    gc.collect()
    assert all(ref() is None for ref in references)


def test_memory_sampler_stops_on_error():
    sampler = MemorySampler()
    with pytest.raises(RuntimeError):
        with sampler:
            sampler.observe()
            raise RuntimeError("cleanup test")
    assert not sampler.thread.is_alive()
    assert sampler.peak >= sampler.baseline > 0


def test_experimental_csv_is_exact_supplied_data():
    assert (ROOT / "data/experimental_model_comparison.csv").read_bytes() == (
        ROOT / "references/four_model_comparison.csv").read_bytes()
    rows = experimental_metrics()
    assert len(rows) == 4
    assert rows[3]["Test accuracy"] == pytest.approx(.9066666666666666)
    assert rows[0]["Test macro F1"] == pytest.approx(.884074014221073)


def test_comparison_ui_clear_and_safe_default(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("ENABLE_MODEL_COMPARISON", "false")
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not app.exception
    assert app.radio[0].value == "Standard"
    assert app.expander[0].label == "Compare All Models"
    assert app.button[2].label == "Compare All Models" and app.button[2].disabled
    assert any("Historical experimental metrics" in entry.value and "0.9067" in entry.value
               for entry in app.markdown)
    app.session_state["comparison_error"] = "stale"
    app.button[1].click().run()
    assert not app.exception
    assert "comparison_error" not in app.session_state


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_available_trained_checkpoints_load_and_predict(video, name):
    path = comparison_paths()[name]
    if not path.is_file():
        pytest.skip(f"Actual trained checkpoint not supplied: {path.name}")
    item = comparison._compare_one(video, path, name, inspect_video(video))
    assert item.prediction.clips == 1
    assert set(item.prediction.labels) == set(CLASS_MAPPING)
    assert np.isfinite(item.prediction.scores).all()
    assert sum(item.prediction.scores) == pytest.approx(1, abs=1e-6)


def test_full_comparison_with_actual_four_checkpoints(video):
    paths = comparison_paths()
    if missing_checkpoints(paths):
        pytest.skip("Full comparison needs the three missing trained CNN checkpoints")
    snapshots, messages = [], []
    original_load = comparison.load_checkpoint
    references = []
    names = []
    cached = configured_model()
    references.append(weakref.ref(cached.network))
    del cached
    def checked_load(path, **kwargs):
        assert all(ref() is None for ref in references)
        model = original_load(path, **kwargs)
        assert model.spec.class_to_index == CLASS_MAPPING
        assert model.spec.num_frames == 16
        assert next(model.network.parameters()).device.type == "cpu"
        head = model.network.head if model.spec.model_name == "mvit_v2_s" else model.network.fc
        assert head[-1].out_features == 5
        references.append(weakref.ref(model.network))
        names.append(kwargs["expected_architecture"])
        return model
    # Patch only lifecycle observation; every architecture and prediction is real.
    from unittest.mock import patch
    with patch.object(comparison, "load_checkpoint", checked_load):
        result = compare_all_models(video, paths,
            lambda done, total, name: messages.append(progress_message(done, total, name)), snapshots.append)
    assert len(result.models) == 4
    assert not result.errors
    assert names == list(MODEL_NAMES)
    assert all(ref() is None for ref in references)
    assert [item.architecture for item in result.models] == list(MODEL_NAMES)
    assert all(item.prediction.clips == 1 for item in result.models)
    assert [len(snapshot.models) for snapshot in snapshots] == [1, 2, 3, 4]
    assert messages == ["Running R(2+1)D-18 (1/4)", "Running R3D-18 (2/4)",
                        "Running MC3-18 (3/4)", "Running MViT-V2-S (4/4)", "Comparison complete"]
    assert all(sum(item.prediction.scores) == pytest.approx(1, abs=1e-6) for item in result.models)
    (ROOT / "test-results/full-comparison-test.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def test_partial_failure_when_one_file_is_missing(video, tmp_path):
    paths = comparison_paths()
    paths["r3d_18"] = tmp_path / "missing.pth"
    result = compare_all_models(video, paths)
    assert [item.architecture for item in result.models] == ["r2plus1d_18", "mc3_18", "mvit_v2_s"]
    assert len(result.errors) == 1 and result.errors[0].architecture == "r3d_18"
    assert "not found" in result.errors[0].message


def test_live_results_render_actual_scores_and_errors(video):
    from streamlit.testing.v1 import AppTest
    result = _run_sequence(video, {"mvit_v2_s": ROOT / "models/model.pth",
                                   "r3d_18": ROOT / "models/model.pth"})
    # Test the renderer directly with actual predictions and an actual load failure.
    app = AppTest.from_string('''
import streamlit as st
from src.motionmirror.comparison_ui import render_live_result
render_live_result(st.session_state.result)
''')
    app.session_state["result"] = result
    app.run(timeout=10)
    assert not app.exception
    assert any("R3D-18 failed" in entry.value for entry in app.error)
    html = "\n".join(entry.value for entry in app.markdown)
    assert "Predicted Action" in html and "Inference Time (s)" in html
    assert "Action score matrix" in html and "role=\"meter\"" in html
    assert "MViT-V2-S" in html
    for score in result.models[0].prediction.scores:
        assert f"{score:.1%}" in html


def test_auto_enables_local_files_and_configured_remote_source(monkeypatch, tmp_path):
    options = {"DEPLOYMENT_MODE": "local"}
    monkeypatch.setattr(comparison, "setting", lambda name, default="": options.get(name, default))
    assert comparison_enabled()  # All four actual local checkpoints are now present.
    options["COMPARISON_MODELS_DIR"] = str(tmp_path)
    options["MODEL_PATH"] = str(tmp_path / "model.pth")
    assert not comparison_enabled()
    options["COMPARISON_HF_REPO"] = "test/public-files"
    assert comparison_enabled()
    options["ENABLE_MODEL_COMPARISON"] = "false"
    assert not comparison_enabled()


def test_remote_download_configuration_uses_real_cached_checkpoint(video, monkeypatch, tmp_path):
    options = {"COMPARISON_HF_REPO": "test/public-files", "COMPARISON_HF_REVISION": "test-commit"}
    monkeypatch.setattr(comparison, "setting", lambda name, default="": options.get(name, default))
    calls = []
    def cached_download(path, **kwargs):
        calls.append(kwargs)
        return ROOT / "models/model.pth"
    # Only transport is substituted; strict loading and inference use actual MViT weights.
    monkeypatch.setattr(comparison, "resolve_checkpoint", cached_download)
    result = comparison._compare_one(video, tmp_path / "missing.pth", "mvit_v2_s", inspect_video(video))
    assert calls == [{"hf_repo": "test/public-files", "hf_filename": "model.pth", "revision": "test-commit"}]
    assert result.prediction.clips == 1
    assert sum(result.prediction.scores) == pytest.approx(1, abs=1e-6)
