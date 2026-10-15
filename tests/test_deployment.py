"""Deployment boundaries; model inference remains real, only download transport is replaced."""
import pytest

from src.motionmirror import comparison, model
from src.motionmirror.config import ROOT


def test_hosted_comparison_requires_recorded_verification(monkeypatch, tmp_path):
    options = {"COMPARISON_MODELS_DIR": str(tmp_path),
               "MODEL_PATH": str(tmp_path / "model.pth"),
               "COMPARISON_HF_REPO": "owner/public-weights"}
    monkeypatch.setattr(comparison, "setting", lambda name, default="": options.get(name, default))
    assert not comparison.comparison_enabled()  # A remote source is insufficient.
    options["ENABLE_MODEL_COMPARISON"] = "true"
    assert not comparison.comparison_enabled()
    options["DEPLOYMENT_MODE"] = "cloud"
    options["HOSTED_COMPARISON_VERIFIED"] = "true"
    assert comparison.comparison_enabled()
    options["ENABLE_MODEL_COMPARISON"] = "false"
    assert not comparison.comparison_enabled()
    # Even a full local bundle cannot bypass explicit cloud policy.
    options.update(ENABLE_MODEL_COMPARISON="true", HOSTED_COMPARISON_VERIFIED="false")
    monkeypatch.setattr(comparison, "missing_checkpoints", lambda: ())
    assert not comparison.comparison_enabled()


def test_invalid_deployment_mode_is_rejected(monkeypatch):
    monkeypatch.setattr(comparison, "setting", lambda name, default="":
                        "typo" if name == "DEPLOYMENT_MODE" else default)
    with pytest.raises(comparison.ComparisonError, match="DEPLOYMENT_MODE"):
        comparison.comparison_enabled()


def test_selected_hf_configuration_checksum_and_cache(monkeypatch, tmp_path):
    import huggingface_hub
    options = {"MODEL_PATH": str(tmp_path / "absent.pth"),
               "HF_MODEL_REPO": "owner/public-weights", "HF_MODEL_FILENAME": "exports/model.pth",
               "HF_MODEL_REVISION": "recorded-commit",
               "MODEL_SHA256": "f8666134bbe97f0ce5dd292e0a9f9c983fccaa7a659aa3d9529b056fca678ccf"}
    calls = []
    def downloaded(**kwargs):
        calls.append(kwargs)
        return str(ROOT / "models/model.pth")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", downloaded)
    monkeypatch.setattr(model, "setting", lambda name, default="": options.get(name, default))
    model.get_model.clear()
    try:
        loaded = model.configured_model()
        assert model.configured_model() is loaded
        assert calls == [{"repo_id": "owner/public-weights", "filename": "exports/model.pth",
                          "revision": "recorded-commit"}]
        assert loaded.spec.labels == ("walk", "pushup", "sit", "drink", "clap")
        assert next(loaded.network.parameters()).device.type == "cpu"
        assert loaded.network.head[-1].out_features == 5
    finally:
        model.get_model.clear()


def test_cloud_ui_preserves_selected_analysis_and_actual_metrics(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DEPLOYMENT_MODE", "cloud")
    monkeypatch.setenv("ENABLE_MODEL_COMPARISON", "true")
    monkeypatch.setenv("HOSTED_COMPARISON_VERIFIED", "false")
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=15)
    assert not app.exception
    assert app.radio[0].value == "Standard"
    assert app.button[2].disabled
    markup = "\n".join(entry.value for entry in app.markdown)
    assert "Experimental model comparison" in markup
    assert "MViT-V2-S" in markup and "0.9046" in markup
    app.button[1].click().run()
    assert not app.exception and app.button[0].disabled
