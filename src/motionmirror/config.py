"""Deployment settings and the verified training specification."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_DURATION_SECONDS = 30
MAX_PIXELS = 1920 * 1080
MAX_FRAMES = 3600
CLASS_MAPPING = {"walk": 0, "pushup": 1, "sit": 2, "drink": 3, "clap": 4}


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    class_to_index: dict
    num_frames: int
    frame_size: int
    temporal_window_seconds: float
    normalization_mean: tuple
    normalization_std: tuple

    @property
    def labels(self):
        return tuple(sorted(self.class_to_index, key=self.class_to_index.get))


def read_spec(path=ROOT / "model_manifest.json"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["class_to_index"] != CLASS_MAPPING:
        raise ValueError("The manifest does not match the trained five-class mapping.")
    return ModelSpec(**{key: tuple(data[key]) if key.startswith("normalization_") else data[key]
                        for key in ModelSpec.__dataclass_fields__})


def setting(name, default=""):
    """Environment takes precedence over optional Streamlit secrets."""
    if os.environ.get(name):
        return os.environ[name]
    import streamlit as st
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default
