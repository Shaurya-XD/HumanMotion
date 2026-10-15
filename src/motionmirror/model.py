"""Strict, safe checkpoint reconstruction; one shared resource per process."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import tempfile
import threading
import urllib.request
from urllib.parse import urlparse

import streamlit as st
import torch
from torch import nn
from torchvision.models import video

from .config import ROOT, ModelSpec, read_spec, setting, CLASS_MAPPING

# Holds the entire load/inference/release lifecycle, across Streamlit sessions.
MODEL_LIFECYCLE_LOCK = threading.RLock()

REGISTRY = {
    "r2plus1d_18": (video.r2plus1d_18, video.R2Plus1D_18_Weights, "fc"),
    "r3d_18": (video.r3d_18, video.R3D_18_Weights, "fc"),
    "mc3_18": (video.mc3_18, video.MC3_18_Weights, "fc"),
    "mvit_v2_s": (video.mvit_v2_s, video.MViT_V2_S_Weights, "head"),
}


def resolve_checkpoint(local_path, hf_repo="", hf_filename="model.pth", url="", revision="main"):
    local = Path(local_path).expanduser()
    if not local.is_absolute():
        local = ROOT / local
    if local.is_file():
        return local
    if hf_repo:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=hf_repo, filename=hf_filename, revision=revision))
    if url:
        if urlparse(url).scheme != "https":
            raise ValueError("Model download requires an HTTPS URL.")
        cache = Path(tempfile.gettempdir()) / "motionmirror-weights"
        cache.mkdir(exist_ok=True)
        destination = cache / (hashlib.sha256(url.encode()).hexdigest() + ".pth")
        if not destination.exists():
            part = None
            try:
                with urllib.request.urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
                    dir=cache, delete=False, suffix=".part"
                ) as output:
                    part = Path(output.name)
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 512 * 1024 * 1024:
                            raise ValueError("Model download exceeds the 512 MB safety limit.")
                        output.write(chunk)
                part.replace(destination)
            finally:
                if part:
                    part.unlink(missing_ok=True)
        return destination
    raise FileNotFoundError("Model weights are unavailable. Set MODEL_PATH or HF_MODEL_REPO in configuration.")


@dataclass
class LoadedModel:
    network: nn.Module
    spec: ModelSpec
    lock: threading.Lock
    parameter_mb: float


def spec_from_checkpoint(checkpoint, expected_architecture):
    """Use each checkpoint's settings; infer only notebook-verified missing metadata.

    The baseline notebook did not save architecture or normalization metadata.
    Its configured architecture and corresponding Kinetics preset supply those
    fields; strict state loading still verifies architecture compatibility.
    """
    if expected_architecture not in REGISTRY:
        raise ValueError("Unsupported comparison architecture.")
    if checkpoint.get("model_name", expected_architecture) != expected_architecture:
        raise ValueError("Checkpoint architecture does not match its configured model.")
    mapping = checkpoint.get("class_to_index")
    if (not isinstance(mapping, dict) or set(mapping) != set(CLASS_MAPPING)
            or any(type(v) is not int for v in mapping.values())
            or set(mapping.values()) != set(range(5))):
        raise ValueError("Checkpoint must contain a complete five-action class mapping.")
    required = ("num_frames", "frame_size", "temporal_window_seconds")
    if any(key not in checkpoint for key in required):
        raise ValueError("Checkpoint is missing its temporal or spatial training settings.")
    preset = REGISTRY[expected_architecture][1].KINETICS400_V1.transforms()
    return ModelSpec(expected_architecture, dict(mapping), checkpoint["num_frames"],
                     checkpoint["frame_size"], checkpoint["temporal_window_seconds"],
                     tuple(checkpoint.get("normalization_mean", preset.mean)),
                     tuple(checkpoint.get("normalization_std", preset.std)))


def load_checkpoint(path, spec=None, *, expected_architecture=None):
    # weights_only prevents loading arbitrary Python objects from pickle.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    spec = (spec_from_checkpoint(checkpoint, expected_architecture)
            if expected_architecture is not None else spec or read_spec())
    if spec.model_name not in REGISTRY:
        raise ValueError("Unsupported model architecture in the manifest.")
    for key in ModelSpec.__dataclass_fields__:
        if key in checkpoint:
            actual, expected = checkpoint[key], getattr(spec, key)
            if key.startswith("normalization_"):
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(f"Checkpoint and manifest disagree on {key}.")
    builder, weights, head = REGISTRY[spec.model_name]
    preset = weights.KINETICS400_V1.transforms()
    if (type(spec.num_frames) is not int or not 1 <= spec.num_frames <= 64
            or (spec.model_name == "mvit_v2_s" and spec.num_frames != 16)
            or not isinstance(spec.temporal_window_seconds, (int, float))
            or not math.isfinite(spec.temporal_window_seconds)
            or not 0 < spec.temporal_window_seconds <= 30
            or spec.frame_size != preset.crop_size[0]
            or spec.normalization_mean != tuple(preset.mean)
            or spec.normalization_std != tuple(preset.std)):
        raise ValueError("Model preprocessing settings do not match the verified evaluation pipeline.")
    torch.set_num_threads(max(1, min(4, int(setting("TORCH_NUM_THREADS", "2")))))
    # Build on meta and assign mmap tensors to avoid duplicate model allocations.
    with torch.device("meta"):
        network = builder(weights=None)
        original = getattr(network, head)
        features = original.in_features if head == "fc" else original[-1].in_features
        setattr(network, head, nn.Sequential(nn.Dropout(0.5), nn.Linear(features, 5)))
    network.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
    network.eval()
    network.requires_grad_(False)
    size = sum(p.numel() * p.element_size() for p in network.parameters()) / 1024**2
    return LoadedModel(network, spec, threading.Lock(), size)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_model(local_path, hf_repo="", hf_filename="model.pth", url="", revision="main", sha256=""):
    path = resolve_checkpoint(local_path, hf_repo, hf_filename, url, revision)
    if sha256:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest.lower() != sha256.lower():
            raise ValueError("Model checksum verification failed.")
    return load_checkpoint(path)


def configured_model():
    return get_model(setting("MODEL_PATH", "models/model.pth"), setting("HF_MODEL_REPO"),
                     setting("HF_MODEL_FILENAME", "model.pth"), setting("MODEL_URL"),
                     setting("HF_MODEL_REVISION", "main"), setting("MODEL_SHA256"))
