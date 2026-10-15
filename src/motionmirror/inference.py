"""Serialized CPU prediction with optional softmax averaging."""
from dataclasses import dataclass
import time

import torch

from .preprocessing import inspect_video, prepare_clips, VideoInfo


@dataclass(frozen=True)
class Prediction:
    action: str
    scores: tuple
    labels: tuple
    clips: int
    elapsed_seconds: float
    video: VideoInfo
    forward_seconds: float = 0.0


def predict(path, model, mode="Standard", *, video_info=None):
    start = time.perf_counter()
    # Serialize decoding and inference across visitors to bound peak working RAM.
    with model.lock, torch.inference_mode():
        info = video_info or inspect_video(path)
        scores = []
        forward_seconds = 0.0
        for clip in prepare_clips(path, info, model.spec, mode):
            forward_start = time.perf_counter()
            logits = model.network(clip)
            forward_seconds += time.perf_counter() - forward_start
            if logits.shape != (1, 5) or not torch.isfinite(logits).all():
                raise ValueError("The model returned invalid output scores.")
            scores.append(logits.float().softmax(dim=1)[0])
        average = torch.stack(scores).mean(dim=0)
    values = tuple(average.tolist())
    labels = model.spec.labels
    return Prediction(labels[int(average.argmax())], values, labels, len(scores),
                      time.perf_counter() - start, info, forward_seconds)


def predict_selected(path, mode="Standard"):
    """Keep cached-model acquisition and use atomic with comparison eviction."""
    from .model import MODEL_LIFECYCLE_LOCK, configured_model
    with MODEL_LIFECYCLE_LOCK:
        return predict(path, configured_model(), mode)
