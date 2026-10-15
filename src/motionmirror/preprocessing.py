"""Bounded sequential decoding reproducing notebook evaluation frame selection."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import math
import tempfile

import cv2
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from .config import (MAX_DURATION_SECONDS, MAX_FRAMES, MAX_PIXELS, MAX_UPLOAD_BYTES,
                     SUPPORTED_EXTENSIONS)
from .model import REGISTRY

cv2.setNumThreads(1)


class VideoError(ValueError):
    """A video issue that can be safely displayed to the visitor."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
    truncated: bool = False


@contextmanager
def uploaded_video(data, filename):
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise VideoError("Use an MP4, MOV, AVI, WebM, or MKV video.")
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise VideoError("Upload a non-empty video of 30 MB or less.")
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            path = Path(handle.name)
            handle.write(data)
        yield path
    finally:
        if path:
            path.unlink(missing_ok=True)


def inspect_video(path):
    """Count actually decodable frames, retaining no frames in memory.

    Deliberately use sequential decoding instead of codec-dependent seeking:
    this handles inaccurate frame counts and keyframe seek offsets correctly.
    Like the notebook, stop at the first undecodable frame.
    """
    path = Path(path)
    if not path.is_file():
        raise VideoError("The video file is missing. Please upload it again.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoError("Use an MP4, MOV, AVI, WebM, or MKV video.")
    if not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
        raise VideoError("Upload a non-empty video of 30 MB or less.")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("This video could not be opened. Try exporting it as an H.264 MP4.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 0 < fps <= 120:
            raise VideoError("The video has missing or unsupported FPS metadata (maximum 120 FPS).")
        reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if math.isfinite(reported) and reported / fps > MAX_DURATION_SECONDS + 1 / fps:
            raise VideoError("Use a video no longer than 30 seconds.")
        count, width, height = 0, 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w * h > MAX_PIXELS or min(w, h) < 16 or max(w, h) > 4096:
                raise VideoError("Use video up to 1920×1080 pixels, with each side at least 16 pixels.")
            if count and (w, h) != (width, height):
                raise VideoError("The video changes resolution. Re-export it with a fixed resolution.")
            width, height = w, h
            count += 1
            if count > MAX_FRAMES or count / fps > MAX_DURATION_SECONDS + 1e-6:
                raise VideoError("Use a video no longer than 30 seconds and 3,600 frames.")
        if not count:
            raise VideoError("No readable frames were found. Upload a different video.")
        return VideoInfo(fps, count, width, height, count / fps,
                         math.isfinite(reported) and reported > count + 1)
    finally:
        cap.release()


def temporal_indices(info, spec, mode="Standard"):
    if mode not in {"Standard", "Enhanced"}:
        raise ValueError("Unknown inference mode.")
    window = max(1, min(round(spec.temporal_window_seconds * info.fps), info.frame_count))
    last_start = info.frame_count - window
    starts = [last_start // 2] if mode == "Standard" else list(dict.fromkeys([0, last_start // 2, last_start]))
    return [torch.linspace(start, start + window - 1, spec.num_frames).round().long().tolist()
            for start in starts]


def spatial_frame(rgb, spec):
    # Float conversion BEFORE resize is essential: uint8 resize rounds differently.
    frame = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)
    preset = REGISTRY[spec.model_name][1].KINETICS400_V1.transforms()
    frame = TF.resize(frame, preset.resize_size, InterpolationMode.BILINEAR, antialias=True)
    frame = TF.center_crop(frame, preset.crop_size)
    return TF.normalize(frame, spec.normalization_mean, spec.normalization_std)


def prepare_clips(path, info, spec, mode="Standard"):
    windows = temporal_indices(info, spec, mode)
    wanted = {index for window in windows for index in window}
    frames = {}
    cap = cv2.VideoCapture(str(path))
    try:
        for index in range(max(wanted) + 1):
            ok, frame = cap.read()
            if not ok:
                raise VideoError("The video could not be decoded consistently. Please re-export it.")
            if index in wanted:
                frames[index] = spatial_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), spec)
    finally:
        cap.release()
    for window in windows:
        # B,C,T,H,W; only small, resized frames are retained.
        yield torch.stack([frames[i] for i in window], dim=1).unsqueeze(0)
