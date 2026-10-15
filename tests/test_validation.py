from dataclasses import replace
import numpy as np
import pytest

from src.motionmirror.config import read_spec, ROOT, MAX_UPLOAD_BYTES
from src.motionmirror.model import load_checkpoint, resolve_checkpoint, get_model
from src.motionmirror.preprocessing import inspect_video, uploaded_video, VideoError


def test_checkpoint_rejects_wrong_metadata():
    with pytest.raises(ValueError, match="disagree"):
        load_checkpoint(ROOT / "models/model.pth", replace(read_spec(), num_frames=8))


def test_resolution_priority_and_missing_weights(tmp_path):
    local = tmp_path / "local.pth"
    local.write_bytes(b"file presence check only")
    assert resolve_checkpoint(local, hf_repo="not-used") == local
    with pytest.raises(FileNotFoundError, match="unavailable"):
        resolve_checkpoint(tmp_path / "missing.pth")
    with pytest.raises(ValueError, match="HTTPS"):
        resolve_checkpoint(tmp_path / "missing.pth", url="http://example.invalid/model.pth")


def test_checksum_rejection():
    with pytest.raises(ValueError, match="checksum"):
        get_model(str(ROOT / "models/model.pth"), sha256="0" * 64)


def test_upload_limits():
    for data in (b"", b"x" * (MAX_UPLOAD_BYTES + 1)):
        with pytest.raises(VideoError, match="30 MB"):
            with uploaded_video(data, "clip.mp4"):
                pass


@pytest.mark.parametrize("fps,frames,shape,message", [
    (float('nan'), 1, (64, 80), "FPS"),
    (10, 301, (64, 80), "30 seconds"),
    (10, 1, (2160, 3840), "1920"),
])
def test_invalid_metadata_is_rejected(monkeypatch, tmp_path, fps, frames, shape, message):
    # Mock video metadata only, never model outputs.
    import cv2
    class Capture:
        released = False
        def isOpened(self): return True
        def get(self, field): return fps if field == cv2.CAP_PROP_FPS else frames
        def read(self): return True, np.zeros((*shape, 3), np.uint8)
        def release(self): self.released = True
    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    path = tmp_path / "test.mp4"
    path.write_bytes(b"metadata fixture")
    with pytest.raises(VideoError, match=message):
        inspect_video(path)
    assert capture.released
