from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest
import torch
from torchvision.transforms import functional as TF, InterpolationMode

from src.motionmirror.config import read_spec, CLASS_MAPPING, ROOT
from src.motionmirror.model import get_model, REGISTRY
from src.motionmirror.preprocessing import (VideoInfo, VideoError, temporal_indices,
    inspect_video, prepare_clips, uploaded_video)
from src.motionmirror.inference import predict


@pytest.fixture(scope="module")
def spec():
    return read_spec()


@pytest.fixture(scope="module")
def model():
    return get_model(str(ROOT / "models/model.pth"))


@pytest.fixture
def video(tmp_path):
    # Codec fixture only; this does NOT constitute action recognition validation.
    path = tmp_path / "codec.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 64))
    assert writer.isOpened()
    for i in range(50):
        frame = np.zeros((64, 80, 3), np.uint8)
        frame[:] = (i * 4, 80, 170)
        writer.write(frame)
    writer.release()
    return path


def test_sampling_matches_notebook(spec):
    info = VideoInfo(25, 126, 100, 100, 5.04)
    expected = torch.linspace(38, 87, 16).round().long().tolist()
    assert temporal_indices(info, spec) == [expected]
    short = VideoInfo(25, 10, 100, 100, .4)
    assert len(temporal_indices(short, spec, "Enhanced")) == 1


def test_preprocessing_matches_notebook(video, spec):
    info = inspect_video(video)
    clip = next(prepare_clips(video, info, spec))
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255
    batch = batch[temporal_indices(info, spec)[0]]
    preset = REGISTRY[spec.model_name][1].KINETICS400_V1.transforms()
    batch = TF.resize(batch, preset.resize_size, InterpolationMode.BILINEAR, antialias=True)
    batch = TF.center_crop(batch, preset.crop_size)
    batch = TF.normalize(batch, spec.normalization_mean, spec.normalization_std)
    expected = batch.permute(1, 0, 2, 3).unsqueeze(0)
    assert clip.shape == (1, 3, 16, 224, 224)
    torch.testing.assert_close(clip, expected, rtol=0, atol=0)


def test_invalid_video_and_cleanup(tmp_path):
    with uploaded_video(b"not a video", "bad.mp4") as path:
        with pytest.raises(VideoError, match="opened|readable"):
            inspect_video(path)
    assert not path.exists()
    with pytest.raises(VideoError):
        with uploaded_video(b"x", "bad.exe"):
            pass
    with pytest.raises(RuntimeError):
        with uploaded_video(b"x", "clip.mp4") as path:
            raise RuntimeError("test cleanup after failure")
    assert not path.exists()


def test_model_load_cache_and_logits(model, spec):
    assert model.spec.class_to_index == CLASS_MAPPING
    assert model is get_model(str(ROOT / "models/model.pth"))
    assert not model.network.training
    with torch.inference_mode():
        logits = model.network(torch.zeros(1, 3, 16, 224, 224))
    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()
    assert logits.softmax(1).sum().item() == pytest.approx(1, abs=1e-6)


def test_real_model_determinism_and_enhanced_average(video, model):
    first, second = predict(video, model), predict(video, model)
    assert first.scores == second.scores
    info = inspect_video(video)
    with torch.inference_mode():
        expected = torch.stack([model.network(c).softmax(1)[0]
                               for c in prepare_clips(video, info, model.spec, "Enhanced")]).mean(0)
    result = predict(video, model, "Enhanced")
    assert result.clips == 3
    torch.testing.assert_close(torch.tensor(result.scores), expected)


@pytest.mark.parametrize("shape", [(64, 80), (96, 32), (32, 96)])
def test_short_portrait_landscape(tmp_path, spec, shape):
    h, w = shape
    path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    writer.write(np.full((h, w, 3), 127, np.uint8))
    writer.release()
    info = inspect_video(path)
    clips = list(prepare_clips(path, info, spec, "Enhanced"))
    assert len(clips) == 1
    assert clips[0].shape == (1, 3, 16, 224, 224)


def test_app_start_clear_and_disabled_analyze():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
    assert not app.exception
    assert app.button[0].disabled
    app.session_state["prediction"] = "stale"
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["upload_generation"] == 1
    assert "prediction" not in app.session_state
