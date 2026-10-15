"""MotionMirror: a short-video workspace backed by the supplied trained model."""
import hashlib
import html
import logging

import streamlit as st

from src.motionmirror.config import ROOT, read_spec, MAX_UPLOAD_BYTES
from src.motionmirror.inference import predict_selected
from src.motionmirror.comparison_ui import render_comparison
from src.motionmirror.preprocessing import uploaded_video, VideoError

st.set_page_config(page_title="MotionMirror · Video action recognition", page_icon=":material/slow_motion_video:", layout="wide")
st.html(ROOT / "assets/styles.css")
spec = read_spec()


def clear_workspace():
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1
    for key in ("prediction", "input_signature", "analysis_error", "comparison", "comparison_error"):
        st.session_state.pop(key, None)


st.markdown('''<header class="app-brand">MotionMirror</header>
<div class="app-intro"><div class="app-title" role="heading" aria-level="1">Analyze human motion</div>
<p>Upload a video to identify one of five human actions.</p></div>''', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="medium")
with left, st.container(border=True, key="input_panel"):
    st.markdown('<div class="panel-title" role="heading" aria-level="2">Your video</div>', unsafe_allow_html=True)
    st.caption("Supports walking, push-ups, sitting, drinking, and clapping.")
    generation = st.session_state.get("upload_generation", 0)
    upload = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "webm", "mkv"],
                              key=f"video_{generation}", label_visibility="collapsed")
    st.caption("MP4, MOV, AVI, WebM or MKV · 30 MB · 30 seconds · up to 1080p")
    valid_size = upload is not None and 0 < upload.size <= MAX_UPLOAD_BYTES
    if upload is not None:
        if not valid_size:
            st.error("Upload a non-empty video of 30 MB or less.")
        else:
            st.video(upload)
            st.caption("Browser playback depends on the video codec. H.264 MP4 works best.")
    mode = st.radio("Inference mode", ["Standard", "Enhanced"], horizontal=True, key="mode")
    st.caption("Standard: one centered clip. Enhanced: up to three clips, averaged.")
    signature = (hashlib.sha256(upload.getbuffer()).hexdigest(), mode) if valid_size else None
    if signature != st.session_state.get("input_signature"):
        st.session_state.pop("prediction", None)
        st.session_state.pop("analysis_error", None)
        st.session_state.pop("comparison", None)
        st.session_state.pop("comparison_error", None)
        st.session_state.input_signature = signature
    analyze, reset = st.columns([3, 1])
    clicked = analyze.button("Analyze Video", type="primary", use_container_width=True, disabled=not valid_size)
    reset.button("Clear", on_click=clear_workspace, use_container_width=True)
    st.caption("Session-only processing. Temporary files are deleted after analysis.")
    if clicked:
        st.session_state.pop("prediction", None)
        try:
            with st.spinner("Extracting frames and analyzing motion on CPU…"):
                with uploaded_video(upload.getbuffer(), upload.name) as path:
                    st.session_state.prediction = predict_selected(path, mode)
            st.session_state.pop("analysis_error", None)
        except VideoError as exc:
            st.session_state.analysis_error = str(exc)
        except FileNotFoundError:
            st.session_state.analysis_error = "Model weights are unavailable. Configure MODEL_PATH or HF_MODEL_REPO and try again."
        except Exception:
            logging.exception("MotionMirror analysis failed")
            st.session_state.analysis_error = "Analysis could not finish. Check the model configuration, or try a shorter H.264 MP4 video."

with right, st.container(border=True, key="result_panel"):
    st.markdown('<div class="panel-title" role="heading" aria-level="2">Prediction</div>', unsafe_allow_html=True)
    result = st.session_state.get("prediction")
    error = st.session_state.get("analysis_error")
    if error:
        st.error(error)
    if result:
        winner = max(result.scores)
        st.markdown(f'''<div class="prediction-summary"><div><div class="result-label">Predicted action</div>
<div class="result-action">{html.escape(result.action.title())}</div></div><div class="top-score">
<div class="result-label">Top softmax score</div><div class="result-value">{winner:.1%}</div></div></div>''', unsafe_allow_html=True)
        st.markdown('<div class="score-heading">Action scores</div>', unsafe_allow_html=True)
        for label, score in sorted(zip(result.labels, result.scores), key=lambda item: item[1], reverse=True):
            safe_label = html.escape(label.title())
            st.markdown(f'''<div class="score-row"><div><span>{safe_label}</span><span>{score:.1%}</span></div>
<div class="score-track" role="meter" aria-label="{safe_label} softmax score" aria-valuemin="0"
aria-valuemax="100" aria-valuenow="{score * 100:.2f}"><div style="width:{score * 100:.2f}%"></div></div></div>''', unsafe_allow_html=True)
        st.caption("Relative model scores, not calibrated confidence.")
        st.divider()
        st.caption(f"MViT-V2-S · CPU · {result.clips} clip(s) · {result.elapsed_seconds:.2f}s processing")
        st.caption(f"{result.video.width}×{result.video.height} · {result.video.fps:.1f} FPS · {result.video.duration:.1f}s readable video")
        if result.video.truncated:
            st.warning("Only the readable prefix was analyzed; the video ended before its reported frame count.")
    else:
        st.markdown('''<div class="empty-result"><p>Results will appear here after analysis.</p></div>''', unsafe_allow_html=True)
        st.caption("MViT-V2-S · CPU inference")

with st.container(key="comparison_panel"):
    render_comparison(upload, valid_size)
