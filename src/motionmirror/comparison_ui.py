"""Compact comparison workspace using the site's existing visual language."""
import html
import logging

import streamlit as st

from .comparison import (MODEL_NAMES, ComparisonError, compare_all_models,
                         comparison_enabled, comparison_paths, experimental_metrics,
                         missing_checkpoints, remote_comparison_available)
from .config import CLASS_MAPPING
from .preprocessing import uploaded_video, VideoError


def comparison_rows(result):
    rows = []
    for item in result.models:
        prediction = item.prediction
        scores = dict(zip(prediction.labels, prediction.scores))
        row = {"Model": MODEL_NAMES[item.architecture], "Action": prediction.action.title(),
               "Top score (%)": max(prediction.scores) * 100,
               "Forward (s)": prediction.forward_seconds,
               "Processing (s)": prediction.elapsed_seconds}
        row.update({label.title() + " (%)": scores[label] * 100 for label in CLASS_MAPPING})
        rows.append(row)
    return rows


def table_html(rows, caption, formats=None, score_columns=()):
    """Four-row accessible table without loading a dataframe/chart runtime."""
    formats = formats or {}
    headers = list(rows[0])
    heading = "".join(f'<th scope="col">{html.escape(key)}</th>' for key in headers)
    body = []
    for row in rows:
        cells = []
        for key in headers:
            value = row[key]
            numeric = isinstance(value, (int, float))
            formatted = format(value, formats.get(key, ".4f")) if numeric else str(value)
            tint = (f' style="background:rgba(40,95,125,{.02 + .12 * value / 100:.3f})"'
                    if key in score_columns else "")
            cells.append(f'<td class="{"numeric" if numeric else "text"}"{tint}>{html.escape(formatted)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="comparison-table-wrap" tabindex="0" role="region" aria-label="{html.escape(caption)}">'
            f'<table class="comparison-table">'
            f'<caption>{html.escape(caption)}</caption><thead><tr>{heading}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def progress_message(done, total, name):
    return "Comparison complete" if done == total else f"Running {name} ({done + 1}/{total})"


def render_live_result(result):
    """Render the same scalar snapshot progressively and after all attempts."""
    attempts = len(result.models) + len(result.errors)
    if attempts == 4:
        st.success(f"Comparison complete · {len(result.models)}/4 predictions available")
    else:
        st.caption(f"{attempts}/4 models completed · results appear as each model finishes")
    for error in result.errors:
        st.error(f"{error.message} Attempt time: {error.elapsed_seconds:.2f}s. Other completed results are preserved.")
    if result.models:
        st.markdown('<div class="comparison-title" role="heading" aria-level="3">This video\'s predictions</div>', unsafe_allow_html=True)
        rows = comparison_rows(result)
        summary = [{"Model": row["Model"], "Predicted Action": row["Action"],
                    "Top Score (%)": row["Top score (%)"], "Inference Time (s)": row["Forward (s)"]}
                   for row in rows]
        st.markdown(table_html(summary, "Independent Standard single-clip predictions",
                    {"Top Score (%)": ".1f", "Inference Time (s)": ".3f"}), unsafe_allow_html=True)
        score_columns = [label.title() + " (%)" for label in CLASS_MAPPING]
        matrix = [{key: row[key] for key in ["Model", *score_columns]} for row in rows]
        st.markdown('<div class="comparison-title" role="heading" aria-level="3">Action score matrix</div>', unsafe_allow_html=True)
        st.markdown(table_html(matrix, "Softmax scores by model and action (%)",
                    {key: ".1f" for key in score_columns}, score_columns), unsafe_allow_html=True)
        for offset in range(0, len(result.models), 2):
            columns = st.columns(2, gap="medium")
            for column, item in zip(columns, result.models[offset:offset + 2]):
                with column:
                    name = html.escape(MODEL_NAMES[item.architecture])
                    prediction = item.prediction
                    markup = [f'<section class="model-scores"><header><strong>{name}</strong>'
                              f'<span>{html.escape(prediction.action.title())}</span></header>']
                    scores = dict(zip(prediction.labels, prediction.scores))
                    for label in CLASS_MAPPING:
                        score = scores[label]
                        markup.append(f'''<div class="score-row comparison-score"><div><span>{label.title()}</span>
<span>{score:.1%}</span></div><div class="score-track" role="meter" aria-label="{name} {label} softmax score"
aria-valuemin="0" aria-valuemax="100" aria-valuenow="{score * 100:.2f}">
<div style="width:{score * 100:.2f}%"></div></div></div>''')
                    markup.append(f'<p>{prediction.forward_seconds:.3f}s inference · '
                                  f'{prediction.elapsed_seconds:.3f}s clip processing</p></section>')
                    st.markdown("".join(markup), unsafe_allow_html=True)
        st.caption("Relative scores, not calibrated confidence. Predictions are independent.")
        if result.models[0].prediction.video.truncated:
            st.warning("The comparison uses only the video's readable prefix.")
    duration_label = "Total comparison" if attempts == 4 else "Elapsed so far"
    st.caption(f"{duration_label}: {result.total_seconds:.2f}s · "
               f"Sampled peak process RAM: {result.peak_rss_mb:.0f} MiB "
               f"(starting RAM: {result.baseline_rss_mb:.0f} MiB)")
    st.caption("Inference time excludes loading and frame extraction. Total includes processing and cleanup. "
               "RAM is sampled process usage; brief peaks may be missed.")


def render_comparison(upload, valid_size):
    with st.expander("Compare All Models", expanded=False):
        st.caption("Compare four trained models on the same video using Standard inference.")
        enabled = comparison_enabled()
        paths = comparison_paths()
        missing = missing_checkpoints(paths)
        remote = remote_comparison_available()
        if not enabled:
            if missing and not remote:
                st.info("Live comparison will be available when all four trained checkpoints are present.")
            else:
                st.info("Live comparison is disabled in this configuration. Local setup and hosted resource requirements are in the README.")
        if missing and not remote:
            st.caption("Required trained checkpoints not found: " +
                       ", ".join(paths[name].name for name in missing) + ". See README for local setup.")
        elif enabled:
            st.caption("Sequential CPU inference · one model at a time")
            if remote:
                st.caption("Missing local checkpoints are downloaded sequentially from the configured public model repository.")
        clicked = st.button("Compare All Models", disabled=not (enabled and valid_size and (not missing or remote)),
                            use_container_width=True, key="compare_all")
        result_slot = st.empty()
        if clicked:
            st.session_state.pop("comparison", None)
            st.session_state.pop("comparison_error", None)
            progress = st.progress(0, text="Preparing the comparison…")
            def update_result(snapshot):
                st.session_state.comparison = snapshot
                with result_slot.container():
                    render_live_result(snapshot)
            try:
                with uploaded_video(upload.getbuffer(), upload.name) as path:
                    st.session_state.comparison = compare_all_models(
                        path, paths,
                        lambda done, total, name: progress.progress(
                            done / total, text=progress_message(done, total, name)), update_result)
            except (VideoError, ComparisonError) as exc:
                st.session_state.comparison_error = str(exc)
            except Exception:
                logging.exception("MotionMirror comparison failed")
                st.session_state.comparison_error = "Comparison could not finish. Check the local checkpoints and video, then try again."
            finally:
                progress.empty()
        if st.session_state.get("comparison_error"):
            st.error(st.session_state.comparison_error)
        result = st.session_state.get("comparison")
        if result:
            with result_slot.container():
                render_live_result(result)
        st.divider()
        st.markdown('<div class="comparison-title" role="heading" aria-level="3">Experimental model comparison</div>', unsafe_allow_html=True)
        st.caption("HMDB51 Official Split 1 · supplied evaluation metrics, separate from this video's predictions.")
        try:
            st.markdown(table_html(experimental_metrics(), "Historical experimental metrics (scale 0–1)"),
                        unsafe_allow_html=True)
        except (OSError, ValueError, KeyError):
            logging.exception("Experimental comparison CSV could not be read")
            st.info("The experimental comparison data is unavailable.")
