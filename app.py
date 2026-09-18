"""
app.py — SatQuery AI: Interactive Vision-Language Assistant for
Multimodal Remote Sensing Image Analysis

Run locally:   streamlit run app.py
Run on Colab:  see notebooks/SatQueryAI_Colab.ipynb
"""

import os

import streamlit as st
from PIL import Image

from utils import load_image, draw_detections, draw_evidence_regions, tile_image
from vision_pipeline import SatQueryVisionPipeline
from llm_reasoner import SatQueryReasoner
from change_detection import build_change_context
from translate import LANGUAGES, to_english, from_english
from report_generator import generate_report

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")


# ---------------- Cached resources ----------------
@st.cache_resource
def get_vision_pipeline():
    return SatQueryVisionPipeline()


@st.cache_resource
def get_reasoner(api_key):
    return SatQueryReasoner(anthropic_api_key=api_key)


def _init_state():
    defaults = {
        "chat_history": [],
        "last_context": None,
        "last_result": None,
        "last_image": None,
        "last_before_image": None,
        "last_after_image": None,
        "last_error": None,
        "analysis_mode": None,
        "source_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_analysis():
    st.session_state.chat_history = []
    st.session_state.last_context = None
    st.session_state.last_result = None
    st.session_state.last_image = None
    st.session_state.last_before_image = None
    st.session_state.last_after_image = None
    st.session_state.last_error = None


def _set_source(source_key, image=None, before=None, after=None):
    if st.session_state.source_key != source_key:
        _reset_analysis()
        st.session_state.source_key = source_key
    if image is not None:
        st.session_state.last_image = image
    if before is not None:
        st.session_state.last_before_image = before
    if after is not None:
        st.session_state.last_after_image = after


def _confidence(context):
    detections = context.get("detections", [])
    if detections:
        average = sum(item.get("conf", 0.0) for item in detections) / len(detections)
        level = "HIGH" if average >= 0.75 else "MEDIUM" if average >= 0.5 else "LOW"
        return {"level": level, "basis": f"Mean object-detector confidence: {average:.2f}."}

    land_cover = context.get("land_cover", [])
    if land_cover and "score" in land_cover[0]:
        score = land_cover[0]["score"]
        level = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.5 else "LOW"
        return {"level": level, "basis": f"Top land-cover model score: {score:.2f}."}

    if "change_pct" in context and context.get("num_change_regions", 0) > 0:
        return {"level": "MEDIUM", "basis": "Change regions were measured from the SSIM difference mask."}

    return {"level": "NOT ESTIMATED", "basis": "The available analysis did not expose a calibrated confidence score."}


def build_analysis_result(query, answer, context, mode, trace):
    findings = []
    measurements = {}
    evidence = []

    if "change_pct" in context:
        findings.append(f"Changed area measured: {context['change_pct']}% of the compared scene.")
        findings.append(f"Change regions identified: {context.get('num_change_regions', 0)}.")
        measurements["Changed area"] = f"{context['change_pct']}%"
        measurements["Change regions"] = context.get("num_change_regions", 0)
        measurements["Similarity score"] = context.get("similarity_score", "N/A")
        evidence.extend(["Temporal change mask computed", "Before/after imagery compared", "Spatial change regions extracted"])

    if context.get("object_counts"):
        counts = ", ".join(f"{count} {label}(s)" for label, count in context["object_counts"].items())
        findings.append(f"Detected objects: {counts}.")
        measurements["Object counts"] = context["object_counts"]
        evidence.append("Object detection performed")

    land_cover = context.get("land_cover", [])
    if land_cover:
        top = land_cover[0]
        label = top.get("label", "unknown")
        findings.append(f"Dominant classified land cover: {label}.")
        evidence.append("Land-cover classification performed")

    if context.get("caption"):
        findings.append(f"Scene description: {context['caption']}.")
    elif context.get("caption_before") or context.get("caption_after"):
        findings.append(f"Before: {context.get('caption_before', 'N/A')}")
        findings.append(f"After: {context.get('caption_after', 'N/A')}")

    return {
        "query": query,
        "summary": answer,
        "findings": findings or ["No structured findings were returned by the analysis pipeline."],
        "measurements": measurements,
        "evidence": evidence or ["No additional evidence layers were produced."],
        "confidence": _confidence(context),
        "uncertainty": {
            "limitations": [
                "Results depend on image resolution and quality.",
                "Cloud and shadow effects may affect optical imagery.",
                "Measurements are limited to the evidence returned by the vision pipeline.",
            ]
        },
        "execution_trace": trace,
        "visual_layers": {
            "original_image": True,
            "temporal_image": "change_pct" in context,
            "change_mask": "diff_heatmap" in context,
            "object_detections": bool(context.get("detections")),
            "evidence_regions": bool(context.get("change_boxes") or context.get("detections")),
        },
        "mode": mode,
    }


def render_result(result):
    st.markdown("## SATQUERY AI")
    st.markdown(f"**Query:**  \n\"{result['query']}\"")
    st.markdown("### Analysis")
    st.write(result["summary"])
    st.markdown("### Key Findings")
    for finding in result["findings"]:
        st.markdown(f"• {finding}")
    if result["measurements"]:
        st.markdown("### Measurements")
        for name, value in result["measurements"].items():
            st.markdown(f"**{name}:** {value}")
    st.markdown("### Evidence")
    for item in result["evidence"]:
        st.markdown(f"✓ {item}")
    st.markdown("### Explanation")
    st.write(result["summary"])
    st.markdown("---")
    st.markdown("### UNCERTAINTY")
    st.markdown(f"**Confidence:** {result['confidence']['level']}")
    st.caption(result["confidence"]["basis"])
    st.markdown("**Limitations:**")
    for limitation in result["uncertainty"]["limitations"]:
        st.markdown(f"• {limitation}")


def render_chat_history():
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant" and turn.get("result"):
                render_result(turn["result"])
            else:
                st.write(turn["content"])


def render_actions(image, context, result, key, evidence_images=None):
    map_key = f"show_map_{key}"
    evidence_key = f"show_evidence_{key}"
    trace_key = f"show_trace_{key}"
    if map_key not in st.session_state:
        st.session_state[map_key] = False
        st.session_state[evidence_key] = False
        st.session_state[trace_key] = False

    buttons = st.columns(4)
    with buttons[0]:
        if st.button("🗺 View Map", key=f"map_button_{key}"):
            st.session_state[map_key] = not st.session_state[map_key]
    with buttons[1]:
        if st.button("🔍 View Evidence", key=f"evidence_button_{key}"):
            st.session_state[evidence_key] = not st.session_state[evidence_key]
    with buttons[2]:
        if st.button("⚙ Execution Trace", key=f"trace_button_{key}"):
            st.session_state[trace_key] = not st.session_state[trace_key]
    with buttons[3]:
        create_pdf = st.button("📄 Generate PDF Report", key=f"pdf_button_{key}")

    if st.session_state[map_key]:
        st.subheader("Map / imagery view")
        st.caption("Pixel-coordinate evidence view. Geospatial coordinates require georeferenced imagery.")
        st.image(image, use_column_width=True)

    if st.session_state[evidence_key]:
        st.subheader("Evidence regions")
        for label, evidence_image in evidence_images or []:
            st.image(evidence_image, caption=label, use_column_width=True)

    if st.session_state[trace_key]:
        st.subheader("Execution trace")
        for step in result["execution_trace"]:
            st.markdown(f"✓ {step}")

    if create_pdf:
        path = generate_report(
            image,
            context,
            st.session_state.chat_history,
            analysis_result=result,
            evidence_images=evidence_images,
        )
        with open(path, "rb") as report_file:
            st.download_button("Download detailed report", report_file, file_name="SatQuery_Report.pdf", key=f"download_{key}")


def render_visual_layers(mode, image, context, key, before_image=None, after_image=None):
    st.subheader("Visual layers")
    show_original = st.checkbox("Original Image", value=True, key=f"original_{key}")
    show_temporal = st.checkbox("Second/Temporal Image", value=mode == "Change Detection (Before/After)", key=f"temporal_{key}", disabled=after_image is None)
    show_mask = st.checkbox("Change Mask", value=True, key=f"mask_{key}", disabled="diff_heatmap" not in context)
    show_objects = st.checkbox("Object Detections", value=True, key=f"objects_{key}", disabled=not bool(context.get("detections")))
    show_regions = st.checkbox("Evidence Regions", value=True, key=f"regions_{key}", disabled=not bool(context.get("change_boxes")))

    if show_original and image is not None:
        st.image(image, caption="Original image", use_column_width=True)
    if show_temporal and after_image is not None:
        st.image(after_image, caption="Second / temporal image", use_column_width=True)
    if show_mask and context.get("diff_heatmap") is not None:
        st.image(context["diff_heatmap"], caption="Change mask", use_column_width=True)
    if show_objects and context.get("detections"):
        base_image = after_image if after_image is not None else image
        st.image(draw_detections(base_image, context["detections"]), caption="Object detections", use_column_width=True)
    if show_regions and context.get("change_boxes"):
        regions = [{**region, "label": f"Evidence {index}"} for index, region in enumerate(context["change_boxes"], start=1)]
        base_image = after_image if after_image is not None else image
        st.image(draw_evidence_regions(base_image, regions), caption="Evidence regions", use_column_width=True)


def run_analysis(query, mode, context_builder, answer_context, image, before_image=None, after_image=None):
    trace = ["Processing imagery..."]
    with st.status("🔍 Analyzing imagery...", expanded=True) as status_box:
        try:
            status_box.write("Processing imagery...")
            context = context_builder()
            trace.append("Running detection...")
            status_box.write("Running detection...")
            trace.append("Performing GIS analysis...")
            status_box.write("Performing GIS analysis...")
            answer_en = get_reasoner(os.environ.get("ANTHROPIC_API_KEY")).answer(query, answer_context(context), st.session_state.chat_history)
            answer = from_english(answer_en, st.session_state.lang_code)
            trace.append("Preparing evidence...")
            status_box.write("Preparing evidence...")
            result = build_analysis_result(query, answer, context, mode, trace + ["Analysis complete"])
            st.session_state.last_context = context
            st.session_state.last_result = result
            st.session_state.last_image = image
            st.session_state.last_before_image = before_image
            st.session_state.last_after_image = after_image
            st.session_state.last_error = None
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "result": result})
            status_box.update(label="✅ Analysis complete", state="complete", expanded=False)
            st.rerun()
        except Exception as error:
            st.session_state.last_error = repr(error)
            status_box.update(label="❌ Analysis could not be completed", state="error", expanded=True)
            st.error("Analysis could not be completed.")
            with st.expander("Developer/debug details"):
                st.code(st.session_state.last_error)


_init_state()
vision = get_vision_pipeline()

with st.sidebar:
    st.title("🛰️ SatQuery AI")
    st.caption("SIH26167 — Vision-Language Assistant for Remote Sensing")
    mode = st.radio("Mode", ["Single Image Analysis", "Change Detection (Before/After)", "Large Scene (Tiled) Analysis"])
    lang_name = st.selectbox("Query language", list(LANGUAGES.keys()))
    st.session_state.lang_code = LANGUAGES[lang_name]
    st.divider()
    st.markdown("**Sample queries**")
    st.code("What percentage of this image is vegetation?\nHow many vehicles are visible?\nIs there any flooding in this scene?\nCompare the before and after images — what changed?", language=None)

if st.session_state.analysis_mode != mode:
    _reset_analysis()
    st.session_state.analysis_mode = mode

st.header(mode)
col_img, col_chat = st.columns([1, 1])

if mode == "Single Image Analysis":
    with col_img:
        uploaded = st.file_uploader("Upload a satellite / aerial image", type=["jpg", "jpeg", "png", "tif", "tiff"])
        if uploaded:
            image, _ = load_image(uploaded)
            _set_source(f"single:{uploaded.name}:{uploaded.size}", image=image)
            render_visual_layers(mode, image, st.session_state.last_context or {}, "single")
    with col_chat:
        if st.session_state.last_image:
            render_chat_history()
            query = st.chat_input("Ask a question about this image...")
            if query:
                query_en = to_english(query, st.session_state.lang_code)
                run_analysis(query_en, mode, lambda: vision.analyze(st.session_state.last_image, query_en), lambda context: context, st.session_state.last_image)
            if st.session_state.last_result:
                render_result(st.session_state.last_result)
                render_actions(st.session_state.last_image, st.session_state.last_context, st.session_state.last_result, "single", [("Detected objects", draw_detections(st.session_state.last_image, st.session_state.last_context.get("detections", [])))])
        else:
            st.info("Upload an image to start querying.")

elif mode == "Change Detection (Before/After)":
    with col_img:
        before_file = st.file_uploader("BEFORE image", type=["jpg", "jpeg", "png"], key="before")
        after_file = st.file_uploader("AFTER image", type=["jpg", "jpeg", "png"], key="after")
        if before_file and after_file:
            img_before, _ = load_image(before_file)
            img_after, _ = load_image(after_file)
            _set_source(f"change:{before_file.name}:{before_file.size}:{after_file.name}:{after_file.size}", before=img_before, after=img_after, image=img_after)
            render_visual_layers(mode, img_after, st.session_state.last_context or {}, "change", before_image=img_before, after_image=img_after)
    with col_chat:
        if before_file and after_file:
            render_chat_history()
            query = st.chat_input("Ask about what changed between these images...")
            if query:
                query_en = to_english(query, st.session_state.lang_code)

                def build_context():
                    context = build_change_context(vision, img_before, img_after, query_en)
                    detections = vision.detect_objects(img_after)
                    context["detections"] = detections
                    context["object_counts"] = {}
                    for detection in detections:
                        label = detection["label"]
                        context["object_counts"][label] = context["object_counts"].get(label, 0) + 1
                    return context

                run_analysis(query_en, mode, build_context, lambda context: {key: value for key, value in context.items() if key != "diff_heatmap"}, img_after, img_before, img_after)
            if st.session_state.last_result:
                render_result(st.session_state.last_result)
                change_context = st.session_state.last_context
                evidence_images = [("Change mask", change_context.get("diff_heatmap"))]
                if change_context.get("change_boxes"):
                    regions = [{**region, "label": f"Evidence {index}"} for index, region in enumerate(change_context["change_boxes"], start=1)]
                    evidence_images.append(("Evidence regions", draw_evidence_regions(img_after, regions)))
                if change_context.get("detections"):
                    evidence_images.append(("Object detections", draw_detections(img_after, change_context["detections"])))
                render_actions(st.session_state.last_after_image, st.session_state.last_context, st.session_state.last_result, "change", evidence_images)
        else:
            st.info("Upload both before and after images to begin.")

else:
    with col_img:
        uploaded = st.file_uploader("Upload a large satellite scene", type=["jpg", "jpeg", "png", "tif", "tiff"])
        if uploaded:
            image, _ = load_image(uploaded)
            _set_source(f"tiled:{uploaded.name}:{uploaded.size}", image=image)
            st.caption(f"Scene size: {image.size[0]}×{image.size[1]} px; {len(tile_image(image))} tile(s)")
            render_visual_layers(mode, image, st.session_state.last_context or {}, "tiled")
    with col_chat:
        if uploaded:
            render_chat_history()
            query = st.chat_input("Ask a question about this scene...")
            if query:
                query_en = to_english(query, st.session_state.lang_code)
                run_analysis(query_en, mode, lambda: vision.analyze_large_scene(image, query_en), lambda context: context, image)
            if st.session_state.last_result:
                render_result(st.session_state.last_result)
                render_actions(st.session_state.last_image, st.session_state.last_context, st.session_state.last_result, "tiled", [("Detected objects", draw_detections(st.session_state.last_image, st.session_state.last_context.get("detections", [])))])
        else:
            st.info("Upload a large scene to start querying.")
