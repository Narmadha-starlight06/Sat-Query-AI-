"""
report_generator.py — Exports a SatQuery AI analysis session as a shareable PDF.

Motivation: for the real users of this system (disaster-response officers,
agriculture extension workers, urban planners) a chat window isn't the final
deliverable — a shareable report is. This gives judges a concrete "output
artifact" to look at beyond the live demo.
"""

import io
from datetime import datetime
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_report(
    image,
    context: dict,
    chat_history: list,
    output_path: str = "SatQuery_Report.pdf",
    analysis_result: dict = None,
    evidence_images: list = None,
):
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20)
    story = []

    story.append(Paragraph("SatQuery AI — Analysis Report", title_style))
    story.append(Paragraph(datetime.now().strftime("Generated: %d %b %Y, %H:%M"), styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    if analysis_result:
        story.append(Paragraph("Query", styles["Heading2"]))
        story.append(Paragraph(escape(str(analysis_result.get("query", "N/A"))), styles["Normal"]))
        story.append(Paragraph("Analysis", styles["Heading2"]))
        story.append(Paragraph(escape(str(analysis_result.get("summary", "N/A"))), styles["Normal"]))

        findings = analysis_result.get("findings", [])
        if findings:
            story.append(Paragraph("Key Findings", styles["Heading2"]))
            for finding in findings:
                story.append(Paragraph(f"• {escape(str(finding))}", styles["Normal"]))

        measurements = analysis_result.get("measurements", {})
        if measurements:
            story.append(Paragraph("Measurements", styles["Heading2"]))
            for name, value in measurements.items():
                story.append(Paragraph(f"{escape(str(name))}: {escape(str(value))}", styles["Normal"]))

        evidence = analysis_result.get("evidence", [])
        if evidence:
            story.append(Paragraph("Evidence Provenance", styles["Heading2"]))
            for item in evidence:
                story.append(Paragraph(f"✓ {escape(str(item))}", styles["Normal"]))

        confidence = analysis_result.get("confidence", {})
        if confidence:
            story.append(Paragraph("Confidence and Uncertainty", styles["Heading2"]))
            story.append(Paragraph(escape(str(confidence.get("level", "Not estimated"))), styles["Normal"]))
            for limitation in analysis_result.get("uncertainty", {}).get("limitations", []):
                story.append(Paragraph(f"• {escape(str(limitation))}", styles["Normal"]))

        trace = analysis_result.get("execution_trace", [])
        if trace:
            story.append(Paragraph("Execution Trace", styles["Heading2"]))
            for step in trace:
                story.append(Paragraph(f"✓ {escape(str(step))}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

    if image is not None:
        story.append(Paragraph("Input Imagery", styles["Heading2"]))
        story.append(Paragraph(f"Image dimensions: {image.size[0]} × {image.size[1]} pixels", styles["Normal"]))
        buf = io.BytesIO()
        img_copy = image.copy()
        img_copy.thumbnail((400, 400))
        img_copy.save(buf, format="PNG")
        buf.seek(0)
        story.append(RLImage(buf, width=8 * cm, height=8 * cm * img_copy.size[1] / img_copy.size[0]))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Scene Description", styles["Heading2"]))
    story.append(Paragraph(escape(str(context.get("caption", "N/A"))), styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    if context.get("land_cover"):
        story.append(Paragraph("Land Cover Classification", styles["Heading2"]))
        rows = [["Label", "Confidence / Votes"]]
        for lc in context["land_cover"]:
            score = lc.get("score", lc.get("tile_votes", ""))
            rows.append([lc["label"], f"{score}"])
        t = Table(rows, colWidths=[10 * cm, 5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5339")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))

    if context.get("object_counts"):
        story.append(Paragraph("Detected Objects", styles["Heading2"]))
        counts_text = ", ".join(f"{v} × {k}" for k, v in context["object_counts"].items())
        story.append(Paragraph(counts_text, styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

    if "change_pct" in context:
        story.append(Paragraph("Change Detection Summary", styles["Heading2"]))
        story.append(Paragraph(
            f"Change detected across {context['change_pct']}% of the scene "
            f"(structural similarity score: {context.get('similarity_score', 'N/A')}). "
            f"{context.get('num_change_regions', 0)} distinct change region(s) identified.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.3 * cm))

    for label, evidence_image in evidence_images or []:
        if evidence_image is None:
            continue
        story.append(Paragraph(escape(str(label)), styles["Heading2"]))
        buf = io.BytesIO()
        img_copy = evidence_image.copy()
        img_copy.thumbnail((500, 500))
        img_copy.save(buf, format="PNG")
        buf.seek(0)
        story.append(RLImage(buf, width=10 * cm, height=10 * cm * img_copy.size[1] / img_copy.size[0]))
        story.append(Spacer(1, 0.3 * cm))

    if chat_history:
        story.append(Paragraph("Query &amp; Answer Log", styles["Heading2"]))
        for turn in chat_history:
            role = "Q" if turn["role"] == "user" else "A"
            story.append(Paragraph(f"<b>{role}:</b> {escape(str(turn['content']))}", styles["Normal"]))
            story.append(Spacer(1, 0.15 * cm))

    doc.build(story)
    return output_path
