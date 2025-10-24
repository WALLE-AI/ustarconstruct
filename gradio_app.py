"""Gradio UI for visualizing `analyze_image` outputs from `simple_rag.py`.

The app expects JSON or JSONL files in the same format as the pipeline output.
Layout requirements:
    - Left column: original image (top) and annotated image (bottom).
    - Right column: hazard details, overall findings, and highlighted KB citations.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import traceback
import uuid

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import html

from simple_rag import mrag_run

# We rely on `analyze_image` output schema from `simple_rag.py`.

JSON_EXTS = {".json", ".jsonl"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


DEFAULT_COLORS = [
    (220, 68, 55),   # red
    (56, 142, 60),   # green
    (25, 118, 210),  # blue
    (255, 143, 0),   # orange
    (123, 31, 162),  # purple
    (0, 151, 167),   # teal
]


@dataclass
class ParsedRecord:
    raw: Dict[str, Any]
    label: str
    image_path: Optional[str]


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_path(record: Dict[str, Any], base_dir: str) -> Optional[str]:
    image_path = record.get("image_path") or record.get("image") or ""
    if not image_path:
        return None
    if os.path.isabs(image_path):
        return image_path
    resolved = os.path.join(base_dir, image_path)
    if os.path.exists(resolved):
        return resolved
    # fallback: try relative to current working directory
    return image_path if os.path.exists(image_path) else None


def load_records_from_file(file_obj) -> List[ParsedRecord]:
    if file_obj is None:
        return []

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    base_dir = os.path.dirname(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    text: str
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    records: List[Dict[str, Any]] = []

    def _append_record(obj: Any):
        if isinstance(obj, dict):
            records.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    records.append(item)

    tried_full_parse = False
    if ext in {".jsonl", ".json"}:
        try:
            data = json.loads(text)
            _append_record(data)
            tried_full_parse = True
        except json.JSONDecodeError:
            tried_full_parse = False

    if not records and ext == ".jsonl" and not tried_full_parse:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            _append_record(data)

    if not records and ext == ".json":
        raise ValueError("JSON file is invalid or does not contain an object.")

    if not records:
        raise ValueError("No valid records could be parsed from the file.")

    parsed: List[ParsedRecord] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        label = str(rec.get("task_id") or f"record-{idx+1}")
        image_path = _resolve_image_path(rec, base_dir)
        parsed.append(ParsedRecord(raw=rec, label=label, image_path=image_path))
    return parsed


def draw_bounding_boxes(
    image_path: Optional[str],
    hazards: List[Dict[str, Any]],
) -> Optional[Image.Image]:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for i, hazard in enumerate(hazards):
        region = hazard.get("evidence_region") or {}
        if not isinstance(region, dict) or region.get("type") != "bbox":
            continue

        bbox = region.get("bbox_xywh_norm")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue

        x, y, w, h = bbox
        x1 = max(0, min(1, float(x))) * width
        y1 = max(0, min(1, float(y))) * height
        x2 = max(0, min(1, float(x + w))) * width
        y2 = max(0, min(1, float(y + h))) * height

        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        title = hazard.get("label") or hazard.get("category") or f"Hazard {i+1}"
        severity = hazard.get("severity")
        text = f"{title}"
        if severity:
            text += f" (Severity {severity})"

        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = 4
        box_coords = [
            x1,
            max(0, y1 - text_height - 2 * padding),
            x1 + text_width + 2 * padding,
            y1,
        ]
        draw.rectangle(box_coords, fill=color)
        draw.text((x1 + padding, box_coords[1] + padding), text, fill="white", font=font)

    return image


def format_overview(record: Dict[str, Any]) -> str:
    scene_info = record.get("scene_guess") or {}
    scene = scene_info.get("scene", "Unknown")
    rationale = scene_info.get("rationale")
    overall_risk = record.get("overall_risk_score", 0)
    summary = record.get("summary") or ""
    auto_queries = record.get("auto_queries") or []

    lines = [
        f"**Scene:** {scene}",
        f"**Overall Risk Score:** {overall_risk}",
    ]
    if rationale:
        lines.append(f"**Scene Rationale:** {rationale}")
    if summary:
        lines.append(f"**Summary:** {summary}")
    if auto_queries:
        joined = ", ".join(auto_queries)
        lines.append(f"**Auto Queries:** {joined}")
    return "\n\n".join(lines)


def format_hazards(hazards: List[Dict[str, Any]]) -> str:
    if not hazards:
        return "No hazard data."

    sections: List[str] = []
    for idx, hazard in enumerate(hazards, start=1):
        lines = [
            f"### Hazard {idx}: {hazard.get('label', 'Unnamed')}",
            f"- Category: {hazard.get('category', '-')}",
            f"- Risk Score: {hazard.get('risk_score', '-')}, Severity: {hazard.get('severity', '-')}",
            f"- Confidence: {hazard.get('confidence', '-')}",
        ]
        rationale = hazard.get("rationale_brief")
        if rationale:
            lines.append(f"- Rationale: {rationale}")
        remediations = hazard.get("remediation") or []
        if remediations:
            joined = "; ".join(remediations)
            lines.append(f"- Recommendations: {joined}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def format_citations(hazards: List[Dict[str, Any]]) -> str:
    collected: List[Tuple[str, str, str]] = []
    for hazard in hazards:
        for citation in hazard.get("kb_citations", []) or []:
            doc = citation.get("doc_id", "")
            clause = citation.get("clause_id", "")
            brief = citation.get("clause_brief", "")
            collected.append((doc, clause, brief))

    if not collected:
        return "<mark>No citations</mark>"

    html_lines = []
    for doc, clause, brief in collected:
        text = f"{doc} {clause}".strip()
        if brief:
            text += f": {brief}"
        text = html.escape(text)
        html_lines.append(f"<div><mark>{text}</mark></div>")
    return "\n".join(html_lines)


def render_record(parsed: ParsedRecord) -> Tuple[Any, Any, str, str, str]:
    record = parsed.raw
    hazards = record.get("hazards", [])
    original_image = None
    annotated_image = None

    if parsed.image_path and os.path.exists(parsed.image_path):
        try:
            original_image = Image.open(parsed.image_path).convert("RGB")
        except Exception:
            original_image = None

    annotated = draw_bounding_boxes(parsed.image_path, hazards)
    if annotated is not None:
        annotated_image = annotated

    overview_md = format_overview(record)
    hazards_md = format_hazards(hazards)
    citations_html = format_citations(hazards)
    return original_image, annotated_image, overview_md, hazards_md, citations_html


def _append_result_record(
    existing_records: List[ParsedRecord],
    result: Dict[str, Any],
    image_path: Optional[str],
) -> Tuple[List[ParsedRecord], ParsedRecord]:
    labels_in_use = {rec.label for rec in existing_records}
    base_label = str(result.get("task_id") or f"mrag-{uuid.uuid4().hex[:8]}")
    label = base_label
    suffix = 2
    while label in labels_in_use:
        label = f"{base_label}-{suffix}"
        suffix += 1

    record_copy = dict(result)
    if image_path:
        record_copy["image_path"] = image_path

    new_record = ParsedRecord(raw=record_copy, label=label, image_path=image_path)
    return existing_records + [new_record], new_record


def handle_file_upload(file_obj, current_records):
    existing_records: List[ParsedRecord] = list(current_records or [])
    dropdown_preserve = gr.update(
        choices=[rec.label for rec in existing_records],
        value=(existing_records[-1].label if existing_records else None),
    )

    if file_obj is None:
        message = "No file provided."
        return (
            existing_records,
            dropdown_preserve,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "<mark>No citations</mark>",
            message,
        )

    file_path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)

    try:
        parsed_records = load_records_from_file(file_obj)
    except Exception as exc:
        # Not a JSON/JSONL file or parsing failed; try treating as image.
        original_error = str(exc)
        if not file_path:
            message = f"File parsing failed. Error: {original_error}"
            return (
                existing_records,
                dropdown_preserve,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "<mark>No citations</mark>",
                message,
            )

        ext = os.path.splitext(str(file_path))[1].lower()
        if ext not in IMAGE_EXTS:
            message = f"File parsing failed. Error: {original_error}"
            return (
                existing_records,
                dropdown_preserve,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "<mark>No citations</mark>",
                message,
            )

        try:
            result = mrag_run(file_path)
        except Exception as run_exc:
            message = f"mrag_run failed: {run_exc}"
            detail = traceback.format_exc()
            return (
                existing_records,
                dropdown_preserve,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "<mark>No citations</mark>",
                message + "\n\n" + detail,
            )

        if not isinstance(result, dict):
            message = "mrag_run returned a non-dict payload; nothing to display."
            return (
                existing_records,
                dropdown_preserve,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "<mark>No citations</mark>",
                message,
            )

        image_path = result.get("image_path") or file_path
        new_records, new_record = _append_result_record(existing_records, result, image_path)
        original_image, annotated_image, overview_md, hazards_md, citations_html = render_record(new_record)
        dropdown = gr.update(
            choices=[rec.label for rec in new_records],
            value=new_record.label,
        )
        hazards_count = len(result.get("hazards", []) or [])
        status_msg = f"mrag_run completed with {hazards_count} hazard(s)."
        return (
            new_records,
            dropdown,
            original_image,
            annotated_image,
            overview_md,
            hazards_md,
            citations_html,
            status_msg,
        )

    if not parsed_records:
        message = "No records found in file."
        empty_dropdown = gr.update(choices=[], value=None)
        return (
            [],
            empty_dropdown,
            None,
            None,
            message,
            "",
            "<mark>No citations</mark>",
            message,
        )

    first = parsed_records[0]
    original_image, annotated_image, overview_md, hazards_md, citations_html = render_record(first)
    dropdown = gr.update(
        choices=[rec.label for rec in parsed_records],
        value=first.label,
    )
    filename = os.path.basename(str(file_path)) if file_path else ""
    status_msg = f"Loaded {len(parsed_records)} record(s) from {filename}."
    return (
        parsed_records,
        dropdown,
        original_image,
        annotated_image,
        overview_md,
        hazards_md,
        citations_html,
        status_msg,
    )


def handle_record_change(selected_label: str, records: List[ParsedRecord]):
    if not selected_label or not records:
        return None, None, "", "", "<mark>No citations</mark>"

    for rec in records:
        if rec.label == selected_label:
            original_image, annotated_image, overview_md, hazards_md, citations_html = render_record(rec)
            return original_image, annotated_image, overview_md, hazards_md, citations_html

    return None, None, "Record not found.", "", "<mark>No citations</mark>"


with gr.Blocks(title="Safety Hazard Review") as demo:
    gr.Markdown(
        "## Hazard Review Dashboard\n"
        "Upload JSON/JSONL output from `simple_rag.analyze_image` to browse existing results, "
        "or provide an image and call `mrag_run` to produce a fresh assessment."
    )

    records_state = gr.State([])

    with gr.Row():
        file_input = gr.File(
            label="Upload result file or image (.json / .jsonl / image)",
            file_types=[".json", ".jsonl", ".png", ".jpg", ".jpeg", ".bmp", ".gif"],
        )
        record_selector = gr.Dropdown(label="Record selection", interactive=True)

    with gr.Row():
        with gr.Column(scale=1):
            original_img = gr.Image(label="Original image", image_mode="RGB")
            annotated_img = gr.Image(label="Annotated image", image_mode="RGB")
        with gr.Column(scale=1):
            overview_md = gr.Markdown()
            hazards_md = gr.Markdown()
            citations_html = gr.HTML()
            status_md = gr.Markdown(value="Awaiting action", label="Status")

    file_input.change(
        fn=handle_file_upload,
        inputs=[file_input, records_state],
        outputs=[
            records_state,
            record_selector,
            original_img,
            annotated_img,
            overview_md,
            hazards_md,
            citations_html,
            status_md,
        ],
        show_progress=True,
    )

    record_selector.change(
        fn=handle_record_change,
        inputs=[record_selector, records_state],
        outputs=[original_img, annotated_img, overview_md, hazards_md, citations_html],
        show_progress=False,
    )

if __name__ == "__main__":
    demo.launch()
