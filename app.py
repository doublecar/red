"""
Ann3 — Patent Figure Annotation App
Flask + Florence-2 (microsoft/Florence-2-base-ft)

Compatible with Python 3.11.  The AttributeErrors seen on Python 3.14
  • 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
  • RobertaTokenizer has no attribute additional_special_tokens
are caused by CPython internals changes in 3.14 that break certain
class-level descriptor access patterns inside transformers.  Running
under Python 3.11 with transformers==4.45.2 eliminates both errors.
"""

from __future__ import annotations

import io
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Optional

# ── Flask ──────────────────────────────────────────────────────────────────
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    jsonify,
    send_from_directory,
    flash,
)
from werkzeug.utils import secure_filename

# ── Image / PDF ────────────────────────────────────────────────────────────
from PIL import Image
import fitz  # PyMuPDF

# ── Florence-2 ─────────────────────────────────────────────────────────────
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ann3")

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB

# Florence-2 model name — change to "microsoft/Florence-2-large-ft" for better quality
FLORENCE2_MODEL = os.environ.get("FLORENCE2_MODEL", "microsoft/Florence-2-base-ft")

# Resolution when rasterising PDF pages (150 dpi is plenty for figures)
PDF_DPI = int(os.environ.get("PDF_DPI", "150"))

# Default Florence-2 task.  Options:
#   <CAPTION>  <DETAILED_CAPTION>  <MORE_DETAILED_CAPTION>
#   <OD>  <DENSE_REGION_CAPTION>  <OCR>
DEFAULT_TASK = os.environ.get("FLORENCE2_TASK", "<DETAILED_CAPTION>")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "ann3-dev-secret-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ──────────────────────────────────────────────────────────────────────────
# Florence-2 model loading (lazy, loaded once on first request)
# ──────────────────────────────────────────────────────────────────────────
_model: Optional[AutoModelForCausalLM] = None
_processor: Optional[AutoProcessor] = None
_device: str = "cpu"


def get_model():
    """Load Florence-2 model and processor, caching in module globals."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    log.info("Loading Florence-2 model: %s", FLORENCE2_MODEL)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", _device)

    # trust_remote_code=True is required for Florence-2 custom modelling code
    _processor = AutoProcessor.from_pretrained(
        FLORENCE2_MODEL,
        trust_remote_code=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        FLORENCE2_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # float32 for CPU; float16 on GPU
    ).to(_device)
    _model.eval()

    log.info("Florence-2 ready on %s", _device)
    return _model, _processor, _device


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def pdf_to_images(pdf_path: Path, dpi: int = PDF_DPI) -> list[Image.Image]:
    """Rasterise every page of a PDF and return a list of PIL Images."""
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0  # PyMuPDF default is 72 dpi
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def run_florence2(image: Image.Image, task: str) -> dict:
    """
    Run a single Florence-2 inference on *image* with *task* prompt.

    Returns the parsed result dict from the processor.
    """
    model, processor, device = get_model()

    inputs = processor(text=task, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height),
    )
    return result


def save_page_image(image: Image.Image, session_dir: Path, page_idx: int) -> str:
    """Save a page image to disk; return path relative to static/."""
    filename = f"page_{page_idx + 1:03d}.jpg"
    dest = session_dir / filename
    image.save(dest, "JPEG", quality=85)
    # Return URL-friendly path relative to static/ (forward slashes for Windows compat)
    return dest.relative_to(BASE_DIR / "static").as_posix()


# ──────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", default_task=DEFAULT_TASK)


@app.route("/annotate", methods=["POST"])
def annotate():
    # ── Validate upload ────────────────────────────────────────────────────
    if "pdf" not in request.files:
        flash("No file part in request.", "error")
        return redirect(url_for("index"))

    file = request.files["pdf"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only PDF files are accepted.", "error")
        return redirect(url_for("index"))

    task = request.form.get("task", DEFAULT_TASK).strip()
    if not task:
        task = DEFAULT_TASK

    pages_param = request.form.get("pages", "all").strip().lower()

    # ── Save PDF ───────────────────────────────────────────────────────────
    session_id = uuid.uuid4().hex
    session_dir = UPLOAD_FOLDER / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(file.filename)
    pdf_path = session_dir / safe_name
    file.save(str(pdf_path))
    log.info("Saved PDF: %s", pdf_path)

    # ── Rasterise pages ────────────────────────────────────────────────────
    try:
        all_images = pdf_to_images(pdf_path)
    except Exception as exc:
        log.exception("PDF rendering failed")
        flash(f"Could not read PDF: {exc}", "error")
        return redirect(url_for("index"))

    total_pages = len(all_images)
    if total_pages == 0:
        flash("PDF has no renderable pages.", "error")
        return redirect(url_for("index"))

    # Determine which page indices to annotate
    if pages_param == "all":
        selected_indices = list(range(total_pages))
    else:
        selected_indices = []
        for part in pages_param.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                selected_indices.extend(range(int(lo) - 1, int(hi)))
            elif part.isdigit():
                selected_indices.append(int(part) - 1)
        selected_indices = [i for i in selected_indices if 0 <= i < total_pages]
        if not selected_indices:
            flash("No valid page numbers specified.", "error")
            return redirect(url_for("index"))

    # ── Annotate ───────────────────────────────────────────────────────────
    results = []
    for idx in selected_indices:
        img = all_images[idx]
        img_rel_path = save_page_image(img, session_dir, idx)
        log.info("Annotating page %d / %d …", idx + 1, total_pages)

        try:
            raw = run_florence2(img, task)
            # Florence-2 returns {task_key: value}; grab the first value
            annotation = next(iter(raw.values()), "")
            log.info("Page %d raw result: %r", idx + 1, raw)
            if isinstance(annotation, dict):
                # e.g. OD returns {"bboxes": [...], "labels": [...]}
                annotation_text = _format_od_result(annotation)
            else:
                annotation_text = str(annotation)
        except Exception as exc:
            log.exception("Annotation failed for page %d", idx + 1)
            annotation_text = f"[Error: {exc}]"

        results.append(
            {
                "page": idx + 1,
                "image_url": url_for("static", filename=img_rel_path),
                "annotation": annotation_text,
            }
        )

    return render_template(
        "results.html",
        results=results,
        filename=safe_name,
        task=task,
        total_pages=total_pages,
    )


def _format_od_result(od: dict) -> str:
    """Pretty-print object-detection result as labelled list."""
    labels = od.get("labels", [])
    bboxes = od.get("bboxes", [])
    lines = []
    for label, bbox in zip(labels, bboxes):
        x1, y1, x2, y2 = [round(v) for v in bbox]
        lines.append(f"{label}: [{x1}, {y1}, {x2}, {y2}]")
    return "\n".join(lines) if lines else "(no objects detected)"


@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": FLORENCE2_MODEL, "device": _device})


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    log.info("Starting Ann3 on http://0.0.0.0:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
