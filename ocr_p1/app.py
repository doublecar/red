"""
OCR p1 — Tesseract test harness for engineering drawing sheets.
Single-page US-letter PDF  →  configurable pre-processing  →  Tesseract OCR.

Version: p1
"""
import io
import json
import os
import tempfile
import traceback
from pathlib import Path

import cv2
import fitz                         # PyMuPDF
import numpy as np
import pytesseract
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ── constants ────────────────────────────────────────────────────────────────
VERSION = "p1"
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024   # 50 MB

PSM_DESCRIPTIONS = {
    0:  "0 – OSD only",
    1:  "1 – Auto with OSD",
    2:  "2 – Auto, no OSD, no OCR",
    3:  "3 – Auto (default)",
    4:  "4 – Single column of variable-size text",
    5:  "5 – Single uniform vertical block",
    6:  "6 – Single uniform block of text",
    7:  "7 – Single text line",
    8:  "8 – Single word",
    9:  "9 – Single word in a circle",
    10: "10 – Single character",
    11: "11 – Sparse text — find as much text as possible",
    12: "12 – Sparse text with OSD",
    13: "13 – Raw line — treat as a single text line",
}

OEM_DESCRIPTIONS = {
    0: "0 – Legacy engine only",
    1: "1 – Neural nets LSTM only",
    2: "2 – Legacy + LSTM",
    3: "3 – Default (best available)",
}

# ── app setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# ── helpers ──────────────────────────────────────────────────────────────────

def pdf_to_pil(pdf_bytes: bytes, dpi: int) -> Image.Image:
    """Render first page of a PDF at the requested DPI and return a PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    """Rotate without cropping (expand canvas)."""
    if angle == 0:
        return img
    return img.rotate(-angle, expand=True, fillcolor=(255, 255, 255))


def despeckle(arr: np.ndarray, strength: int) -> np.ndarray:
    """
    Remove small noise specks.
    strength 1-5 maps to morphological opening kernel size (1=off → no change,
    2-5 → kernel 2-5 px).
    """
    if strength <= 1:
        return arr
    k = strength
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return result


def remove_long_lines(arr: np.ndarray, min_length: int, gap: int) -> np.ndarray:
    """
    Detect long straight lines (HoughLinesP) and paint them white to prevent
    lead lines / border lines from confusing Tesseract.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_length,
        maxLineGap=gap,
    )
    out = arr.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
    return out


def preprocess(
    img: Image.Image,
    rotation: float,
    despeckle_strength: int,
    remove_lines: bool,
    line_min_length: int,
    line_max_gap: int,
) -> Image.Image:
    """Apply all pre-processing steps and return the final PIL image."""
    img = rotate_image(img, rotation)
    arr = pil_to_cv2(img)
    if remove_lines:
        arr = remove_long_lines(arr, line_min_length, line_max_gap)
    if despeckle_strength > 1:
        arr = despeckle(arr, despeckle_strength)
    return cv2_to_pil(arr)


def run_tesseract(img: Image.Image, psm: int, oem: int, whitelist: str) -> dict:
    """Run Tesseract and return words + full text."""
    config_parts = [f"--psm {psm}", f"--oem {oem}"]
    if whitelist.strip():
        config_parts.append(f"-c tessedit_char_whitelist={whitelist.strip()}")
    config = " ".join(config_parts)

    full_text = pytesseract.image_to_string(img, config=config)

    # word-level data with confidence
    data = pytesseract.image_to_data(
        img, config=config, output_type=pytesseract.Output.DICT
    )
    words = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = int(data["conf"][i])
        words.append({
            "text": word,
            "conf": conf,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "block": data["block_num"][i],
            "line": data["line_num"][i],
        })

    return {"full_text": full_text, "words": words, "config": config}


def pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        version=VERSION,
        psm_descriptions=PSM_DESCRIPTIONS,
        oem_descriptions=OEM_DESCRIPTIONS,
    )


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        # ── PDF bytes ───────────────────────────────────────────────────────
        if "pdf" not in request.files:
            return jsonify({"error": "No PDF uploaded."}), 400
        pdf_file = request.files["pdf"]
        pdf_bytes = pdf_file.read()
        if not pdf_bytes:
            return jsonify({"error": "Empty file."}), 400

        # ── parameters ──────────────────────────────────────────────────────
        def gi(key, default):
            try:
                return int(request.form.get(key, default))
            except (ValueError, TypeError):
                return default

        def gf(key, default):
            try:
                return float(request.form.get(key, default))
            except (ValueError, TypeError):
                return default

        dpi            = gi("dpi", 300)
        psm            = gi("psm", 11)
        oem            = gi("oem", 3)
        whitelist      = request.form.get("whitelist", "0123456789")
        rotation       = gf("rotation", 0.0)
        despeckle_str  = gi("despeckle", 1)
        remove_lines   = request.form.get("remove_lines", "false").lower() == "true"
        line_min_len   = gi("line_min_length", 100)
        line_max_gap   = gi("line_max_gap", 10)

        # clamp
        dpi           = max(72, min(dpi, 600))
        psm           = max(0, min(psm, 13))
        oem           = max(0, min(oem, 3))
        despeckle_str = max(1, min(despeckle_str, 5))

        # ── pipeline ────────────────────────────────────────────────────────
        raw_img = pdf_to_pil(pdf_bytes, dpi)
        processed_img = preprocess(
            raw_img,
            rotation=rotation,
            despeckle_strength=despeckle_str,
            remove_lines=remove_lines,
            line_min_length=line_min_len,
            line_max_gap=line_max_gap,
        )
        result = run_tesseract(processed_img, psm, oem, whitelist)

        # ── encode preview images ────────────────────────────────────────────
        # Downscale for preview if very large
        def thumb(img, max_w=1200):
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
            return img

        result["raw_preview"]       = pil_to_data_uri(thumb(raw_img))
        result["processed_preview"] = pil_to_data_uri(thumb(processed_img))
        result["image_size"]        = {"w": processed_img.width, "h": processed_img.height}
        result["version"]           = VERSION

        return jsonify(result)

    except Exception:
        tb = traceback.format_exc()
        return jsonify({"error": tb}), 500


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5100))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
