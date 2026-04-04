"""
OCR p2 — Tesseract test harness with built-in auto-tune.
Single-page US-letter PDF  →  configurable pre-processing  →  Tesseract OCR.

Version: p2
Changes from p1:
  - Defaults updated to best-found settings (DPI 300, PSM 6, OEM 1)
  - Auto-tune endpoint: grid search runs in a background thread,
    progress polled by the frontend, best params returned and applied.
"""
import io
import itertools
import os
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ── Windows tesseract path ────────────────────────────────────────────────────
import platform, shutil
if platform.system() == "Windows":
    _candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    if not shutil.which("tesseract"):
        for _p in _candidates:
            if Path(_p).exists():
                pytesseract.pytesseract.tesseract_cmd = _p
                break

# ── constants ─────────────────────────────────────────────────────────────────
VERSION = "p2"
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

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

# Tune grid
TUNE_GRID = {
    "dpi":          [200, 300, 400],
    "psm":          [6, 7, 11, 12, 13],
    "oem":          [1, 3],
    "despeckle":    [1, 2, 3],
    "line_removal": [
        (False, 0,   0),
        (True,  50,  5),
        (True,  80,  10),
        (True,  100, 10),
        (True,  150, 15),
        (True,  200, 20),
    ],
    "rotation":     [0],
}

# In-memory store for tune jobs  {job_id: {status, progress, total, results, best}}
_tune_jobs: dict = {}
_tune_lock = threading.Lock()

# ── app ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# ── image pipeline helpers ────────────────────────────────────────────────────

def pdf_to_pil(pdf_bytes: bytes, dpi: int) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def pil_to_cv2(img): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def cv2_to_pil(arr): return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def rotate_image(img: Image.Image, angle: float) -> Image.Image:
    if angle == 0:
        return img
    return img.rotate(-angle, expand=True, fillcolor=(255, 255, 255))

def despeckle_img(arr, strength):
    if strength <= 1:
        return arr
    k = strength
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

def remove_long_lines(arr, min_length, gap):
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=min_length, maxLineGap=gap)
    out = arr.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
    return out

def preprocess(img, rotation, despeckle_strength, remove_lines, line_min, line_gap):
    img = rotate_image(img, rotation)
    arr = pil_to_cv2(img)
    if remove_lines:
        arr = remove_long_lines(arr, line_min, line_gap)
    if despeckle_strength > 1:
        arr = despeckle_img(arr, despeckle_strength)
    return cv2_to_pil(arr)

def run_tesseract(img, psm, oem, whitelist):
    config_parts = [f"--psm {psm}", f"--oem {oem}"]
    if whitelist.strip():
        config_parts.append(f"-c tessedit_char_whitelist={whitelist.strip()}")
    config = " ".join(config_parts)
    full_text = pytesseract.image_to_string(img, config=config)
    data = pytesseract.image_to_data(img, config=config,
                                     output_type=pytesseract.Output.DICT)
    words = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        words.append({
            "text": word, "conf": int(data["conf"][i]),
            "left": data["left"][i], "top": data["top"][i],
            "width": data["width"][i], "height": data["height"][i],
            "block": data["block_num"][i], "line": data["line_num"][i],
        })
    return {"full_text": full_text, "words": words, "config": config}

def pil_to_data_uri(img, fmt="PNG"):
    import base64
    buf = io.BytesIO()
    if img.width > 1200:
        ratio = 1200 / img.width
        img = img.resize((1200, int(img.height * ratio)), Image.LANCZOS)
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ── tune helpers ──────────────────────────────────────────────────────────────

def score_words(found_words, expected):
    need = Counter(expected)
    have = Counter(found_words)
    hits = sum(min(have.get(n, 0), c) for n, c in need.items())
    found  = []
    missed = []
    for n, c in need.items():
        got = min(have.get(n, 0), c)
        found  += [n] * got
        missed += [n] * (c - got)
    return hits, found, missed

def run_single(pdf_bytes, dpi, psm, oem, ds, rotation, do_lines, line_min, line_gap, whitelist):
    img = pdf_to_pil(pdf_bytes, dpi)
    img = rotate_image(img, rotation)
    arr = pil_to_cv2(img)
    if do_lines:
        arr = remove_long_lines(arr, line_min, line_gap)
    if ds > 1:
        arr = despeckle_img(arr, ds)
    img = cv2_to_pil(arr)
    cfg = f"--psm {psm} --oem {oem}"
    if whitelist.strip():
        cfg += f" -c tessedit_char_whitelist={whitelist.strip()}"
    data = pytesseract.image_to_data(img, config=cfg,
                                     output_type=pytesseract.Output.DICT)
    return [w.strip() for w in data["text"] if w.strip()]

def tune_worker(job_id, pdf_bytes, expected, whitelist):
    combos = list(itertools.product(
        TUNE_GRID["dpi"], TUNE_GRID["psm"], TUNE_GRID["oem"],
        TUNE_GRID["despeckle"], TUNE_GRID["line_removal"], TUNE_GRID["rotation"],
    ))
    total = len(combos)
    results = []

    with _tune_lock:
        _tune_jobs[job_id].update({"total": total, "progress": 0, "status": "running"})

    for i, (dpi, psm, oem, ds, (do_lines, line_min, line_gap), rot) in enumerate(combos):
        try:
            words = run_single(pdf_bytes, dpi, psm, oem, ds, rot,
                               do_lines, line_min, line_gap, whitelist)
            hits, found, missed = score_words(words, expected)
        except Exception:
            hits, found, missed, words = 0, [], list(expected), []

        results.append({
            "score": hits, "total": len(expected),
            "found": found, "missed": missed,
            "dpi": dpi, "psm": psm, "oem": oem, "despeckle": ds,
            "lines": do_lines, "line_min": line_min, "line_gap": line_gap,
            "rotation": rot,
        })

        with _tune_lock:
            _tune_jobs[job_id]["progress"] = i + 1

    results.sort(key=lambda r: (-r["score"], r["dpi"]))
    best = results[0]

    with _tune_lock:
        _tune_jobs[job_id].update({
            "status": "done",
            "top": results[:10],
            "best": best,
        })


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", version=VERSION,
                           psm_descriptions=PSM_DESCRIPTIONS,
                           oem_descriptions=OEM_DESCRIPTIONS)

@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No PDF uploaded."}), 400
        pdf_bytes = request.files["pdf"].read()
        if not pdf_bytes:
            return jsonify({"error": "Empty file."}), 400

        def gi(k, d):
            try: return int(request.form.get(k, d))
            except: return d
        def gf(k, d):
            try: return float(request.form.get(k, d))
            except: return d

        dpi           = max(72,  min(gi("dpi", 300), 600))
        psm           = max(0,   min(gi("psm", 6),   13))
        oem           = max(0,   min(gi("oem", 1),   3))
        whitelist     = request.form.get("whitelist", "0123456789")
        rotation      = gf("rotation", 0.0)
        despeckle_str = max(1, min(gi("despeckle", 1), 5))
        remove_lines  = request.form.get("remove_lines", "false").lower() == "true"
        line_min_len  = gi("line_min_length", 100)
        line_max_gap  = gi("line_max_gap", 10)

        raw_img = pdf_to_pil(pdf_bytes, dpi)
        proc_img = preprocess(raw_img, rotation, despeckle_str,
                              remove_lines, line_min_len, line_max_gap)
        result = run_tesseract(proc_img, psm, oem, whitelist)
        result["raw_preview"]       = pil_to_data_uri(raw_img)
        result["processed_preview"] = pil_to_data_uri(proc_img)
        result["image_size"]        = {"w": proc_img.width, "h": proc_img.height}
        result["version"]           = VERSION
        return jsonify(result)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/tune/start", methods=["POST"])
def tune_start():
    try:
        if "pdf" not in request.files:
            return jsonify({"error": "No PDF."}), 400
        pdf_bytes = request.files["pdf"].read()
        if not pdf_bytes:
            return jsonify({"error": "Empty PDF."}), 400
        expected_raw = request.form.get("expected", "")
        whitelist    = request.form.get("whitelist", "0123456789")
        expected = [t.strip() for t in expected_raw.replace(",", " ").split() if t.strip()]
        if not expected:
            return jsonify({"error": "Enter at least one expected number."}), 400

        job_id = str(uuid.uuid4())[:8]
        with _tune_lock:
            _tune_jobs[job_id] = {"status": "starting", "progress": 0, "total": 1}

        t = threading.Thread(target=tune_worker,
                             args=(job_id, pdf_bytes, expected, whitelist),
                             daemon=True)
        t.start()
        return jsonify({"job_id": job_id})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/tune/status/<job_id>")
def tune_status(job_id):
    with _tune_lock:
        job = _tune_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job."}), 404
    return jsonify(job)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5200))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
