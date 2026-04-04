"""
tune.py — automated parameter search for OCR p1
Usage:
    .venv\\Scripts\\python tune.py path\\to\\sheet.pdf

Ground-truth numbers are set in EXPECTED below.
The script tries every parameter combination, scores each by how many
expected numbers it recovers, and prints a ranked results table.
"""
import sys
import time
import itertools
from collections import Counter
from pathlib import Path

# ── ground truth ──────────────────────────────────────────────────────────────
# Edit this list to match the numbers on your sheet.
# Duplicates matter: if 403 appears twice, list it twice.
EXPECTED = ["403", "401", "404", "403", "408"]

# ── search grid ───────────────────────────────────────────────────────────────
GRID = {
    "dpi":            [200, 300, 400],
    "psm":            [6, 7, 11, 12, 13],
    "oem":            [1, 3],
    "despeckle":      [1, 2, 3],
    # (remove_lines, min_length, max_gap)
    "line_removal":   [
        (False, 0,   0),
        (True,  50,  5),
        (True,  80,  10),
        (True,  100, 10),
        (True,  150, 15),
        (True,  200, 20),
    ],
    "rotation":       [0],          # add e.g. 0.5, -0.5 if sheet is slightly skewed
}

WHITELIST = "0123456789"

# ── imports (same stack as app.py) ────────────────────────────────────────────
try:
    import cv2
    import fitz
    import numpy as np
    import pytesseract
    from PIL import Image
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
except ImportError as e:
    sys.exit(f"Missing package: {e}\nActivate the venv first.")


# ── pipeline (mirrors app.py) ─────────────────────────────────────────────────

def pdf_to_pil(pdf_bytes, dpi):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def rotate_image(img, angle):
    if angle == 0:
        return img
    return img.rotate(-angle, expand=True, fillcolor=(255, 255, 255))

def despeckle(arr, strength):
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

def run(pdf_bytes, dpi, psm, oem, despeckle_str, rotation,
        do_lines, line_min, line_gap):
    img = pdf_to_pil(pdf_bytes, dpi)
    img = rotate_image(img, rotation)
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if do_lines:
        arr = remove_long_lines(arr, line_min, line_gap)
    if despeckle_str > 1:
        arr = despeckle(arr, despeckle_str)
    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    cfg = f"--psm {psm} --oem {oem} -c tessedit_char_whitelist={WHITELIST}"
    data = pytesseract.image_to_data(img, config=cfg,
                                     output_type=pytesseract.Output.DICT)
    words = [w.strip() for w in data["text"] if w.strip()]
    return words


# ── scoring ───────────────────────────────────────────────────────────────────

def score(found_words, expected):
    """
    Returns (hit_count, total, found_list, missed_list).
    Uses multiset matching so duplicate expectations must each be matched.
    """
    need  = Counter(expected)
    have  = Counter(found_words)
    hits  = 0
    found = []
    missed = []
    for num, count in need.items():
        got = min(have.get(num, 0), count)
        hits += got
        found  += [num] * got
        missed += [num] * (count - got)
    return hits, len(expected), found, missed


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python tune.py path/to/sheet.pdf")

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        sys.exit(f"File not found: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    print(f"\nPDF      : {pdf_path.name}")
    print(f"Expected : {EXPECTED}")
    print(f"Total    : {len(EXPECTED)} numbers\n")

    # Build all combinations
    combos = list(itertools.product(
        GRID["dpi"],
        GRID["psm"],
        GRID["oem"],
        GRID["despeckle"],
        GRID["line_removal"],
        GRID["rotation"],
    ))
    total = len(combos)
    print(f"Trying {total} combinations…\n")

    results = []
    t0 = time.time()

    for i, (dpi, psm, oem, ds, (do_lines, line_min, line_gap), rot) in enumerate(combos):
        pct = (i + 1) / total * 100
        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"\r[{i+1:4d}/{total}] {pct:5.1f}%  ETA {eta:5.0f}s", end="", flush=True)

        try:
            words = run(pdf_bytes, dpi, psm, oem, ds, rot, do_lines, line_min, line_gap)
            hits, n, found, missed = score(words, EXPECTED)
        except Exception as e:
            hits, n, found, missed, words = 0, len(EXPECTED), [], EXPECTED, []

        results.append({
            "score":      hits,
            "total":      n,
            "found":      found,
            "missed":     missed,
            "all_words":  words,
            "dpi":        dpi,
            "psm":        psm,
            "oem":        oem,
            "despeckle":  ds,
            "lines":      do_lines,
            "line_min":   line_min,
            "line_gap":   line_gap,
            "rotation":   rot,
        })

    print(f"\r{' '*60}\r", end="")  # clear progress line

    # Sort best first, then by fewest spurious words (noise)
    results.sort(key=lambda r: (-r["score"], len(r["all_words"])))

    # ── print top 20 ─────────────────────────────────────────────────────────
    print(f"\n{'─'*100}")
    print(f"{'RANK':<5} {'SCORE':<7} {'DPI':<5} {'PSM':<5} {'OEM':<5} "
          f"{'DESP':<6} {'LINES':<6} {'MINLEN':<8} {'GAP':<5} {'ROT':<6} "
          f"{'FOUND':<25} {'MISSED'}")
    print(f"{'─'*100}")

    for rank, r in enumerate(results[:20], 1):
        lines_str = f"Y/{r['line_min']}/{r['line_gap']}" if r["lines"] else "N"
        found_str  = " ".join(r["found"])  or "—"
        missed_str = " ".join(r["missed"]) or "—"
        pct = r["score"] / r["total"] * 100
        print(f"{rank:<5} {r['score']}/{r['total']} {pct:4.0f}%  "
              f"{r['dpi']:<5} {r['psm']:<5} {r['oem']:<5} "
              f"{r['despeckle']:<6} {lines_str:<10} {r['rotation']:<6} "
              f"{found_str:<25} {missed_str}")

    # ── best result detail ────────────────────────────────────────────────────
    best = results[0]
    print(f"\n{'═'*60}")
    print(f"  BEST RESULT  —  {best['score']}/{best['total']} numbers found")
    print(f"{'═'*60}")
    print(f"  DPI          : {best['dpi']}")
    print(f"  PSM          : {best['psm']}")
    print(f"  OEM          : {best['oem']}")
    print(f"  Despeckle    : {best['despeckle']}")
    print(f"  Line removal : {'Yes' if best['lines'] else 'No'}"
          + (f"  (min {best['line_min']} px, gap {best['line_gap']} px)"
             if best["lines"] else ""))
    print(f"  Rotation     : {best['rotation']}°")
    print(f"  Found        : {best['found']}")
    print(f"  Missed       : {best['missed'] or 'none'}")
    print(f"  All words    : {best['all_words']}")
    print(f"{'═'*60}")
    print(f"\nTotal time: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
