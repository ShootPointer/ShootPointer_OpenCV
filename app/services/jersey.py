# app/services/jersey.py
from __future__ import annotations

import re
import logging
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
import pytesseract

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────
# OCR helpers
# ─────────────────────────────────────
def _final_conf(digits: str, conf_vals: List[float]) -> float:
    """Tesseract conf 보정: 값이 없으면 자릿수 기반 기본치."""
    if conf_vals:
        return float(np.mean(conf_vals))
    n = len(digits)
    if n == 1:
        return 0.85
    if n == 2:
        return 0.90
    return 0.80

def _tess_digits_with_conf(img: np.ndarray, psm: int) -> Tuple[str, float]:
    """image_to_data로 숫자+평균 confidence(0~1) 산출."""
    config = (
        f"-l eng --oem {settings.JERSEY_TESSERACT_OEM} "
        f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    )
    try:
        data = pytesseract.image_to_data(
            img,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=settings.OCR_TIMEOUT_SEC,
        )
    except RuntimeError as e:
        logger.warning(f"[ocr] timeout after {settings.OCR_TIMEOUT_SEC}s: {e}")
        return "", 0.0
    except Exception as e:
        logger.exception(f"[ocr] failed: {e}")
        return "", 0.0

    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []
    digits = ""
    conf_vals: List[float] = []

    for t, c in zip(texts, confs):
        t = t or ""
        only = re.sub(r"\D+", "", t)
        if only:
            digits += only
            try:
                ci = float(c)
                if ci >= 0:
                    conf_vals.append(ci / 100.0)
            except Exception:
                pass

    if not digits:
        return "", 0.0
    conf = _final_conf(digits, conf_vals)
    return digits, conf

def _prep_variants(bgr: np.ndarray, invert: bool) -> List[np.ndarray]:
    """그레이/CLAHE/Otsu/Adaptive/Morph 변형 세트."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)

    outs = [gray]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    outs.append(clahe)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    outs.append(otsu)

    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    outs.append(adap)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    outs.append(mor)

    return outs

def _best_digits_with_hint(cands: List[Tuple[str, float]], expected: str | None) -> Tuple[str, float]:
    if not cands:
        return "", 0.0
    if expected:
        def _score(c: Tuple[str, float]):
            d, conf = c
            pri = 2 if d == expected else (1 if expected in d else 0)
            return (pri, conf, -abs(len(d) - len(expected)))
        cands.sort(key=_score, reverse=True)
        return cands[0]
    cands.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
    return cands[0]

def _ocr_try_all(bgr: np.ndarray, expected: str | None = None) -> Tuple[str, float, Dict[str, Any]]:
    """스케일×반전×전처리×PSM 조합으로 최적 숫자 탐색."""
    H, W = bgr.shape[:2]
    tried = 0
    cands: List[Tuple[str, float]] = []

    scales = getattr(settings, "OCR_SCALES", [1.0]) or [1.0]
    psms = getattr(settings, "OCR_PSMS", [7, 6, 10]) or [7, 6, 10]
    invert_opts = [False, True] if getattr(settings, "OCR_TRY_INVERT", True) else [False]

    max_combos = int(getattr(settings, "OCR_MAX_COMBOS", 120))
    max_sec = float(getattr(settings, "OCR_MAX_SEC", 8.0))
    t0 = cv2.getTickCount()
    tick = cv2.getTickFrequency()

    def over_budget() -> bool:
        return tried >= max_combos or (cv2.getTickCount() - t0) / tick >= max_sec

    for s in scales:
        scaled = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s != 1.0 else bgr
        for inv in invert_opts:
            for v in _prep_variants(scaled, invert=inv):
                for psm in psms:
                    tried += 1
                    digs, cf = _tess_digits_with_conf(v, psm)
                    digs = re.sub(r"\D+", "", digs)
                    if digs:
                        cands.append((digs, cf))
                        if expected and digs == expected:
                            return digs, cf, {"imageWidth": W, "imageHeight": H, "triedCombos": tried, "shortCircuit": "expected_match"}
                    if over_budget():
                        d, c = _best_digits_with_hint(cands, expected)
                        return d, float(c), {"imageWidth": W, "imageHeight": H, "triedCombos": tried, "budgetStop": True}

    d, c = _best_digits_with_hint(cands, expected)
    debug: Dict[str, Any] = {"imageWidth": W, "imageHeight": H, "triedCombos": tried}
    if getattr(settings, "DEBUG_OCR", False) and cands:
        top = sorted(cands, key=lambda x: (len(x[0]), x[1]), reverse=True)[:5]
        debug["topCandidates"] = [{"digits": dd, "conf": round(cc, 3)} for dd, cc in top]
    return d, float(c), debug

# ─────────────────────────────────────
# 공개 API (라우터에서 사용)
# ─────────────────────────────────────
def ocr_jersey_image_bytes(image_bytes: bytes) -> Tuple[str, float]:
    if not image_bytes:
        return "", 0.0
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return "", 0.0
    digits, conf, _ = _ocr_try_all(bgr)
    return digits, conf

def ocr_jersey_image_bytes_with_hint(image_bytes: bytes, expected: str | None) -> Tuple[str, float]:
    if not image_bytes:
        return "", 0.0
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return "", 0.0
    digits, conf, _ = _ocr_try_all(bgr, expected=expected)
    return digits, conf
