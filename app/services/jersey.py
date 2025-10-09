# app/services/jersey.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import subprocess
import re
import tempfile
import shutil
import logging
import time

import cv2
import numpy as np
import pytesseract

from app.core.config import settings
from app.services.ffmpeg import get_duration

logger = logging.getLogger(__name__)

# ─────────────────────────────────────
# CUDA helpers (선택적 가속)
# ─────────────────────────────────────
def _cuda_available() -> bool:
    """OpenCV가 CUDA 빌드이고 USE_CUDA=true이면 True."""
    try:
        return bool(settings.USE_CUDA) and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

def _fast_resize(img: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """CUDA 가능하면 GPU resize, 아니면 CPU resize."""
    if fx == 1.0 and fy == 1.0:
        return img
    if _cuda_available():
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(img)
            new_w = int(round(img.shape[1] * fx))
            new_h = int(round(img.shape[0] * fy))
            gpu_resized = cv2.cuda.resize(gpu, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return gpu_resized.download()
        except Exception:
            # GPU 경로 실패 시 CPU 폴백
            pass
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

def _maybe_downscale_for_cuda(bgr: np.ndarray) -> np.ndarray:
    """
    이미지가 너무 크면 긴 변 기준으로 CUDA_RESIZE_MAX에 맞추어 미리 축소.
    CPU 경로에서도 과도한 메모리/시간을 줄이는 효과.
    """
    try:
        max_side = int(getattr(settings, "CUDA_RESIZE_MAX", 1920))
        if max_side <= 0:
            return bgr
        h, w = bgr.shape[:2]
        long_side = max(h, w)
        if long_side > max_side:
            scale = max_side / float(long_side)
            return _fast_resize(bgr, scale, scale)
    except Exception:
        pass
    return bgr

# ─────────────────────────────────────
# FFmpeg helpers (타임아웃/로깅 포함)
# ─────────────────────────────────────
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """공통 서브프로세스 실행(타임아웃/로깅 포함)"""
    to = settings.FFMPEG_TIMEOUT_SEC
    try:
        logger.debug(f"[jersey.ffproc] exec (timeout={to}s): {' '.join(cmd[:8])} ...")
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=to,
        )
        logger.debug(f"[jersey.ffproc] returncode={p.returncode}")
        return p
    except subprocess.TimeoutExpired:
        logger.warning(f"[jersey.ffproc] timeout after {to}s")
        raise
    except Exception as e:
        logger.exception(f"[jersey.ffproc] failed: {e}")
        raise

# ─────────────────────────────────────
# 프레임 샘플링 (세그먼트 검출에서 사용)
# ─────────────────────────────────────
def _sample_frames_to_dir(video_path: Path, fps: float) -> tuple[Path, list[tuple[float, Path]]]:
    """
    ffmpeg로 일정 fps 간격 프레임을 추출하고 showinfo 로그에서 pts_time(초)을 파싱.
    (옵션) settings.FFMPEG_HWACCEL,* 설정 시 하드웨어 가속 디코딩 사용.
    """
    assert video_path.exists(), f"input not found: {video_path}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="jersey_"))
    out_pattern = tmp_dir / "f_%05d.jpg"

    vf = f"fps={fps},scale='min(960,iw)':-2,showinfo"

    cmd = ["ffmpeg", "-y"]
    # 하드웨어 가속(옵션) — config의 FFMPEG_HWACCEL / FFMPEG_HWACCEL_DEVICE 사용
    if getattr(settings, "FFMPEG_HWACCEL", ""):
        cmd += ["-hwaccel", settings.FFMPEG_HWACCEL]
        if getattr(settings, "FFMPEG_HWACCEL_DEVICE", ""):
            cmd += ["-hwaccel_device", settings.FFMPEG_HWACCEL_DEVICE]

    cmd += [
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", "3",
        str(out_pattern),
    ]

    proc = _run(cmd)

    times: list[float] = []
    for line in (proc.stderr or "").splitlines():
        m = re.search(r"pts_time:([0-9]+\.[0-9]+)", line)
        if m:
            try:
                times.append(float(m.group(1)))
            except Exception:
                pass

    images = sorted(tmp_dir.glob("f_*.jpg"))
    pairs: list[tuple[float, Path]] = []
    for i, img in enumerate(images):
        if i < len(times):
            pairs.append((times[i], img))

    logger.info(f"[jersey.sample] frames={len(pairs)} (fps={fps})")
    return tmp_dir, pairs

# ─────────────────────────────────────
# ROI 탐색(숫자 후보)
# ─────────────────────────────────────
def _digit_roi_candidates(gray: np.ndarray) -> list[np.ndarray]:
    """
    숫자 후보 ROI 추출: 블러 → 적응형 이진화 → 모폴로지 → 컨투어 필터.
    상단 HUD(스코어보드) 오검을 줄이기 위해 화면 상단 일부는 제외.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois: list[np.ndarray] = []
    H, W = gray.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 🔧 잡음 억제: 최소 면적 상향(400 → 700)
        if w * h < 700:
            continue
        ar = w / (h + 1e-6)
        if ar < 0.3 or ar > 6.0:
            continue
        if y < H * 0.1:  # 화면 상단 10%는 HUD일 확률↑ → 제외(필요시 0.15~0.2로 조절)
            continue
        x0 = max(0, x - 2)
        y0 = max(0, y - 2)
        x1 = min(W, x + w + 2)
        y1 = min(H, y + h + 2)
        roi = gray[y0:y1, x0:x1]
        # 🔧 끊긴 획 보정: ROI에 close 한 번 더
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        rois.append(roi)
    return rois

# ─────────────────────────────────────
# OCR 유틸 (강화 루틴)
# ─────────────────────────────────────
def _final_conf(digits: str, conf_vals: list[float]) -> float:
    """
    OCR conf가 누락/음수(-1)인 경우를 위한 보정 규칙.
    - conf_vals 있으면 평균을 그대로 사용
    - 없으면 자릿수별 기본값(단일자:0.85, 두 자리:0.90, 그 외:0.80)
    """
    if conf_vals:
        return float(np.mean(conf_vals))
    n = len(digits)
    if n == 1:
        return 0.85
    if n == 2:
        return 0.90
    return 0.80

def _tess_digits_with_conf(img: np.ndarray, psm: int) -> tuple[str, float]:
    """
    pytesseract image_to_data로 숫자와 평균 confidence(0~1)를 얻는다.
    settings.OCR_TIMEOUT_SEC 적용 + conf 보정.
    """
    config = (
        f"-l eng --oem {settings.JERSEY_TESSERACT_OEM} "
        f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    )
    try:
        data = pytesseract.image_to_data(
            img, config=config, output_type=pytesseract.Output.DICT, timeout=settings.OCR_TIMEOUT_SEC
        )
    except RuntimeError as e:
        logger.warning(f"[jersey.ocr] timeout after {settings.OCR_TIMEOUT_SEC}s: {e}")
        return "", 0.0
    except Exception as e:
        logger.exception(f"[jersey.ocr] failed: {e}")
        return "", 0.0

    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []
    digits = ""
    conf_vals: list[float] = []

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

def _prep_variants(bgr: np.ndarray, invert: bool) -> list[np.ndarray]:
    """
    다양한 전처리 변형: 그레이, CLAHE, Otsu, Adaptive, Morph
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)

    outs = [gray]

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    outs.append(clahe)

    # Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    outs.append(otsu)

    # Adaptive
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    outs.append(adap)

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    outs.append(mor)

    return outs

def _best_digits_with_hint(digs: list[tuple[str, float]], expected: str | None) -> tuple[str, float]:
    if not digs:
        return "", 0.0
    if expected:
        def _score(cand: tuple[str, float]):
            d, c = cand
            # 정확일치(2) > 부분일치(1) > 불일치(0), 동률이면 conf, 길이차이 작은 쪽 우선
            pri = 2 if d == expected else (1 if expected in d else 0)
            return (pri, c, -abs(len(d) - len(expected)))
        digs = sorted(digs, key=_score, reverse=True)
        return digs[0]
    # 힌트 없으면: 길이 우선 → conf
    digs = sorted(digs, key=lambda x: (len(x[0]), x[1]), reverse=True)
    return digs[0]

def _ocr_try_all(bgr: np.ndarray, expected: str | None = None) -> tuple[str, float, Dict[str, Any]]:
    """
    멀티 스케일 × 반전 × 전처리 × PSM 조합으로 최적 결과 탐색.
    - 예산(콤보 수/총 소요시간) 초과 시 즉시 중단
    - 힌트(expected)와 정확히 일치하면 즉시 단락(short-circuit)
    """
    # 너무 크면 미리 축소(속도/메모리)
    bgr = _maybe_downscale_for_cuda(bgr)

    H, W = bgr.shape[:2]
    tried = 0
    candidates: list[tuple[str, float]] = []

    scales = getattr(settings, "OCR_SCALES", [1.0]) or [1.0]
    psms = getattr(settings, "OCR_PSMS", [7, 6, 10]) or [7, 6, 10]
    invert_opts = [False, True] if getattr(settings, "OCR_TRY_INVERT", True) else [False]

    # 🔧 예산(최대 조합/최대 시간) — .env에 없으면 기본값 사용
    max_combos = int(getattr(settings, "OCR_MAX_COMBOS", 120))
    max_sec = float(getattr(settings, "OCR_MAX_SEC", 8.0))
    t0 = time.perf_counter()

    def over_budget() -> bool:
        return (tried >= max_combos) or ((time.perf_counter() - t0) >= max_sec)

    for s in scales:
        scaled = _fast_resize(bgr, s, s) if s != 1.0 else bgr
        for inv in invert_opts:
            variants = _prep_variants(scaled, invert=inv)

            # ① 전체 이미지
            for v in variants:
                for psm in psms:
                    tried += 1
                    digs, cf = _tess_digits_with_conf(v, psm=psm)
                    digs = re.sub(r"\D+", "", digs)
                    if digs:
                        candidates.append((digs, cf))
                        if expected and digs == expected:
                            # 예상값 정확 일치 → 즉시 반환
                            return digs, cf, {
                                "imageWidth": W, "imageHeight": H,
                                "triedCombos": tried, "numCandidates": len(candidates),
                                "shortCircuit": "expected_match"
                            }
                    if over_budget():
                        digits, conf = _best_digits_with_hint(candidates, expected)
                        dbg = {
                            "imageWidth": W, "imageHeight": H,
                            "triedCombos": tried, "numCandidates": len(candidates),
                            "budgetStop": True
                        }
                        return digits, float(conf), dbg

            # ② ROI들
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            if inv:
                gray = cv2.bitwise_not(gray)
            rois = _digit_roi_candidates(gray)
            for roi in rois:
                for fx in (1.5, 2.0, 3.0):
                    roi_big = _fast_resize(roi, fx, fx)
                    for psm in psms:
                        tried += 1
                        digs, cf = _tess_digits_with_conf(roi_big, psm=psm)
                        digs = re.sub(r"\D+", "", digs)
                        if digs:
                            candidates.append((digs, cf))
                            if expected and digs == expected:
                                return digs, cf, {
                                    "imageWidth": W, "imageHeight": H,
                                    "triedCombos": tried, "numCandidates": len(candidates),
                                    "shortCircuit": "expected_match"
                                }
                        if over_budget():
                            digits, conf = _best_digits_with_hint(candidates, expected)
                            dbg = {
                                "imageWidth": W, "imageHeight": H,
                                "triedCombos": tried, "numCandidates": len(candidates),
                                "budgetStop": True
                            }
                            return digits, float(conf), dbg

    digits, conf = _best_digits_with_hint(candidates, expected)
    debug: Dict[str, Any] = {
        "imageWidth": W, "imageHeight": H,
        "triedCombos": tried, "numCandidates": len(candidates)
    }
    if getattr(settings, "DEBUG_OCR", False):
        top = sorted(candidates, key=lambda x: (len(x[0]), x[1]), reverse=True)[:5]
        debug["topCandidates"] = [{"digits": d, "conf": round(c, 3)} for d, c in top]
    return digits, float(conf), debug

# ✅ 힌트 지원 공개 API (라우터에서 backNumber 힌트 사용)
def ocr_jersey_image_bytes_with_hint(image_bytes: bytes, expected: str | None) -> tuple[str, float]:
    if not image_bytes:
        return "", 0.0
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return "", 0.0
    bgr = _maybe_downscale_for_cuda(bgr)
    digits, conf, _ = _ocr_try_all(bgr, expected=expected)
    return digits, conf

# ─────────────────────────────────────
# 공개 OCR API
# ─────────────────────────────────────
def ocr_jersey_image_bytes(image_bytes: bytes) -> tuple[str, float]:
    """
    업로드된 등번호 '이미지'에서 숫자만 추출.
    반환: (digits, conf) — conf는 0.0~1.0
    (기존 라우터와의 호환을 위해 2-튜플 유지)
    """
    if not image_bytes:
        return "", 0.0
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return "", 0.0
    bgr = _maybe_downscale_for_cuda(bgr)
    digits, conf, _dbg = _ocr_try_all(bgr)
    return digits, conf

def ocr_jersey_image_bytes_debug(image_bytes: bytes) -> tuple[str, float, Dict[str, Any]]:
    """
    디버그 정보까지 필요한 경우 사용.
    반환: (digits, conf, debugDict)
    """
    if not image_bytes:
        return "", 0.0, {"reason": "empty"}
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return "", 0.0, {"reason": "decode_failed"}
    bgr = _maybe_downscale_for_cuda(bgr)
    return _ocr_try_all(bgr)

# ─────────────────────────────────────
# time utils
# ─────────────────────────────────────
def _merge_times(times: list[float], max_gap: float) -> list[tuple[float, float]]:
    """검출 시점들을 인접 간격(max_gap) 기준으로 [start,end] 구간으로 묶기."""
    if not times:
        return []
    times = sorted(times)
    segs: list[tuple[float, float]] = []
    s = e = times[0]
    for t in times[1:]:
        if t - e <= max_gap:
            e = t
        else:
            segs.append((s, e))
            s = e = t
    segs.append((s, e))
    return segs

def _expand_and_filter(
    segs: list[tuple[float, float]],
    pad: float,
    min_dur: float,
    total: float,
) -> list[tuple[float, float]]:
    """구간 앞뒤 pad 확장 후, 너무 짧은 구간 제거 + [0,total] 범위 클램프."""
    out: list[tuple[float, float]] = []
    for s, e in segs:
        s2 = max(0.0, s - pad)
        e2 = min(total, e + pad)
        if e2 - s2 >= min_dur:
            out.append((s2, e2))
    return out

# ─────────────────────────────────────
# main API: 세그먼트 검출
# ─────────────────────────────────────
def detect_player_segments(
    video_path: Path, jersey_number: int
) -> List[Tuple[float, float]]:
    """
    간단한 등번호 감지 파이프라인:
      1) fps로 프레임 샘플링 (ffmpeg)
      2) 숫자 후보 ROI 추출(OpenCV) → 강화된 Tesseract OCR(숫자만)
      3) jersey_number와 일치하는 프레임의 시점들을 병합해 구간 반환
    """
    assert video_path.exists(), f"input not found: {video_path}"

    fps = settings.JERSEY_SAMPLE_FPS
    min_seg = settings.JERSEY_MIN_SEG_DUR
    merge_gap = settings.JERSEY_MERGE_GAP
    num_conf = settings.JERSEY_NUM_CONF

    tmp_dir: Optional[Path] = None
    try:
        tmp_dir, frames = _sample_frames_to_dir(video_path, fps=fps)
        total = get_duration(video_path)
        target = str(jersey_number)

        hit_times: list[float] = []
        for t, img_path in frames:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 후보 ROI들에 대해 빠른 경로로 OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rois = _digit_roi_candidates(gray)
            found = False

            # ROI 우선(속도): 각 ROI 2배/3배 × 빠른 PSM 세트
            fast_psms = settings.OCR_PSMS or [7, 6, 10]
            for roi in rois:
                for fx in (2.0, 3.0):
                    roi_big = _fast_resize(roi, fx, fx)
                    for psm in fast_psms:
                        digs, cf = _tess_digits_with_conf(roi_big, psm=psm)
                        digs = re.sub(r"\D+", "", digs)
                        if digs and target in digs and cf >= num_conf:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

            # ROI에서 못 찾으면, 이미지 전체를 보조 탐색(예산 내)
            if not found:
                img_small = _maybe_downscale_for_cuda(img)
                digits, cf, _ = _ocr_try_all(img_small, expected=target)
                if digits and target in digits and cf >= num_conf:
                    found = True

            if found:
                hit_times.append(t)

        # 시점 → 구간 병합, 확장, 최소 길이 필터
        segs = _merge_times(hit_times, max_gap=merge_gap)
        segs = _expand_and_filter(segs, pad=0.5, min_dur=min_seg, total=total)
        logger.info(f"[jersey.segs] jersey={jersey_number} hits={len(hit_times)} segments={len(segs)}")
        return segs

    finally:
        # 임시 프레임 폴더 정리
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
