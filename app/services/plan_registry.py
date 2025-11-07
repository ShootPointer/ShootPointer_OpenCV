# app/services/plan_registry.py
# 목적: 업로드된 영상이 1/2/3번 중 무엇인지 "파이썬 모듈 레지스트리"에서 불러와 식별
#  - JSON 지원 제거 (오직 Python 모듈만)
#  - 매칭 순서: (힌트) → sha256 → duration±tol + (w,h) → pHash

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import os, importlib, hashlib, subprocess, runpy

import cv2  # opencv-python-headless
import numpy as np

from app.core.config import settings

# ── 공통 유틸 ─────────────────────────────────────────────────────
def file_sha256(path: Path, chunk: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def ffprobe_meta(path: Path) -> Tuple[float,int,int]:
    """영상 전체 길이(초), 첫 번째 비디오 스트림의 (width,height)를 ffprobe로 얻는다."""
    p1 = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration",
         "-of","default=noprint_wrappers=1:nokey=1", str(path)],
        stdout=subprocess.PIPE, text=True
    )
    try: dur = float((p1.stdout or "0").strip())
    except: dur = 0.0

    p2 = subprocess.run(
        ["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=width,height",
         "-of","csv=s=x:p=0", str(path)],
        stdout=subprocess.PIPE, text=True
    )
    w, h = 0, 0
    if p2.stdout:
        try: w, h = [int(x) for x in p2.stdout.strip().split("x")]
        except: pass
    return dur, w, h

def _phash(img: np.ndarray, hash_size: int = 8) -> int:
    """DCT 기반 perceptual hash (간단 프레임 지문)."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (hash_size*4, hash_size*4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(img))
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).astype(np.uint8)
    val = 0
    for b in bits.flatten():
        val = (val << 1) | int(b)
    return int(val)

def video_phash_signature(path: Path, samples: int = 5) -> List[int]:
    """영상에서 5개 지점을 샘플해 pHash 시그니처 배열을 만든다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (fps or 30.0)
    ts = [dur * t for t in (0.1, 0.3, 0.5, 0.7, 0.9)] if dur > 0 else [0,1,2,3,4]
    sig: List[int] = []
    for t in ts[:samples]:
        if dur > 0 and fps > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
        ok, frame = cap.read()
        if not ok: break
        sig.append(_phash(frame))
    cap.release()
    return sig

def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))

def phash_distance(sig1: List[int], sig2: List[int]) -> int:
    if not sig1 or not sig2: return 1_000
    m = min(len(sig1), len(sig2))
    if m == 0: return 1_000
    return sum(hamming(sig1[i], sig2[i]) for i in range(m)) // m

# ── 파이썬 레지스트리 로더 ───────────────────────────────────────
def _normalize_registry_payload(namespace: Dict[str, Any]) -> Dict[str, Any]:
    if "PLANS" not in namespace:
        raise ValueError("registry must define PLANS list")
    plans = namespace["PLANS"]
    tol = float(namespace.get("DURATION_TOLERANCE_SEC", 1.5))
    thr = int(namespace.get("PHASH_HAMMING_THRESHOLD", 6))
    return {"plans": plans, "duration_tolerance_sec": tol, "phash_hamming_threshold": thr}


def load_registry_from_py(module_path: str) -> Dict[str, Any]:
    """module_path 예: 'app.data.registry'"""    
    mod = importlib.import_module(module_path)
    payload = {
        key: getattr(mod, key)
        for key in ("PLANS", "DURATION_TOLERANCE_SEC", "PHASH_HAMMING_THRESHOLD")
        if hasattr(mod, key)
    }
    return _normalize_registry_payload(payload)


def load_registry_from_file(file_path: str) -> Dict[str, Any]:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        candidate = (project_root / path).resolve()
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        raise FileNotFoundError(f"registry file not found: {file_path}")
    payload = runpy.run_path(str(path))
    return _normalize_registry_payload(payload)    

def load_registry() -> Dict[str, Any]:
    """
    PLAN_REGISTRY_PY 값이 '.py' 파일 경로나 모듈 경로 모두 가능하도록 로드한다.
    우선순위:
      1) 환경변수 PLAN_REGISTRY_PY
      2) settings.PLAN_REGISTRY_PY
      3) 기본값 'app.data.registry'
    """
    raw = os.getenv("PLAN_REGISTRY_PY", "").strip() or getattr(settings, "PLAN_REGISTRY_PY", "").strip()
    if not raw:
        raw = "app.data.registry"

    looks_like_path = raw.endswith(".py") or "/" in raw or "\\" in raw or raw.startswith(".")
    if looks_like_path:
        try:
            return load_registry_from_file(raw)
        except FileNotFoundError:
            # 파일 경로로 지정했지만 찾을 수 없는 경우만 모듈 시도로 폴백
            pass
    return load_registry_from_py(raw)

# ── 매칭기 ────────────────────────────────────────────────────────
class PlanRegistry:
    def __init__(self, registry: Dict[str, Any]):
        self.obj = registry
        self.plans: List[Dict[str,Any]] = list(self.obj.get("plans", []))
        self.tol = float(self.obj.get("duration_tolerance_sec", 1.5))
        self.phash_thr = int(self.obj.get("phash_hamming_threshold", 6))

    @classmethod
    def from_config(cls) -> "PlanRegistry":
        return cls(load_registry())

    def match(self, video_path: Path, hint: Optional[str]=None) -> Optional[Dict[str,Any]]:
        dur, w, h = ffprobe_meta(video_path)

        # 0) 힌트 우선
        if hint:
            for p in self.plans:
                if str(p.get("id")) == str(hint):
                    return p

        # 1) sha256
        try:
            digest = file_sha256(video_path)
            for p in self.plans:
                s = (p.get("sha256") or "").strip()
                if s and s == digest:
                    return p
        except Exception:
            pass

        # 2) duration±tol + (width,height)
        candidates: List[Dict[str,Any]] = []
        for p in self.plans:
            pd = float(p.get("duration_sec") or 0.0)
            pw, ph = int(p.get("width") or 0), int(p.get("height") or 0)
            dur_ok = (pd == 0.0) or (abs(pd - dur) <= self.tol)
            wh_ok  = (pw == 0 and ph == 0) or (pw == w and ph == h)
            if dur_ok and wh_ok:
                candidates.append(p)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # 3) pHash 최종결정
            sig = video_phash_signature(video_path)
            best, best_d = None, 1_000
            for p in candidates:
                psig = p.get("phash") or []
                d = phash_distance(sig, psig)
                if d < best_d:
                    best, best_d = p, d
            if best and best_d <= self.phash_thr:
                return best

        return None
