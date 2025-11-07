"""Utilities for generating AI demo highlight clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.bh_edit import cut_and_overlay, concat_videos, ffprobe_duration
from app.services.media_tags import annotate_inplace
from app.services.pipeline import build_zip_in_memory, save_clip_as_uuid
from app.services.plan_registry import PlanRegistry


class AIHighlightError(RuntimeError):
    """Raised when AI highlight generation fails."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AIHighlightResult:
    base_name: str
    plan_id: str
    counts: Dict[str, int]
    segments: List[Dict[str, str | float | int]]
    public_urls: List[str]
    clips: List[Path]
    summary_path: Path
    merged_url: Optional[str]

    def build_zip(self):
        """Return an in-memory ZIP buffer of the generated clips."""
        return build_zip_in_memory(self.clips)

    @property
    def summary_url(self) -> str:
        base = settings.STATIC_BASE_URL.rstrip("/")
        rel = self.summary_path.relative_to(settings.SAVE_ROOT)
        return f"{base}/{rel.as_posix()}"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def find_latest_original(member_id: str, highlight_key: str) -> Optional[Tuple[Path, str]]:
    root = Path(settings.SAVE_ROOT) / member_id / highlight_key
    candidates = sorted(root.glob("*/original_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    p = candidates[0]
    return p, p.parent.name


class AISelector:
    def __init__(self, plan: dict, tag: str = "AI-Selector"):
        self.plan = plan
        self.tag = tag

    def select(self) -> List[Tuple[float, float, str]]:
        segments = self.plan.get("segments", [])
        return [(float(a), float(b), str(lbl)) for (a, b, lbl) in segments]


def _prepare_segments(
    src_path: Path,
    base_name: str,
    member_id: str,
    highlight_key: str,
    overlay_model_tag: str,
    merge_output: bool,
    video_code: Optional[str] = None,
) -> AIHighlightResult:
    reg = PlanRegistry.from_config()
    plan = reg.match(src_path, hint=video_code)
    if not plan:
        raise AIHighlightError("no matching plan (check registry configuration)", status_code=404)

    video_id = str(plan.get("id", "X"))
    selector = AISelector(plan, tag=overlay_model_tag)
    segments = selector.select()
    if not segments:
        raise AIHighlightError("plan has empty segments")

    duration = ffprobe_duration(str(src_path))
    work_dir = Path("/tmp/uploads")
    work_dir.mkdir(parents=True, exist_ok=True)

    counts = {"2PT": 0, "3PT": 0}
    manifest: List[Dict[str, str | float | int]] = []
    public_urls: List[str] = []
    clips: List[Path] = []

    for i, (start, end, label) in enumerate(segments, start=1):
        start = max(0.0, start)
        if duration and duration > 0:
            end = min(end, duration)
        if end - start < 0.3:
            continue

        tmp_out = work_dir / f"{base_name}_seg{i:02d}_{label}_s{int(start * 1000)}_e{int(end * 1000)}.mp4"
        overlay_txt = f"[{overlay_model_tag}] {label}"
        cut_and_overlay(str(src_path), start, end, overlay_txt, str(tmp_out))

        points = 2 if label == "2PT" else (3 if label == "3PT" else 1)
        annotate_inplace(str(tmp_out), {
            "created_at": _now_iso(),
            "points": str(points),
        })

        if label == "2PT":
            counts["2PT"] += 1
        if label == "3PT":
            counts["3PT"] += 1

        dst_path, url = save_clip_as_uuid(tmp_out, member_id, highlight_key, subdir="shorts", prefix="short_")
        clips.append(dst_path)
        public_urls.append(url)

        manifest.append({
            "file": dst_path.name,
            "clip_type": label,
            "video_id": video_id,
            "segment_index": i,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "url": url,
        })

    if not clips:
        raise AIHighlightError("no valid segments")

    merged_url: Optional[str] = None
    if merge_output:
        merged_tmp = work_dir / f"{base_name}_merged_ai_demo.mp4"
        concat_videos([str(p) for p in clips], str(merged_tmp))
        annotate_inplace(str(merged_tmp), {"created_at": _now_iso()})
        merged_dst, merged_url = save_clip_as_uuid(merged_tmp, member_id, highlight_key, subdir="shorts", prefix="merged_")
        clips.append(merged_dst)
        public_urls.append(merged_url)

    dst_dir = Path(settings.SAVE_ROOT) / member_id / highlight_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "video_id": video_id,
        "member_id": member_id,
        "highlight_key": highlight_key,
        "counts": counts,
        "segments": manifest,
        "merged": merged_url,
    }
    summary_path = dst_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return AIHighlightResult(
        base_name=base_name,
        plan_id=video_id,
        counts=counts,
        segments=manifest,
        public_urls=public_urls,
        clips=clips,
        summary_path=summary_path,
        merged_url=merged_url,
    )


def run_ai_demo_for_existing(
    member_id: str,
    highlight_key: str,
    *,
    overlay_model_tag: str = "AI-Selector",
    merge_output: bool = True,
    video_code: Optional[str] = None,
) -> AIHighlightResult:
    found = find_latest_original(member_id, highlight_key)
    if not found:
        raise AIHighlightError("no original_*.mp4 under the highlightKey", status_code=404)
    src_path, _ = found
    base_name = src_path.stem
    return _prepare_segments(src_path, base_name, member_id, highlight_key, overlay_model_tag, merge_output, video_code)


def run_ai_demo_from_path(
    src_path: Path,
    *,
    member_id: str,
    highlight_key: str,
    base_name: Optional[str] = None,
    overlay_model_tag: str = "AI-Selector",
    merge_output: bool = True,
    video_code: Optional[str] = None,
) -> AIHighlightResult:
    if base_name is None:
        base_name = Path(src_path).stem
    return _prepare_segments(Path(src_path), base_name, member_id, highlight_key, overlay_model_tag, merge_output, video_code)