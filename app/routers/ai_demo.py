# app/routers/ai_demo.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import json
import glob

from app.core.config import settings
from app.routers.highlight import _save_upload
from app.services.bh_edit import cut_and_overlay, concat_videos, ffprobe_duration
from app.services.pipeline import build_zip_in_memory, save_clip_as_uuid
from app.services.plan_registry import PlanRegistry
from app.services.media_tags import annotate_inplace

router = APIRouter()

def _now_iso() -> str:
    """메타데이터용 LocalDateTime(초 단위)"""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def _find_latest_original(member_id: str, highlight_key: str) -> Optional[Tuple[Path, str]]:
    """
    SAVE_ROOT/{member}/{highlightKey}/*/original_*.mp4 중 최신 파일을 찾아 반환.
    return: (original_path, ldt_folder) 또는 None
    (※ 원본은 LDT 폴더 구조 유지)
    """
    root = Path(settings.SAVE_ROOT) / member_id / highlight_key
    pattern = str(root / "*" / "original_*.mp4")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    p = Path(candidates[0])
    return p, p.parent.name  # (원본 경로, LDT 폴더명)

class AISelector:
    """겉모습만 AI처럼: 실제로는 레지스트리의 segments를 그대로 사용"""
    def __init__(self, plan: dict, tag: str = "AI-Selector"):
        self.plan = plan
        self.tag = tag
    def select(self) -> List[Tuple[float, float, str]]:
        segs = self.plan.get("segments", [])
        return [(float(a), float(b), str(lbl)) for (a, b, lbl) in segs]

@router.post("/highlight/ai-demo")
async def highlight_ai_demo(
    # 업로드 모드: 파일 업로드. useExisting=True면 생략 가능
    video: Optional[UploadFile] = File(default=None, description="데모 영상 1개(업로드 모드일 때만 필요)"),
    useExisting: bool = Form(False, description="이미 저장된 original_*.mp4를 사용"),
    videoCode: Optional[str] = Form(default=None, description="선택: 힌트(1|2|3) – 없어도 자동 식별"),
    overlay_model_tag: str = Form("AI-Selector"),
    merge_output: bool = Form(True, description="개별 클립들을 하나로 이어 붙일지"),
    memberId: str = Form(..., description="저장 멤버 ID (필수)"),
    highlightKey: str = Form(..., description="저장 하이라이트 키 (필수)"),
    returnZip: bool = Form(True, description="ZIP 스트림으로 반환(아니면 URL JSON)"),
):
    """
    데모용 하이라이트 생성:
    - PlanRegistry로 영상 자동 식별 → segments대로 컷팅
    - 결과 저장: SAVE_ROOT/{memberId}/{highlightKey}/shorts/short_<uuid>.mp4
    - summary.json: SAVE_ROOT/{memberId}/{highlightKey}/summary.json
    - MP4 메타: { created_at, points } 만 기록
    """
    # 0) 입력/소스 결정
    cleanup_needed = False
    if useExisting:
        found = _find_latest_original(memberId, highlightKey)
        if not found:
            return JSONResponse(status_code=404, content={"error": "no original_*.mp4 under the highlightKey"})
        src_path, _ldt_folder = found  # 원본은 LDT 폴더에 있지만, 숏츠 저장엔 LDT 미사용
        tmp_in = src_path
        base_name = src_path.stem
    else:
        if not video:
            return JSONResponse(status_code=400, content={"error": "upload mode requires 'video' file"})
        tmp_in = _save_upload(video)  # /tmp 에 임시 저장
        cleanup_needed = True
        base_name = Path(video.filename or tmp_in.stem).stem

    clips: List[Path] = []
    merged: Optional[Path] = None

    try:
        # 1) 자동 식별 (파이썬 레지스트리만 사용)
        reg = PlanRegistry.from_config()
        plan = reg.match(tmp_in, hint=videoCode)
        if not plan:
            return JSONResponse(
                status_code=404,
                content={"error": "no matching plan (check duration/size/sha256/phash in registry.py)"}
            )
        video_id = str(plan.get("id", "X"))

        selector = AISelector(plan, tag=overlay_model_tag)
        segments = selector.select()
        if not segments:
            return JSONResponse(status_code=400, content={"error": "plan has empty segments"})

        # 2) 컷 생성 + 메타데이터 태깅(최소) + 집계
        duration = ffprobe_duration(str(tmp_in))
        work_dir = Path("/tmp/uploads"); work_dir.mkdir(parents=True, exist_ok=True)

        counts = {"2PT": 0, "3PT": 0}
        manifest: List[Dict[str, str | float | int]] = []
        public_urls: List[str] = []

        for i, (s, e, lbl) in enumerate(segments, start=1):
            s2 = max(0.0, s)
            e2 = min(e, duration) if duration and duration > 0 else e
            if e2 - s2 < 0.3:
                continue

            # 임시 출력 파일(워킹 디렉터리)
            tmp_out = work_dir / f"{base_name}_seg{i:02d}_{lbl}_s{int(s2*1000)}_e{int(e2*1000)}.mp4"

            # 오버레이(시간 제거): "[AI-Selector] 2PT"
            overlay_txt = f"[{overlay_model_tag}] {lbl}"
            cut_and_overlay(str(tmp_in), s2, e2, overlay_txt, str(tmp_out))

            # 점수 계산 (FT는 1로 표기 예시)
            points = 2 if lbl == "2PT" else (3 if lbl == "3PT" else 1)
            # ✅ 메타데이터 최소(생성시각/점수만)
            annotate_inplace(str(tmp_out), {
                "created_at": _now_iso(),
                "points": str(points),
            })

            if lbl == "2PT": counts["2PT"] += 1
            if lbl == "3PT": counts["3PT"] += 1

            # ✅ 최종 저장: …/{member}/{highlightKey}/shorts/short_<uuid>.mp4
            dst_path, url = save_clip_as_uuid(tmp_out, memberId, highlightKey, subdir="shorts", prefix="short_")
            clips.append(dst_path)
            public_urls.append(url)

            # summary.json용 매니페스트(백엔드 파싱 편의를 위해 유지)
            manifest.append({
                "file": dst_path.name,
                "clip_type": lbl,         # 필요 없으면 제거 가능
                "video_id": video_id,     # 필요 없으면 제거 가능
                "segment_index": i,
                "start_sec": round(s2, 3),
                "end_sec": round(e2, 3),
                "url": url
            })

        if not clips:
            return JSONResponse(status_code=400, content={"error": "no valid segments"})

        # 3) 병합본(옵션) — 같은 shorts 폴더에 저장
        merged_url = None
        if merge_output and clips:
            merged_tmp = work_dir / f"{base_name}_merged_ai_demo.mp4"
            concat_videos([str(p) for p in clips], str(merged_tmp))
            # 병합본 메타는 created_at만(원하면 생략 가능)
            annotate_inplace(str(merged_tmp), {
                "created_at": _now_iso(),
            })
            merged_dst, merged_url = save_clip_as_uuid(
                merged_tmp, memberId, highlightKey, subdir="shorts", prefix="merged_"
            )
            clips.append(merged_dst)
            public_urls.append(merged_url)

        # 4) summary.json — SAVE_ROOT/{member}/{highlightKey}/summary.json
        dst_dir = Path(settings.SAVE_ROOT) / memberId / highlightKey
        dst_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "video_id": video_id,
            "member_id": memberId,
            "highlight_key": highlightKey,
            "counts": counts,         # {"2PT": n, "3PT": m}
            "segments": manifest,     # 각 클립 상세(+url)
            "merged": merged_url
        }
        (dst_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        # 5) 응답
        if returnZip:
            buf = build_zip_in_memory(clips)
            fname = f"{base_name}_{video_id}_ai_demo.zip"
            return StreamingResponse(
                buf,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{fname}"'}
            )
        else:
            return JSONResponse(content={
                "planId": video_id,
                "counts": counts,
                "segments": manifest,
                "public_urls": public_urls,
                "summary_url": f"{settings.STATIC_BASE_URL.rstrip('/')}/{memberId}/{highlightKey}/summary.json"
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if cleanup_needed:
            try:
                tmp_in.unlink(missing_ok=True)
            except Exception:
                pass
