from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
import shutil
import uuid

from app.core.config import settings
from app.services.jersey import detect_player_segments
from app.services.pipeline import generate_highlight_clips, build_zip_in_memory

router = APIRouter(prefix="/player")

TMP_DIR = Path("/tmp/uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)

def _save_upload(video: UploadFile) -> Path:
    suffix = Path(video.filename).suffix or ".mp4"
    tmp = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return tmp

def _in_any_segment(t: float, segments: list[tuple[float, float]]) -> bool:
    for s, e in segments:
        if s <= t <= e:
            return True
    return False

@router.post("/highlight")
async def player_highlight_zip(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    jersey_number: int = Form(..., description="찾을 등번호 (예: 23)"),
    timestamps: str = Form(..., description="쉼표구분 초 리스트, 예: 30,75,120"),
    pre: float = Form(default=settings.DEFAULT_PRE),
    post: float = Form(default=settings.DEFAULT_POST),
    max_clips: int = Form(default=settings.MAX_CLIPS),
):
    """
    선수(등번호) 구간 안에 포함되는 타임스탬프만 필터링해 하이라이트 ZIP 반환.
    현재 detect_player_segments는 '영상 전체'를 반환하므로 사실상 통과.
    모델 붙이면 자동으로 필터가 엄격해짐.
    """
    tmp_in = _save_upload(video)

    try:
        # 1) 등번호 기반 플레이 구간 검출(스텁: 전체 길이)
        segments = detect_player_segments(tmp_in, jersey_number)

        # 2) 입력 timestamps 문자열 → float 리스트
        events = []
        for s in timestamps.split(","):
            s = s.strip()
            if s:
                events.append(float(s))

        # 3) 선수 구간 필터
        events = [t for t in events if _in_any_segment(t, segments)]
        if not events:
            raise RuntimeError("선수 구간과 겹치는 이벤트가 없습니다. (timestamps/등번호 확인)")

        # 4) 클립 생성 및 ZIP
        results = generate_highlight_clips(
            tmp_in, jersey_number, events, pre=pre, post=post, max_clips=max_clips
        )
        clip_paths = [p for _, p in results]
        buf = build_zip_in_memory(clip_paths)
    except Exception as e:
        try:
            tmp_in.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse(status_code=400, content={"error": str(e)})

    # 정리
    try:
        tmp_in.unlink(missing_ok=True)
    except Exception:
        pass
    for p in clip_paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="player_highlights.zip"'},
    )
