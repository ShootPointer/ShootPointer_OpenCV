from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid

from app.services.frames import sample_frames_with_timestamps

router = APIRouter(prefix="/frames")

TMP_DIR = Path("/tmp/uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)

def _save_upload(video: UploadFile) -> Path:
    suffix = Path(video.filename).suffix or ".mp4"
    tmp = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return tmp

@router.post("/probe")
async def frames_probe(
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    fps: float = Form(0.5, description="초당 추출 프레임 수. 예: 0.5면 2초에 1장"),
    max_frames: int = Form(40, description="최대 프레임 수 (상한)"),
    scale_width: int = Form(640, description="썸네일 가로 최대 폭"),
):
    """
    영상에서 일정 간격(fps)으로 썸네일을 추출하고, 각 프레임의 실제 타임스탬프(초)를 함께 반환.
    → 반환 JSON의 't' 값들을 골라서 /highlight timestamps에 넣으면 하이라이트 ZIP 생성 가능.
    """
    tmp_in = _save_upload(video)
    try:
        frames = sample_frames_with_timestamps(tmp_in, fps=fps, max_frames=max_frames, scale_width=scale_width)
    except Exception as e:
        try: tmp_in.unlink(missing_ok=True)
        except: pass
        return JSONResponse(status_code=400, content={"error": str(e)})

    # 업로드 임시파일은 바로 제거 (썸네일은 /tmp/frames에 남음)
    try: tmp_in.unlink(missing_ok=True)
    except: pass

    return {"count": len(frames), "frames": frames}
