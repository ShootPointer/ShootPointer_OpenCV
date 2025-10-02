from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil
import uuid

router = APIRouter()

TMP_DIR = Path("/tmp/uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_clip(
    video: UploadFile = File(..., description="풀경기 영상 파일"),
    jersey_number: int = Form(..., description="찾을 등번호 (예: 23)")
):
    suffix = Path(video.filename).suffix or ".mp4"
    tmp_path = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return {"status": "received", "tmp_path": str(tmp_path), "jersey_number": jersey_number}
