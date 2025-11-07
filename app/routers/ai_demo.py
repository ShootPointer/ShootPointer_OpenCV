# app/routers/ai_demo.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse

from app.routers.highlight import _save_upload
from app.services.ai_demo_runner import (
    AIHighlightError,
    run_ai_demo_for_existing,
    run_ai_demo_from_path,
)

router = APIRouter()


@router.post("/highlight/ai-demo")
async def highlight_ai_demo(
    # 업로드 모드: 파일 업로드. useExisting=True면 생략 가능
    video: Optional[UploadFile] = File(
        default=None,
        description="데모 영상 1개(업로드 모드일 때만 필요)",
    ),
    useExisting: bool = Form(
        False,
        description="이미 저장된 original_*.mp4를 사용",
    ),
    videoCode: Optional[str] = Form(
        default=None,
        description="선택: 힌트(1|2|3) – 없어도 자동 식별",
    ),
    overlay_model_tag: str = Form(
        "AI-Selector",
        description="오버레이에 표시할 모델 태그",
    ),
    merge_output: bool = Form(
        True,
        description="개별 클립들을 하나로 이어 붙일지",
    ),
    memberId: str = Form(
        ...,
        description="저장 멤버 ID (필수)",
    ),
    highlightKey: str = Form(
        ...,
        description="저장 하이라이트 키 (필수)",
    ),
    returnZip: bool = Form(
        True,
        description="ZIP 스트림으로 반환(아니면 URL JSON)",
    ),
):
    """
    데모용 하이라이트 생성 엔드포인트.

    - useExisting=True:
        SAVE_ROOT/{memberId}/{highlightKey} 아래의 최신 original_*.mp4를 찾아 사용.
    - useExisting=False:
        업로드된 video 파일을 /tmp 에 저장 후 사용.

    실제 컷팅/요약/summary.json 생성, shorts 저장 등 상세 로직은
    app.services.ai_demo_runner 모듈에 위임한다.
    Generate highlight clips either from an uploaded file or an existing original.
    """
    tmp_in: Optional[Path] = None

    try:
        # 이미 저장된 original_* 사용 모드
        if useExisting:
            result = run_ai_demo_for_existing(
                member_id=memberId,
                highlight_key=highlightKey,
                overlay_model_tag=overlay_model_tag,
                merge_output=merge_output,
                video_code=videoCode,
            )

        # 업로드 모드
        else:
            if not video:
                return JSONResponse(
                    status_code=400,
                    content={"error": "upload mode requires 'video' file"},
                )

            # 업로드 파일을 /tmp 에 저장
            tmp_in = _save_upload(video)
            cleanup_needed = True

            base_name = Path(video.filename or tmp_in.stem).stem

            result = run_ai_demo_from_path(
                tmp_in,
                member_id=memberId,
                highlight_key=highlightKey,
                base_name=base_name,
                overlay_model_tag=overlay_model_tag,
                merge_output=merge_output,
                video_code=videoCode,
            )

        # ZIP 형태로 반환
        if returnZip:
            buf = result.build_zip()
            fname = f"{result.base_name}_{result.plan_id}_ai_demo.zip"

            return StreamingResponse(
                buf,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{fname}"'
                },
            )

        # JSON 형태로 메타 정보 반환
        return JSONResponse(
            content={
                "planId": result.plan_id,
                "counts": result.counts,
                "segments": result.segments,
                "public_urls": result.public_urls,
                "summary_url": result.summary_url,
            }
        )

    except AIHighlightError as e:
        status_code = getattr(e, "status_code", 400)
        return JSONResponse(
            status_code=status_code,
            content={"error": str(e)},
        )

    except Exception as e:
        # 예기치 못한 서버 에러
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

    finally:
        # 업로드 모드에서만 /tmp 파일 정리
        if tmp_in:
            try:
                tmp_in.unlink(missing_ok=True)
            except Exception:
                pass