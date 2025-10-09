# app/core/logging.py
from __future__ import annotations
import logging
import sys
from typing import Optional


DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | pid=%(process)d tid=%(threadName)s | %(message)s"
)
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    try:
        return getattr(logging, str(level).upper())
    except Exception:
        return logging.INFO


def _tune_noisy_loggers(root_level: int, quiet_third_party: bool) -> None:
    """
    써드파티 로거들의 소음을 줄인다.
    - quiet_third_party=True이고, 전체 레벨이 DEBUG가 아닐 때만 적용
    """
    if not quiet_third_party or root_level == logging.DEBUG:
        return

    noisy_levels = {
        # uvicorn/fastapi 웹서버 관련
        "uvicorn": logging.WARNING,
        "uvicorn.access": logging.WARNING,
        "uvicorn.error": logging.WARNING,
        # HTTP 클라이언트
        "httpx": logging.WARNING,
        "urllib3": logging.WARNING,
        # 비동기 루프
        "asyncio": logging.WARNING,
        # 이미지/비디오 라이브러리에서 종종 떠들 수 있는 채널
        "PIL": logging.WARNING,
        "matplotlib": logging.WARNING,
    }
    for name, lvl in noisy_levels.items():
        try:
            logging.getLogger(name).setLevel(lvl)
        except Exception:
            pass


def setup_logging(level: str = "INFO", *, quiet_third_party: bool = True) -> None:
    """
    루트 로거를 초기화하고 표준 출력 스트림 핸들러를 연결한다.
    - level: "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL" 또는 int
    - quiet_third_party: 써드파티 로거(uvicorn/httpx 등) 소음 억제
    """
    root = logging.getLogger()
    root_level = _normalize_level(level)
    root.setLevel(root_level)

    # 기존 핸들러 제거(중복 로그 방지)
    for h in list(root.handlers):
        root.removeHandler(h)

    # 콘솔 핸들러
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(fmt=DEFAULT_FORMAT + ".%(msecs)03d", datefmt=DEFAULT_DATEFMT)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # noisy logger 조정
    _tune_noisy_loggers(root_level, quiet_third_party)

    # 일부 프레임워크 로거가 루트로 전파되도록(핸들러 중복 방지)
    logging.getLogger("uvicorn").propagate = True
    logging.getLogger("uvicorn.error").propagate = True
    logging.getLogger("uvicorn.access").propagate = True


def add_file_logger(
    filepath: str,
    level: Optional[str | int] = None,
    *,
    format: str = DEFAULT_FORMAT + ".%(msecs)03d",
    datefmt: str = DEFAULT_DATEFMT,
    mode: str = "a",
    encoding: str = "utf-8",
) -> logging.Handler:
    """
    선택적으로 파일 로거를 추가한다. 회전/압축은 포함하지 않는다.
    - filepath: 로그 파일 경로
    - level: 파일 핸들러 전용 레벨(미지정 시 루트 레벨 사용)
    사용 예) main.py에서 setup_logging() 후:
        from app.core.logging import add_file_logger
        add_file_logger("/var/log/app.log", level="INFO")
    """
    root = logging.getLogger()
    file_handler = logging.FileHandler(filepath, mode=mode, encoding=encoding, delay=True)
    file_handler.setFormatter(logging.Formatter(fmt=format, datefmt=datefmt))
    if level is not None:
        file_handler.setLevel(_normalize_level(level))
    root.addHandler(file_handler)
    return file_handler
