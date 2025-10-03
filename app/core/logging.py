# app/core/logging.py
import logging
import sys

def setup_logging(level: str = "INFO"):
    root = logging.getLogger()
    root.setLevel(level.upper())

    # 기존 핸들러 제거
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)

    # noisy logger 조정(원하면)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
