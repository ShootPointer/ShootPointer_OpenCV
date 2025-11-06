# tools/fingerprint_video.py
from pathlib import Path
import hashlib, subprocess, cv2, numpy as np, sys

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1_048_576), b""): h.update(b)
    return h.hexdigest()

def ffprobe_dur_w_h(p: Path):
    q1 = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                         "-of","default=noprint_wrappers=1:nokey=1", str(p)],
                        stdout=subprocess.PIPE, text=True)
    q2 = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
                         "-show_entries","stream=width,height",
                         "-of","csv=s=x:p=0", str(p)],
                        stdout=subprocess.PIPE, text=True)
    dur = float((q1.stdout or "0").strip() or 0)
    try: w,h = [int(x) for x in (q2.stdout or "").strip().split("x")]
    except: w,h = 0,0
    return dur,w,h

def _phash(img: np.ndarray, hash_size=8) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (hash_size*4, hash_size*4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(g))[:hash_size,:hash_size]
    med = np.median(dct)
    bits = (dct > med).astype(np.uint8).flatten()
    v = 0
    for b in bits: v = (v<<1) | int(b)
    return int(v)

def video_sig(p: Path, samples=5):
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened(): return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (fps or 30.0)
    ts = [dur * t for t in (0.1,0.3,0.5,0.7,0.9)] if dur>0 else [0,1,2,3,4]
    sig=[]
    for t in ts[:samples]:
        if dur>0 and fps>0: cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
        ok,frame = cap.read()
        if not ok: break
        sig.append(_phash(frame))
    cap.release()
    return sig

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python tools/fingerprint_video.py /path/to/video.mp4"); sys.exit(1)
    p = Path(sys.argv[1]); assert p.exists()
    sha = file_sha256(p)
    dur,w,h = ffprobe_dur_w_h(p)
    sig = video_sig(p)
    print("\n# registry.py에 채울 값 예시")
    print(f'"sha256": "{sha}",')
    print(f'"duration_sec": {dur:.1f},  # {int(dur//60):02d}:{int(dur%60):02d}')
    print(f'"width": {w}, "height": {h},')
    print(f'"phash": {sig},\n')
