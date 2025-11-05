from typing import List
import subprocess, tempfile, os

def ffprobe_duration(path: str) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=nokey=1:noprint_wrappers=1", path
        ]).decode().strip()
        return float(out)
    except Exception:
        return 0.0

def cut_and_overlay(src: str, start: float, end: float, text: str, out_path: str):
    start = max(0.0, start); dur = max(0.1, end-start)
    vf = ("scale=w=1080:h=1920:force_original_aspect_ratio=decrease,"
          "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
          f"drawtext=text='{text}':x=40:y=50:fontsize=48:fontcolor=white:borderw=2")
    cmd = ["ffmpeg","-y","-ss",f"{start:.3f}","-i",src,"-t",f"{dur:.3f}",
           "-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","20",
           "-c:a","aac","-b:a","128k", out_path]
    subprocess.run(cmd, check=True)

def concat_videos(paths: List[str], dst: str):
    if len(paths)==1:
        subprocess.run(["ffmpeg","-y","-i",paths[0],"-c","copy",dst], check=True); return
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        for p in paths: f.write(f"file '{os.path.abspath(p)}'\n")
        list_path = f.name
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",list_path,"-c","copy",dst], check=True)
    os.remove(list_path)
