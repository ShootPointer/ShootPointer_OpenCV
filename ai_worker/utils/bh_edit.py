#ai_worker/utils/bh_edit.py
import subprocess
import json
import os
from typing import Dict, Any

def get_video_metadata(filepath: str) -> Dict[str, Any]:
    """
    FFprobe를 사용하여 영상 파일의 메타데이터 (길이, 크기, 해상도)를 추출합니다.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Video file not found at: {filepath}")

    # 파일 크기 (바이트) 추출
    file_size_bytes = os.path.getsize(filepath)

    # FFprobe 명령어 실행 (JSON 형식 출력 요청)
    cmd = [
        "ffprobe", 
        "-v", "quiet", 
        "-print_format", "json", 
        "-show_format", 
        "-show_streams", 
        filepath
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e.stderr}")
        return {"error": "FFprobe failed to analyze file"}
    except FileNotFoundError:
        print("Error: ffprobe is not installed or not in PATH.")
        return {"error": "FFprobe not found"}
    except Exception as e:
        print(f"General error during metadata extraction: {e}")
        return {"error": "Metadata parsing failed"}

    metadata = {
        "filepath": filepath,
        "size_bytes": file_size_bytes,
        "duration_sec": 0.0,
        "width": 0,
        "height": 0,
    }

    # Duration 추출 (format 섹션에서)
    if 'format' in data and 'duration' in data['format']:
        try:
            metadata["duration_sec"] = float(data['format']['duration'])
        except ValueError:
            pass
    
    # 해상도 추출 (streams 섹션에서)
    if 'streams' in data:
        for stream in data['streams']:
            if stream.get('codec_type') == 'video':
                metadata["width"] = stream.get('width', 0)
                metadata["height"] = stream.get('height', 0)
                # 첫 번째 비디오 스트림 정보만 사용
                break
    
    return metadata