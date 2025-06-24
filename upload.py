from flask import request, jsonify
from user_store import user_data_store
from run_score_detection import run_score_detection
import os

def register_upload_route(app):
    @app.route("/upload", methods=["POST"])
    def upload_video():
        print("📥 [UPLOAD] 요청 도착")

        member_id = request.headers.get("X-Member-Id", "test123")
        video_file = request.files.get("video")

        if not video_file:
            return jsonify({
                "status": "fail",
                "message": "video 파일이 없습니다"
            }), 400

        # ✅ 등번호 이미지 확인
        user_info = user_data_store.get(member_id)
        if not user_info or "jersey_img" not in user_info:
            return jsonify({
                "status": "fail",
                "message": "등번호 이미지가 없습니다. /api/send-img 먼저 호출해야 합니다."
            }), 400

        jersey_img = user_info["jersey_img"]

        # ✅ 영상 임시 저장
        temp_path = f"temp_{member_id}.mp4"
        try:
            with open(temp_path, "wb") as f:
                f.write(video_file.read())
            print(f"✅ 영상 저장 성공: {temp_path}")
        except Exception as e:
            print("❌ 영상 저장 실패:", e)
            return jsonify({"status": "fail", "message": "영상 저장 실패"}), 500

        # ✅ 분석 시작
        try:
            print("🧠 분석 시작...")
            highlights = run_score_detection(temp_path, jersey_img)
            print(f"🎬 분석 완료 - 클립 수: {len(highlights)}")
        except Exception as e:
            print("❌ 분석 중 오류:", e)
            return jsonify({
                "status": "fail",
                "message": f"분석 실패: {str(e)}"
            }), 500

        # ✅ 응답 개선: 프론트에 필요한 최소 정보 제공
        return jsonify({
            "status": 200,
            "message": "분석 성공 및 전송 완료",
            "highlight_count": len(highlights),
            "highlight_times": [round(h["time"], 2) for h in highlights]
        }), 200
