# app.py

from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS

from user_store import user_data_store
from upload import register_upload_route  # 👈 upload.py에 정의된 라우터 등록

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB 제한
CORS(app)


# ✅ 헬스 체크 라우트
@app.route("/health-check", methods=["GET"])
def healthcheck():
    return jsonify({
        "status": "ok",
        "message": "Flask server is healthy"
    }), 200


# ✅ 백엔드가 보내는 등번호 이미지 수신
@app.route("/api/send-img", methods=["POST"])
def receive_user():
    jwt_token = request.headers.get("Authorization")
    member_id = request.headers.get("X-Member-Id")
    back_number = request.form.get("backNumber")
    image_file = request.files.get("image")

    if not jwt_token or not member_id or not back_number or not image_file:
        return jsonify({
            "status": "fail",
            "message": "필수 정보 누락"
        }), 400

    jersey_img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if jersey_img is None:
        return jsonify({"status": "fail", "message": "이미지 디코딩 실패"}), 400

    user_data_store[member_id] = {
        "jwt_token": jwt_token,
        "back_number": back_number,
        "jersey_img": jersey_img
    }

    print(f"✅ 등번호 이미지 저장 완료 (member_id={member_id}, backNumber={back_number})")
    return jsonify({"status": 200, "success": True}), 200


# ✅ upload 라우트 등록
register_upload_route(app)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)