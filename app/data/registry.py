# 역할: 데모 3개 영상 자동식별 + 자를 구간(시작,끝,라벨)
# 라벨: "2PT" | "3PT" | "FT"

# 메타데이터 매칭 허용 오차 (Worker에서 사용)
DURATION_TOLERANCE_SEC = 2.0 
PHASH_HAMMING_THRESHOLD = 6
# 💡 [NEW] 파일 크기 허용 오차: 10KB (10240 Bytes)를 기준으로 매칭합니다.
SIZE_TOLERANCE_BYTES = 10240 

PLANS = [
    {
        "id": "1",
        # 크기: 19.5MB (20,490,752 Bytes)
        "size_bytes": 20490752, 
        "duration_sec": 79.0,      # 00:01:19
        "width": 1304,             # 프레임 너비
        "height": 720,             # 💡 [업데이트] 높이 720으로 통일
        "sha256": "",
        "phash": [],
        "segments": [
            (5.0, 15.0, "2PT"),
            (33.0, 43.0, "3PT"),
            (65.0, 75.0, "FT"),
        ],
    },
    {
        "id": "2",
        # 크기: 37.0MB (38,863,703 Bytes)
        "size_bytes": 38863703,
        "duration_sec": 153.0,     # 00:02:33
        "width": 1282,             # 프레임 너비
        "height": 720,             # 💡 [업데이트] 높이 720으로 통일
        "sha256": "",
        "phash": [],
        "segments": [
            (3.0, 13.0, "3PT"),
            (135.0, 145.0, "2PT"),
        ],
    },
    {
        "id": "3",
        # 크기: 39.9MB (41,912,920 Bytes)
        "size_bytes": 41912920,
        "duration_sec": 163.0,     # 00:02:43
        "width": 1274,             # 프레임 너비
        "height": 720,             # 💡 [업데이트] 높이 720으로 통일
        "sha256": "",
        "phash": [],
        "segments": [
            (2.0, 12.0, "2PT"),
            (25.0, 35.0, "FT"),
            (151.0, 161.0, "2PT"),
        ],
    },
]