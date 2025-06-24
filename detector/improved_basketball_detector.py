import cv2
from ultralytics import YOLO
import os
from collections import deque
import numpy as np
import math

class StableBasketballScoreDetector:
    def __init__(self):
        # 안정적인 골대 위치 추적을 위한 큐 (더 많은 프레임으로 안정화)
        self.basket_positions = deque(maxlen=50)
        self.ball_positions = deque(maxlen=20)
        
        # 안정화된 골대 정보
        self.stable_basket_center = None
        self.stable_basket_top = None
        self.stable_basket_width = None
        self.basket_confidence_threshold = 10  # 최소 10프레임 누적 후 안정화
        
        # 득점 감지 상태
        self.score_detection_state = {
            'phase': 'idle',  # idle, approaching, scoring, cooldown
            'ball_trajectory': [],
            'frames_in_phase': 0,
            'last_score_frame': -1000,
            'min_cooldown_frames': 90  # 3초 쿨다운 (30fps 기준)
        }
        
        # 조정된 파라미터 (더 관대하게)
        self.params = {
            'horizontal_tolerance': 80,  # 픽셀 단위로 고정
            'approach_zone_height': 60,  # 접근 구역 높이
            'scoring_zone_height': 40,   # 득점 구역 높이
            'min_descent_speed': 2,      # 최소 하강 속도
            'trajectory_length': 8,      # 궤적 분석 길이
            'stability_frames': 5        # 안정성 확인 프레임
        }
        
    def update_basket_tracking(self, basket_bbox):
        """골대 위치 안정적 추적"""
        if basket_bbox is None:
            return False
            
        x1, y1, x2, y2 = basket_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        
        # 골대 위치 기록
        self.basket_positions.append({
            'center': (center_x, center_y),
            'top': y1,
            'width': width
        })
        
        # 충분한 데이터가 모이면 안정화된 위치 계산
        if len(self.basket_positions) >= self.basket_confidence_threshold:
            centers = [pos['center'] for pos in self.basket_positions]
            tops = [pos['top'] for pos in self.basket_positions]
            widths = [pos['width'] for pos in self.basket_positions]
            
            # 중앙값 사용 (이상치에 더 강함)
            centers_x = sorted([c[0] for c in centers])
            centers_y = sorted([c[1] for c in centers])
            
            self.stable_basket_center = (
                centers_x[len(centers_x)//2],
                centers_y[len(centers_y)//2]
            )
            self.stable_basket_top = sorted(tops)[len(tops)//2]
            self.stable_basket_width = sorted(widths)[len(widths)//2]
            
            return True
        return False
    
    def is_basket_stable(self):
        """골대 위치가 안정화되었는지 확인"""
        return (self.stable_basket_center is not None and 
                self.stable_basket_top is not None and 
                self.stable_basket_width is not None)
    
    def get_ball_trajectory_info(self):
        """공의 궤적 정보 분석"""
        if len(self.ball_positions) < 3:
            return None
            
        positions = list(self.ball_positions)
        
        # 수직 속도 계산 (최근 5프레임)
        if len(positions) >= 5:
            recent_positions = positions[-5:]
            vertical_speeds = []
            for i in range(1, len(recent_positions)):
                speed = recent_positions[i][1] - recent_positions[i-1][1]
                vertical_speeds.append(speed)
            avg_vertical_speed = sum(vertical_speeds) / len(vertical_speeds)
        else:
            avg_vertical_speed = 0
            
        # 궤적의 일관성 확인
        if len(positions) >= 3:
            # 최근 3포인트가 모두 하강 또는 상승하는지 확인
            recent_3 = positions[-3:]
            is_consistent_descent = all(
                recent_3[i][1] < recent_3[i+1][1] for i in range(len(recent_3)-1)
            )
            is_consistent_ascent = all(
                recent_3[i][1] > recent_3[i+1][1] for i in range(len(recent_3)-1)
            )
        else:
            is_consistent_descent = False
            is_consistent_ascent = False
            
        return {
            'vertical_speed': avg_vertical_speed,
            'is_descending': is_consistent_descent,
            'is_ascending': is_consistent_ascent,
            'current_y': positions[-1][1]
        }
    
    def detect_scoring_simple_and_effective(self, ball_bbox, basket_bbox):
        """단순하고 효과적인 득점 감지"""
        current_frame = self.score_detection_state['frames_in_phase']
        
        # 골대 추적 업데이트
        basket_stable = self.update_basket_tracking(basket_bbox)
        
        # 골대가 안정화되지 않았으면 대기
        if not self.is_basket_stable():
            return False
            
        # 공이 감지되지 않았으면 대기
        if ball_bbox is None:
            return False
            
        # 쿨다운 체크
        if (current_frame - self.score_detection_state['last_score_frame'] < 
            self.score_detection_state['min_cooldown_frames']):
            return False
            
        # 공 위치 업데이트
        ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
        ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
        self.ball_positions.append((ball_center_x, ball_center_y))
        
        # 공이 골대 수평 범위 내에 있는지 확인
        horizontal_distance = abs(ball_center_x - self.stable_basket_center[0])
        if horizontal_distance > self.params['horizontal_tolerance']:
            return False
            
        # 궤적 정보 획득
        trajectory_info = self.get_ball_trajectory_info()
        if trajectory_info is None:
            return False
            
        # 간단한 3단계 득점 감지
        basket_top = self.stable_basket_top
        approach_zone = basket_top - self.params['approach_zone_height']
        scoring_zone_bottom = basket_top + self.params['scoring_zone_height']
        
        current_state = self.score_detection_state
        
        # 상태 기계 업데이트
        if current_state['phase'] == 'idle':
            # 공이 접근 구역에 진입하고 하강 중인지 확인
            if (ball_center_y > approach_zone and ball_center_y < basket_top and
                trajectory_info['is_descending']):
                current_state['phase'] = 'approaching'
                current_state['frames_in_phase'] = 0
                print(f"🎯 접근 단계: 공이 골대 위쪽에서 하강 중")
                
        elif current_state['phase'] == 'approaching':
            current_state['frames_in_phase'] += 1
            
            # 공이 골대 림을 통과했는지 확인
            if (ball_center_y > basket_top and ball_center_y < scoring_zone_bottom and
                trajectory_info['is_descending'] and
                trajectory_info['vertical_speed'] > self.params['min_descent_speed']):
                current_state['phase'] = 'scoring'
                current_state['frames_in_phase'] = 0
                print(f"🏀 득점 단계: 공이 골대 림 통과")
                
            # 타임아웃 또는 조건 불만족
            elif current_state['frames_in_phase'] > 30 or ball_center_y < approach_zone:
                current_state['phase'] = 'idle'
                current_state['frames_in_phase'] = 0
                
        elif current_state['phase'] == 'scoring':
            current_state['frames_in_phase'] += 1
            
            # 공이 득점 구역을 완전히 통과했는지 확인
            if ball_center_y > scoring_zone_bottom:
                print(f"✅ 득점 확정!")
                current_state['phase'] = 'cooldown'
                current_state['last_score_frame'] = current_frame
                current_state['frames_in_phase'] = 0
                return True
                
            # 타임아웃
            elif current_state['frames_in_phase'] > 15:
                current_state['phase'] = 'idle'
                current_state['frames_in_phase'] = 0
                
        elif current_state['phase'] == 'cooldown':
            current_state['frames_in_phase'] += 1
            if current_state['frames_in_phase'] > 30:  # 1초 쿨다운
                current_state['phase'] = 'idle'
                current_state['frames_in_phase'] = 0
                
        return False
    
    def visualize_stable_detection(self, frame, ball_bbox, basket_bbox, is_scoring):
        """안정화된 시각화"""
        # 안정화된 골대 영역 표시
        if self.is_basket_stable():
            center_x, center_y = self.stable_basket_center
            top_y = self.stable_basket_top
            width = self.stable_basket_width
            
            # 골대 박스 (안정화됨)
            x1 = int(center_x - width/2)
            x2 = int(center_x + width/2)
            y1 = int(top_y)
            y2 = int(top_y + width/2)  # 골대 높이를 너비의 절반으로 추정
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Stable Hoop", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 안정화된 득점 영역 표시
            tolerance = self.params['horizontal_tolerance']
            approach_zone = top_y - self.params['approach_zone_height']
            scoring_zone = top_y + self.params['scoring_zone_height']
            
            # 득점 영역 (고정된 박스)
            zone_x1 = int(center_x - tolerance)
            zone_x2 = int(center_x + tolerance)
            
            cv2.rectangle(frame, (zone_x1, int(approach_zone)), 
                         (zone_x2, int(scoring_zone)), (0, 255, 255), 2)
            cv2.putText(frame, "Score Zone", (zone_x1, int(approach_zone) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 골대 중심선
            cv2.line(frame, (int(center_x), int(approach_zone)), 
                    (int(center_x), int(scoring_zone)), (255, 0, 255), 2)
        
        # 현재 감지된 골대 (반투명)
        if basket_bbox is not None:
            x1, y1, x2, y2 = map(int, basket_bbox)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # 공 표시
        if ball_bbox is not None:
            x1, y1, x2, y2 = map(int, ball_bbox)
            color = (0, 255, 0) if not is_scoring else (0, 0, 255)
            thickness = 4 if is_scoring else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, "Ball", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 공의 궤적 표시 (안정화됨)
        if len(self.ball_positions) > 1:
            points = [(int(p[0]), int(p[1])) for p in list(self.ball_positions)]
            for i in range(1, len(points)):
                thickness = max(1, 3 - i//3)  # 최근일수록 굵게
                cv2.line(frame, points[i-1], points[i], (255, 255, 0), thickness)
        
        # 득점 감지 시 강조 표시
        if is_scoring:
            cv2.putText(frame, "🏀 SCORE!", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
        
        # 상태 정보 표시
        state_info = [
            f"Phase: {self.score_detection_state['phase']}",
            f"Frames in phase: {self.score_detection_state['frames_in_phase']}",
            f"Basket stable: {self.is_basket_stable()}",
            f"Ball positions: {len(self.ball_positions)}"
        ]
        
        for i, info in enumerate(state_info):
            cv2.putText(frame, info, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

# 메인 실행 코드
def main():
    # 경로 설정
    video_path = r"C:\Users\이동현\Desktop\shootpoint\basketball_strong.mp4"
    ring_model_path = r"C:\Users\이동현\Desktop\shootpoint\runs\detect\basketball_yolo_model4\weights\best.pt"
    ball_model_path = r"C:\Users\이동현\Desktop\shootpoint\runs\detect\ball_model4\weights\best.pt"
    output_dir = r"C:\Users\이동현\Desktop\shootpoint\shorts"
    os.makedirs(output_dir, exist_ok=True)

    # 모델 로드
    print("🔄 모델 로딩 중...")
    ring_model = YOLO(ring_model_path)
    ball_model = YOLO(ball_model_path)
    print("✅ 모델 로딩 완료")

    # 안정화된 득점 감지기 초기화
    detector = StableBasketballScoreDetector()

    # 비디오 설정
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 비디오 정보: {total_frames} 프레임, {fps:.2f} FPS")

    scored_frames = []
    frame_idx = 0
    
    print("🔍 안정화된 득점 장면 분석 시작...")
    
    # 시각화 옵션 (디버깅용)
    show_visualization = True  # True로 설정하면 실시간 화면 표시
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 감지 (신뢰도 조정)
        ring_results = ring_model.predict(frame, conf=0.3, iou=0.5, verbose=False)[0]
        ball_results = ball_model.predict(frame, conf=0.3, iou=0.5, verbose=False)[0]

        # 골대와 공 박스 추출 (가장 확신도 높은 것 선택)
        ring_boxes = [(box, conf) for box, conf, cls in 
                     zip(ring_results.boxes.xyxy.cpu().numpy(), 
                         ring_results.boxes.conf.cpu().numpy(),
                         ring_results.boxes.cls.tolist()) if cls == 0]
        
        ball_boxes = [(box, conf) for box, conf, cls in 
                     zip(ball_results.boxes.xyxy.cpu().numpy(), 
                         ball_results.boxes.conf.cpu().numpy(),
                         ball_results.boxes.cls.tolist()) if cls == 0]

        # 가장 확신도 높은 감지 결과 선택
        ring_box = max(ring_boxes, key=lambda x: x[1])[0] if ring_boxes else None
        ball_box = max(ball_boxes, key=lambda x: x[1])[0] if ball_boxes else None

        # 프레임 카운터 업데이트
        detector.score_detection_state['frames_in_phase'] = frame_idx

        # 안정화된 득점 감지
        is_scoring = detector.detect_scoring_simple_and_effective(ball_box, ring_box)
        
        if is_scoring:
            scored_frames.append(frame_idx)
            print(f"🎯 득점 감지! 프레임: {frame_idx}, 시간: {frame_idx / fps:.2f}초")

        # 시각화 (옵션)
        if show_visualization:
            frame = detector.visualize_stable_detection(frame, ball_box, ring_box, is_scoring)
            # 창 크기 조정
            height, width = frame.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imshow("Stable Basketball Score Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        
        # 진행률 표시
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"진행률: {progress:.1f}% ({frame_idx}/{total_frames})")

    cap.release()
    cv2.destroyAllWindows()

    print(f"📊 총 {len(scored_frames)}개의 득점 감지")

    # 중복 제거는 더 짧은 간격으로 (2초)
    filtered_frames = []
    min_interval = int(fps * 2)  # 2초 간격
    
    for idx in scored_frames:
        if not filtered_frames or idx - filtered_frames[-1] > min_interval:
            filtered_frames.append(idx)

    print(f"🎬 중복 제거 후 {len(filtered_frames)}개의 득점 장면 확정")

    # 득점 클립 추출
    def extract_clip(start_frame, end_frame, clip_index):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out_path = os.path.join(output_dir, f"stable_score_clip_{clip_index + 1}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        out.release()
        cap.release()
        return out_path

    # 득점 클립들 생성
    if filtered_frames:
        print("✂️ 득점 클립 생성 중...")
        created_clips = []
        
        for i, score_frame in enumerate(filtered_frames):
            start_frame = max(0, score_frame - int(fps * 3))  # 3초 전
            end_frame = min(total_frames - 1, score_frame + int(fps * 3))  # 3초 후
            
            clip_path = extract_clip(start_frame, end_frame, i)
            created_clips.append(clip_path)
            
            score_time = score_frame / fps
            print(f"  📹 클립 {i + 1}: {score_time:.2f}초 지점")

        print(f"\n🎉 작업 완료!")
        print(f"✅ 총 {len(created_clips)}개의 득점 장면 숏츠 영상 생성")
        print(f"📁 저장 위치: {output_dir}")
    else:
        print("❌ 감지된 득점 장면이 없습니다.")
        print("💡 팁: show_visualization = True로 설정하여 실시간 감지 과정을 확인해보세요.")
    
    return filtered_frames

if __name__ == "__main__":
    main()