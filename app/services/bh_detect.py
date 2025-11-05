from typing import Tuple, Optional, List
import numpy as np, cv2

def detect_ball_hsv(frame) -> Optional[Tuple[int,int,int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([5,60,60]); upper = np.array([25,255,255])  # 오렌지 계열
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand, bestR = None, 0
    for c in cnts:
        (x,y), r = cv2.minEnclosingCircle(c)
        if 4<r<60 and cv2.contourArea(c)>40:
            if r>bestR: bestR=r; cand=(int(x),int(y),int(r))
    return cand

def is_score_event(traj: List[Tuple[int,int,int]], hoop_px: Tuple[int,int]) -> bool:
    if not traj or hoop_px is None or len(traj)<6: return False
    hx,hy = hoop_px
    xs = [x for x,_,_ in traj[-6:]]
    ys = [y for _,y,_ in traj[-6:]]
    if min(ys)<hy-20 and max(ys)>hy+10 and min(abs(np.array(xs)-hx))<40:
        return True
    return False

def ocr_digits(img) -> str:
    try:
        import pytesseract
    except Exception:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,31,5)
    cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"
    txt = pytesseract.image_to_string(gray, config=cfg)
    return "".join([c for c in txt if c.isdigit()])

def shooter_roi_near_ball(frame, b: Tuple[int,int,int]):
    h,w = frame.shape[:2]
    x,y,r = b
    x0 = max(0, x-80); x1 = min(w, x+80)
    y0 = max(0, y-200); y1 = min(h, y-60)
    if x1<=x0 or y1<=y0: return None
    return frame[y0:y1, x0:x1]
