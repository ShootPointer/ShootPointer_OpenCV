from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np, cv2, math

@dataclass
class CourtSpec:
    three_radius: float
    corner_x: float
    ft_line_y: float
    rim_to_baseline: float
    ft_circle_radius: float

FIBA = CourtSpec(6.75, 6.60, 4.225, 1.575, 1.80)
NBA  = CourtSpec(7.239, 6.706, 4.572, 1.219, 1.80)

def warp_pixel_to_world(H: np.ndarray, xy: Tuple[int,int]) -> Optional[Tuple[float,float]]:
    if H is None: return None
    p = np.array([[xy]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0,0]
    return (float(out[0]), float(out[1]))

def is_free_throw(world_xy: Tuple[float,float], spec: CourtSpec, tol=0.6)->bool:
    x, y = world_xy
    y_ft_center = -spec.rim_to_baseline + spec.ft_line_y
    return math.hypot(x, y - y_ft_center) <= (spec.ft_circle_radius + tol)

def classify_2pt3pt(world_xy: Tuple[float,float], spec: CourtSpec, corner_tol=0.2)->str:
    x, y = world_xy
    if abs(x) >= (spec.corner_x - corner_tol): return "3PT"
    return "3PT" if math.hypot(x,y) >= spec.three_radius else "2PT"

# ── 자동 코트 보정(Homography) ──────────────────────────────────────
def _detect_hoop_px(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                               param1=120, param2=25, minRadius=12, maxRadius=90)
    if circles is None: return None
    cand = np.uint16(np.around(circles[0]))
    best, best_score = None, 0.0
    for x,y,r in cand:
        if y < frame.shape[0]//3:  # 가짜 억제
            continue
        roi = gray[max(0,y-r):min(gray.shape[0],y+r), max(0,x-r):min(gray.shape[1],x+r)]
        if roi.size == 0: continue
        score = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        if score > best_score:
            best_score, best = score, (int(x), int(y), int(r))
    return best

def _detect_court_lines(frame):
    h,w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),1.2)
    edges = cv2.Canny(gray,60,140)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=120,
                            minLineLength=int(0.25*w),maxLineGap=20)
    if lines is None: return None, None, None
    lines = lines.reshape(-1,4)
    vertical, horizontal = [], []
    for x1,y1,x2,y2 in lines:
        dx,dy = x2-x1, y2-y1
        ang = abs(math.degrees(math.atan2(dy,dx))); ang = 180-ang if ang>90 else ang
        if ang>=70: vertical.append((x1,y1,x2,y2))
        elif ang<=20: horizontal.append((x1,y1,x2,y2))
    left_line = right_line = None
    if vertical:
        xsL = [min(x1,x2) for x1,y1,x2,y2 in vertical]
        xsR = [max(x1,x2) for x1,y1,x2,y2 in vertical]
        left_line  = vertical[int(np.argmin(xsL))]
        right_line = vertical[int(np.argmax(xsR))]
    base_line = None
    if horizontal:
        ys = [max(y1,y2) for x1,y1,x2,y2 in horizontal]
        base_line = horizontal[int(np.argmax(ys))]
    return left_line, right_line, base_line

def _line_intersection(l1,l2):
    if l1 is None or l2 is None: return None
    x1,y1,x2,y2 = l1; x3,y3,x4,y4 = l2
    den = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(den)<1e-6: return None
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
    return (px,py)

def compute_homography_auto(frame, spec: CourtSpec):
    hoop = _detect_hoop_px(frame)
    L,R,B = _detect_court_lines(frame)
    if hoop is None or L is None or R is None or B is None:
        return None, None
    pL = _line_intersection(L,B); pR = _line_intersection(R,B)
    if pL is None or pR is None: return None, None
    hx,hy,_ = hoop
    pix = np.array([[hx,hy],
                    [hx, max(0, hy - int(0.6*(hy - min(hy, int(frame.shape[0]*0.2)))))],
                    [pL[0],pL[1]],[pR[0],pR[1]]], dtype=np.float32)
    world = np.array([[0.0,0.0],[0.0,spec.three_radius],
                      [-spec.corner_x,-1.0],[spec.corner_x,-1.0]], dtype=np.float32)
    H,_ = cv2.findHomography(pix, world, cv2.RANSAC, 3.0)
    return H, (hx,hy)
