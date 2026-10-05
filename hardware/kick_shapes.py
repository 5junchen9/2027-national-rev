"""圆形球与矩形/三边门框候选；只检测外形，不声称理解目标身份。"""
from dataclasses import dataclass
import cv2
import numpy as np
from round_object import find_round_objects


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def cx(self): return self.x + self.width / 2

    @property
    def cy(self): return self.y + self.height / 2

    @property
    def bottom(self): return self.y + self.height

    def normalized(self, shape):
        height, width = shape[:2]
        return Box(self.x / width, self.y / height, self.width / width, self.height / height)


def overlap(a, b):
    area = max(0, min(a.x+a.width, b.x+b.width)-max(a.x, b.x)) * max(
        0, min(a.bottom, b.bottom)-max(a.y, b.y))
    return area / max(1, a.width*a.height + b.width*b.height-area)


def find_goals(frame):
    height, width = frame.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0), 40, 120)
    candidates = []
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, .025*perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            x, y, w, h = cv2.boundingRect(polygon)
            if cv2.contourArea(polygon) / max(1, w*h) > .75:
                candidates.append(Box(x, y, w, h))
    # 门可能没有画底边：用两根立柱与顶部横梁补充U形框检测。
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 35,
                            minLineLength=max(20, int(width*.08)), maxLineGap=12)
    vertical, horizontal = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if abs(x2-x1) < .18*abs(y2-y1):
                vertical.append((int((x1+x2)/2), min(y1,y2), max(y1,y2)))
            if abs(y2-y1) < .18*abs(x2-x1):
                horizontal.append((min(x1,x2), max(x1,x2), int((y1+y2)/2)))
        vertical = sorted(vertical, key=lambda p: p[2]-p[1], reverse=True)[:20]
        horizontal = sorted(horizontal, key=lambda p: p[1]-p[0], reverse=True)[:20]
        for left in vertical:
            for right in vertical:
                span = right[0]-left[0]
                h = min(left[2]-left[1], right[2]-right[1])
                if span <= 0 or h <= 0 or abs(left[1]-right[1]) > .15*h:
                    continue
                top = (left[1]+right[1])/2
                if any(a <= left[0]+.08*span and b >= right[0]-.08*span
                       and abs(y-top) < max(8, .12*h) for a,b,y in horizontal):
                    candidates.append(Box(left[0], top, span, h))
    goals = []
    for box in sorted(candidates, key=lambda b: b.width*b.height, reverse=True):
        if (1.1 <= box.width/max(1,box.height) <= 4.5 and box.height >= .08*height
                and .02 <= box.width*box.height/(width*height) <= .8
                and not any(overlap(box, old) > .55 for old in goals)):
            goals.append(box)
    return goals


def observe(frame, with_goal=True):
    balls = [Box(b.centerX-b.width/2, b.centerY-b.height/2, b.width, b.height)
             for b in find_round_objects(frame)]
    goals = find_goals(frame) if with_goal else []
    return {'ball': balls[0] if len(balls) == 1 else None,
            'goal': goals[0] if len(goals) == 1 else None,
            'ball_count': len(balls), 'goal_count': len(goals)}
