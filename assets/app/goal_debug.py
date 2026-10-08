"""门框外形识别：矩形或两根立柱加横梁，连续画面确认。"""
import cv2
import numpy as np
from kick_shapes import find_goals, overlap


def find_goal_frames(frame):
    """门口必须同时有左右立柱和顶部横梁，不仅是一个矩形。"""
    boxes = find_goals(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),40,120)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,25,minLineLength=20,maxLineGap=12)
    if lines is None:
        return []
    vertical,horizontal = [],[]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        if abs(x2-x1) <= .25*abs(y2-y1):
            vertical.append(((x1+x2)/2,min(y1,y2),max(y1,y2)))
        if abs(y2-y1) <= .25*abs(x2-x1):
            horizontal.append((min(x1,x2),max(x1,x2),(y1+y2)/2))
    goals = []
    for box in boxes:
        if not 1.1 <= box.width/box.height <= 3.5:
            continue
        tolerance_x = max(8,box.width*.08)
        tolerance_y = max(8,box.height*.15)
        def has_post(side):
            return any(abs(x-side) <= tolerance_x and abs(top-box.y) <= tolerance_y
                       and bottom-top >= box.height*.6 for x,top,bottom in vertical)
        has_bar = any(abs(y-box.y) <= tolerance_y and left <= box.x+tolerance_x
                      and right >= box.x+box.width-tolerance_x
                      for left,right,y in horizontal)
        if has_post(box.x) and has_post(box.x+box.width) and has_bar:
            goals.append(box)
    return goals


class GoalTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.box = None
        self.stable_frames = 0
        self.reason = 'waiting for goal frame'

    def update(self, frame):
        candidates = find_goal_frames(frame)
        # 门没有可靠的身份评分，多门框时暂停，避免朝另一个矩形走。
        if len(candidates) != 1:
            self.reset()
            self.reason = 'no goal frame' if not candidates else 'multiple goal frames; hold'
            return None
        box = candidates[0]
        stable = self.box is not None and overlap(box,self.box) >= .65
        self.stable_frames = self.stable_frames+1 if stable else 1
        self.box = box
        self.reason = 'goal confirmed' if self.stable_frames >= 5 else 'confirming goal frame'
        return box
