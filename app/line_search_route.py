"""独立寻线路线：参考同学line_follow的底部ROI、质心和分级纠偏。"""
import cv2
import numpy as np


def detect_line(image, color="white", previous_x=None):
    height, width = image.shape[:2]
    top = int(height * .8)
    roi = image[top:]
    if color == "white":
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 150), (179, 80, 255))
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 50 * width / 640 or w > width * .45:
            continue
        moments = cv2.moments(contour)
        if moments["m00"]:
            candidates.append(moments["m10"] / moments["m00"])
    if not candidates:
        return None
    anchor = width / 2 if previous_x is None else previous_x
    # 优先跟随上一次线的位置，减少跳到旁边地板图案。
    return min(candidates, key=lambda x: abs(x - anchor))


class LinePlanner:
    def __init__(self):
        self.search_index = 0
        self.previous_x = None

    def decide(self, x, width):
        if x is None:
            # 右探一步、回中、左探一步、回中；仍无目标就停止。
            search = ("TURN_RIGHT", "TURN_LEFT", "TURN_LEFT", "TURN_RIGHT")
            if self.search_index == len(search):
                return "STOP"
            action = search[self.search_index]
            self.search_index += 1
            return action
        self.search_index = 0
        self.previous_x = x
        error = x - width / 2
        scale = width / 640
        if abs(error) < 60 * scale:
            return "UP_LITTLE"
        if abs(error) < 100 * scale:
            return "SIDE_LEFT" if error < 0 else "SIDE_RIGHT"
        return "TURN_LEFT" if error < 0 else "TURN_RIGHT"


def run_course(io, color):
    settings = io.settings
    io.phase("寻找face")
    io.scan_until(settings["face_qr"], "belly")
    io.backward(1)
    io.right(1)
    io.squat()
    io.identity()
    io.stand()
    io.left(1)
    io.phase("局部寻线到搬运入口")
    io.follow_to_carry(color)
    io.phase("指定颜色搬运")
    io.carry(color, settings["drop_qr"])
    io.sport()
    io.right(settings["after_sport_right_actions"])
    io.set_head(settings["dance_head_position"])
    io.scan_until(settings["dance_qr"], "head", confirm_frames=2)
    io.enter_blue()
    io.dance()
    return True
