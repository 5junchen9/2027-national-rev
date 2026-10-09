"""独立寻线路线：参考同学line_follow的底部ROI、质心和分级纠偏。"""
import cv2
import numpy as np
import math


def line_mask(image, color="white"):
    """先遮掉二维码，再检测细长路线；反光仍需靠形状和位置连续性排除。"""
    if color == "white":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 150), (179, 80, 255))
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    if color == "white":
        # 白纸上的二维码即使漏定位，仍常有多个被白色包围的黑块。
        # 只排除这种带多个孔洞的白色区域，不把普通实心白线整体遮掉。
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for index, contour in enumerate(contours):
                if hierarchy[0][index][3] != -1:
                    continue
                _, _, w, h = cv2.boundingRect(contour)
                child = hierarchy[0][index][2]
                holes = 0
                hole_area = 0
                while child != -1:
                    area = cv2.contourArea(contours[child])
                    if area >= 4:
                        holes += 1
                        hole_area += area
                    child = hierarchy[0][child][0]
                if w >= 24 and h >= 24 and holes >= 3 and hole_area > w*h*.03:
                    cv2.drawContours(mask, contours, index, 0, cv2.FILLED)
    # 只定位二维码，不要求解出内容，避免用left/action1触发路线。
    found, points = cv2.QRCodeDetector().detectMulti(image)
    if found:
        for polygon in points:
            center = polygon.mean(axis=0)
            polygon = center + (polygon-center)*1.3
            cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 0)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_line(image, color="white", previous_x=None, band=(.8, 1.0), mask=None):
    height, width = image.shape[:2]
    top = int(height * band[0])
    if mask is None:
        mask = line_mask(image, color)
    roi = mask[top:int(height * band[1])]
    contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    candidates = []
    for contour in contours:
        _, _, _, h = cv2.boundingRect(contour)
        # 旋转包围矩形沿着线条本身量长度和粗细，斜线不会因水平跨度大被排除。
        _, (side_a, side_b), _ = cv2.minAreaRect(contour)
        thickness = min(side_a, side_b)
        length = max(side_a, side_b)
        if (cv2.contourArea(contour) < 50 * width / 640
                or h < roi.shape[0]*.55
                or thickness > width*.08 or length < thickness*2):
            continue
        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = moments["m10"] / moments["m00"]
            if previous_x is None or abs(cx-previous_x) <= width*.18:
                candidates.append(cx)
    if not candidates:
        return None
    anchor = width / 2 if previous_x is None else previous_x
    # 优先跟随上一次线的位置，减少跳到旁边地板图案。
    return min(candidates, key=lambda x: abs(x - anchor))


def route_segments(mask, mirrored=False, recovering=False):
    """在整个腹部画面分开找竖直段和向实际右方延伸的斜段。"""
    height, width = mask.shape
    filtered = mask.copy()
    contours = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    for contour in contours:
        _, (a,b), _ = cv2.minAreaRect(contour)
        # 排除宽厚实心亮块；相连的折线占包围矩形面积较小，不能一起删掉。
        if min(a,b) > width*.08 and a*b and cv2.contourArea(contour)/(a*b) > .6:
            cv2.drawContours(filtered,[contour],0,0,cv2.FILLED)
    edges = cv2.Canny(filtered, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(10, round(width*20/640)),
                            minLineLength=max(12, round(height*.06)),
                            maxLineGap=max(3, round(height*.02)))
    vertical, diagonal = [], []
    if lines is None:
        return vertical, diagonal
    direction = -1 if mirrored else 1
    # OpenCV可能返回(N,1,4)或(N,4)，先统一成每行四个坐标。
    for x1, y1, x2, y2 in lines.reshape(-1,4):
        # 统一从下向上看，正角度为向画面右上方延伸。
        if y1 < y2:
            x1,y1,x2,y2 = x2,y2,x1,y1
        angle = math.degrees(math.atan2(float(x2-x1), float(y1-y2)))
        segment = tuple(map(int,(x1,y1,x2,y2)))
        # 转完弯后，倾斜的新路线也需要进入角度纠偏；不能只接受竖直线。
        angle_limit = 80 if recovering else 20
        if abs(angle) <= angle_limit:
            vertical.append(segment)
        elif 20 <= angle*direction <= 80:
            diagonal.append(segment)
    return vertical, diagonal


def near_line_sample(segments, height, width, previous_x=None):
    """只用底部25%的线段计算位置和角度，坐标仍对应完整画面。"""
    top = height*.75
    candidates = []
    for x1, y1, x2, y2 in segments:  # 端点已经按从下到上排列。
        sample_top = max(y2, top)
        if y1-sample_top < height*.04:
            continue
        top_x = x1+(x2-x1)*(y1-sample_top)/(y1-y2)
        x = (x1+top_x)/2
        angle = math.degrees(math.atan2(x2-x1, y1-y2))
        candidates.append((x, angle))
    if not candidates:
        return None, 0
    anchor = width/2 if previous_x is None else previous_x
    return min(candidates, key=lambda item:abs(item[0]-anchor))


def stripe_width(mask, x, y, angle):
    """沿白带的法线量宽度；Hough端点在边缘，允许向两侧寻找白像素。"""
    height, width = mask.shape
    radians = math.radians(angle)
    dx, dy = math.cos(radians), math.sin(radians)

    def white_at(distance):
        px = round(x + distance*dx)
        py = round(y + distance*dy)
        return 0 <= px < width and 0 <= py < height and mask[py, px] != 0

    start = next((offset for offset in (0, 1, -1, 2, -2, 3, -3)
                  if white_at(offset)), None)
    if start is None:
        return None
    left = right = start
    limit = round(width*.12)
    while left > start-limit and white_at(left-1):
        left -= 1
    while right < start+limit and white_at(right+1):
        right += 1
    if left == start-limit or right == start+limit:
        return None
    return right-left+1


class LineWidthReference:
    """前进后积累10帧有效近线，逐高度保存宽度；完成后不再更新。"""
    def __init__(self):
        self.frames = []
        self.widths = {}
        self.previous = None
        self.ready = False

    def sample(self, mask, segments):
        if self.ready:
            return
        height, width = mask.shape
        x, angle = near_line_sample(segments, height, width)
        if x is None or abs(angle) > 10:
            self.previous = None
            return
        if self.previous is not None:
            old_x, old_angle = self.previous
            if abs(x-old_x) > width*.06 or abs(angle-old_angle) > 10:
                self.previous = (x, angle)
                return  # 跳变帧不采，保留之前有效采样。
        self.previous = (x, angle)
        near_segments = [segment for segment in segments
                         if near_line_sample([segment], height, width)[0] is not None]
        anchor = min(near_segments,
                     key=lambda segment: abs(near_line_sample([segment], height, width)[0]-x))
        anchor_x1, anchor_y1, anchor_x2, anchor_y2 = anchor
        # 从选中的近线向上找同一路线，仅记录实际有线段覆盖的高度。
        values = {}
        for y in range(12, height-12, 24):
            expected_x = anchor_x1+(anchor_x2-anchor_x1)*(anchor_y1-y)/(anchor_y1-anchor_y2)
            candidates = []
            for x1, y1, x2, y2 in segments:
                if y2 <= y <= y1 and y1 != y2:
                    segment_x = x1+(x2-x1)*(y1-y)/(y1-y2)
                    if abs(segment_x-expected_x) <= width*.04:
                        candidates.append((abs(segment_x-expected_x), segment_x,
                                           math.degrees(math.atan2(x2-x1, y1-y2))))
            if candidates:
                _, segment_x, segment_angle = min(candidates)
                measured = stripe_width(mask, segment_x, y, segment_angle)
                if measured is not None:
                    values[y] = measured
        if not any(y >= height*.75 for y in values):
            return
        self.frames.append(values)
        if len(self.frames) == 10:
            for y in sorted({row for frame in self.frames for row in frame}):
                samples = [frame[y] for frame in self.frames if y in frame]
                if len(samples) >= 7:
                    self.widths[y] = float(np.median(samples))
            self.ready = True

    def filter(self, mask, segments):
        if not self.ready:
            return segments
        accepted = []
        for segment in segments:
            x1, y1, x2, y2 = segment
            angle = math.degrees(math.atan2(x2-x1, y1-y2))
            matches = []
            for y, reference in self.widths.items():
                if y2 <= y <= y1 and y1 != y2:
                    x = x1+(x2-x1)*(y1-y)/(y1-y2)
                    measured = stripe_width(mask, x, y, angle)
                    matches.append(measured is not None and
                                   reference*.7 <= measured <= reference*1.3)
            # 没采过的高度放行；多数取样点符合宽度才接受，容许局部阴影或弯角。
            if not matches or sum(matches) > len(matches)/2:
                accepted.append(segment)
        return accepted


class LinePlanner:
    def __init__(self):
        self.previous_x = None
        self.previous_angle = None
        self.vertical_frames = 0
        self.diagonal_frames = 0
        self.seen_vertical = False
        self.after_bend = False

    def decide(self, x, width, right_diagonal=False, angle=0, mirrored=False, original_visible=False):
        if x is not None:
            stable = (self.previous_x is not None
                      and abs(x-self.previous_x) <= width*.06
                      and abs(angle-self.previous_angle) <= 10)
            self.vertical_frames = self.vertical_frames+1 if stable else 1
            self.previous_x = x
            self.previous_angle = angle
            self.diagonal_frames = 0
            if self.vertical_frames < 2:
                return "WAIT"
            if abs(angle) <= 10:
                self.seen_vertical = True
                self.after_bend = False
            # 先平移居中，即使当前角度偏大也优先处理横向偏差。
            offset = x-width/2
            if abs(offset) > width*60/640:
                if mirrored:
                    offset = -offset
                return "SIDE_LEFT" if offset > 0 else "SIDE_RIGHT"
            # 位置进入中心范围后，再单次转向调角度，随后重新确认新图。
            if abs(angle) > 10:
                turn_angle = -angle if mirrored else angle
                self.previous_x = None
                self.previous_angle = None
                self.vertical_frames = 0
                return "TURN_RIGHT" if turn_angle > 0 else "TURN_LEFT"
            # 原竖直线仍在时，弯道斜线不能触发提前转向。
            return "UP_LITTLE"
        self.previous_x = None
        self.previous_angle = None
        self.vertical_frames = 0
        # 近处取样不到，但原直线还在画面上方时，不能当作已到弯口。
        if original_visible:
            self.diagonal_frames = 0
            # 已跟稳原路线时继续向前，让弯口到达脚下；转弯后先等近线调正。
            return "UP_LITTLE" if self.seen_vertical and not self.after_bend else "WAIT"
        if right_diagonal and self.seen_vertical and not self.after_bend:
            self.diagonal_frames += 1
            if self.diagonal_frames >= 2:
                self.diagonal_frames = 0
                self.after_bend = True
                return "TURN_RIGHT_2"
        else:
            self.diagonal_frames = 0
        return "WAIT"


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
    io.phase("人脸完成后前进3步回到寻线起点")
    io.forward(3)
    return run_from_line(io, color)


def run_from_line(io, color):
    """从当前位置寻线到跳舞；正式比赛和临时测试共用后半程。"""
    io.phase("开始寻线，寻找指定颜色物块（尚未进入搬运）")
    io.follow_to_carry(color)
    return run_from_carry(io, color)


def run_from_carry(io, color):
    """从当前位置找物搬运，再连续执行足球和舞蹈。"""
    settings = io.settings
    io.phase("指定颜色搬运")
    io.carry(color, settings["drop_qr"])
    io.phase("放下完成，视觉找球后开始足球")
    io.sport()
    io.phase("右转寻找dance，最多9次")
    io.find_dance()
    io.forward(10)
    io.dance()
    return True
