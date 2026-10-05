"""不依赖颜色的圆形物体检测。"""

import math
import os
from robot_config import ROOT
from dataclasses import dataclass

import cv2


DEFAULT_MIN_RADIUS = int(os.environ.get("ROBOT_BALL_MIN_RADIUS", "3"))
DEFAULT_MAX_RADIUS = int(os.environ.get("ROBOT_BALL_MAX_RADIUS", "135"))
DEFAULT_MIN_CIRCULARITY = float(os.environ.get(
    "ROBOT_BALL_MIN_CIRCULARITY", "0.55"
))


@dataclass
class RoundObject:
    centerX: int
    centerY: int
    width: int
    height: int
    circularity: float


def _round_candidates(frame, min_radius, max_radius, min_circularity, strict=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)
    contours = cv2.findContours(
        edges, cv2.RETR_LIST if strict else cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if not (min_radius <= radius <= max_radius and
                circularity >= min_circularity):
            continue
        if strict:
            _, _, width, height = cv2.boundingRect(contour)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            # 门框、方块也可能有较高圆度，踢球不能只靠原来的单一圆度阈值。
            if (len(polygon) < 6 or max(width, height) / max(1, min(width, height)) > 1.3
                    or area / (math.pi * radius * radius) < 0.75):
                continue
        score = area * circularity
        diameter = round(radius * 2)
        candidates.append((score, RoundObject(round(center_x), round(center_y),
                                             diameter, diameter, circularity)))
    return sorted(candidates, key=lambda item: item[0], reverse=True)


def find_round_objects(frame):
    """踢球用全部候选；合并同一个圆的内外边缘，不把多个球强行选成一个。"""
    objects = []
    for _, candidate in _round_candidates(frame, DEFAULT_MIN_RADIUS,
                                          DEFAULT_MAX_RADIUS, 0.7, strict=True):
        if not any(math.hypot(candidate.centerX - old.centerX, candidate.centerY - old.centerY)
                   < max(5, min(candidate.width, old.width) * 0.25)
                   and abs(candidate.width - old.width) < max(6, old.width * 0.25)
                   for old in objects):
            objects.append(candidate)
    return objects


def find_round_object(frame, min_radius=DEFAULT_MIN_RADIUS,
                      max_radius=DEFAULT_MAX_RADIUS,
                      min_circularity=DEFAULT_MIN_CIRCULARITY):
    candidates = _round_candidates(frame, min_radius, max_radius, min_circularity)
    best = candidates[0][1] if candidates else None

    if best is not None:
        cv2.circle(frame, (best.centerX, best.centerY), best.width // 2,
                   (0, 255, 255), 2)
    return best, frame


def main():
    """只预览圆形识别，不驱动机器人。"""
    from roboteye import RobotEye

    eye = RobotEye()
    try:
        while True:
            ok, frame = eye.getImage()
            if not ok:
                continue
            target, frame = find_round_object(frame)
            if target is not None:
                message = "circle=({}, {}) size={} score={:.2f}".format(
                    target.centerX, target.centerY, target.width,
                    target.circularity,
                )
                cv2.putText(frame, message, (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            eye.showImage(frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return True
    finally:
        eye.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
