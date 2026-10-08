"""借用同学line_follow的黑线质心检测，仅返回一次纠偏建议，不打开硬件。"""
import cv2
import numpy as np


def correction_action(image):
    height, width = image.shape[:2]
    roi = image[int(height * .8):, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, columns = np.where(mask > 0)
    # 沿用旧版640宽画面的阈值，随当前分辨率缩放。
    scale = width / 640
    if len(columns) < 50 * scale * height / 480 or columns.std() > 150 * scale:
        return None
    error = float(columns.mean()) - width / 2
    if abs(error) < 60 * scale:
        return None
    if abs(error) < 100 * scale:
        return "SIDE_LEFT" if error < 0 else "SIDE_RIGHT"
    return "TURN_LEFT" if error < 0 else "TURN_RIGHT"
