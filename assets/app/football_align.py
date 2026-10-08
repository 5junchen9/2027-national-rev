"""足球区入场粗对齐：双层白边框 + 中线/中圈；不使用SVG尺寸。"""
import time

import cv2
import numpy as np

from dual_kick import camera_settings
from carry_vision import steering


def ordered_quad(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    points = points[np.argsort(points[:, 1])]
    top = points[:2][np.argsort(points[:2, 0])]
    bottom = points[2:][np.argsort(points[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], np.float32)


def has_midfield(mask, quad):
    """拉正候选后检查横向中线与椭圆环；只有矩形或两条长线不够。"""
    target = np.float32([[0, 0], [299, 0], [299, 419], [0, 419]])
    transform = cv2.getPerspectiveTransform(quad, target)
    court = cv2.warpPerspective(mask, transform, (300, 420))
    rows = (court[150:270, 30:270] > 0).mean(axis=1)
    middle = int(np.argmax(rows)) + 150
    if rows.max() < .65:
        return False
    angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    # 不知道实际长宽，允许中圈在拉正图里呈不同扁率的椭圆。
    thick = cv2.dilate(court, np.ones((5, 5), np.uint8))
    for radius_x in (30, 40, 50, 60):
        for radius_y in (20, 30, 40, 50, 60):
            x = np.rint(150 + radius_x*np.cos(angles)).astype(int)
            y = np.rint(middle + radius_y*np.sin(angles)).astype(int)
            hits = thick[y, x] > 0
            if all(part.mean() > .55 for part in np.array_split(hits, 4)):
                return True
    return False


def find_court(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 130), (179, 75, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    quads = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not width*height*.035 < area < width*height*.9:
            continue
        polygon = cv2.approxPolyDP(contour, .025*cv2.arcLength(contour, True), True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            quad = ordered_quad(polygon)
            if np.any(quad[:, 0] < 3) or np.any(quad[:, 0] > width-4) or np.any(quad[:, 1] < 3) or np.any(quad[:, 1] > height-4):
                continue  # 画面裁掉的边界不能当作完整球场。
            quads.append((area, quad))
    candidates = []
    for area, inner in quads:
        for outer_area, outer in quads:
            if not .65 < area/outer_area < .95:
                continue
            # 排除同一根粗白线的两条轮廓；要求四角有独立的框间间距。
            gaps = [cv2.pointPolygonTest(outer, tuple(map(float, p)), True) for p in inner]
            if min(gaps) < max(3, width*.006):
                continue
            if has_midfield(mask, inner):
                candidates.append(inner)
                break
    # 同一内框的内外轮廓去重；多个不同球场候选时拒绝猜选。
    unique = []
    for quad in candidates:
        if not any(np.linalg.norm(quad-old, axis=1).mean() < width*.025 for old in unique):
            unique.append(quad)
    return unique[0] if len(unique) == 1 else None


def alignment_action(quad, shape, mirrored=False):
    height, width = shape[:2]
    points = np.column_stack((quad, np.ones(4)))
    left = np.cross(points[0], points[3])
    right = np.cross(points[1], points[2])
    vanishing = np.cross(left, right)
    # 两长边交点表示纵向方向；用齐次坐标兼容近似平行的投影。
    dx = vanishing[0] - width/2*vanishing[2]
    dy = height*vanishing[2] - vanishing[1]
    if abs(dy) < 1e-6:
        return 'WAIT'  # 方向退化，不能据此宣告对齐。
    heading = dx/dy
    if abs(heading) > .12:
        action = 'TURN_RIGHT' if heading > 0 else 'TURN_LEFT'
        if mirrored:
            action = 'TURN_LEFT' if action == 'TURN_RIGHT' else 'TURN_RIGHT'
        return action
    near_center = (quad[2, 0]+quad[3, 0])/2
    near_width = quad[2, 0]-quad[3, 0]
    offset = (near_center-width/2)/near_width
    if abs(offset) > .15:
        # 沿用现有实机横移映射；镜像单独处理。
        return steering('SIDE_RIGHT' if offset > 0 else 'SIDE_LEFT', '1' if mirrored else 'none')
    return 'DONE'


def run(io):
    """复用比赛双摄，动作后重新确认；缺失时不盲目前进。"""
    io.set_head(io.settings['sport_head_position'])
    deadline = min(io.deadline, time.monotonic()+60)
    pending, frames, previous = None, 0, None
    seen = False
    missing = 0
    searches = 0
    actions = 0
    mirrored = camera_settings()['head']['flip'] in ('1', '-1')
    while time.monotonic() < deadline and actions < 20:
        head, _, _ = io.observe_ready()
        quad = find_court(head)
        display = head.copy()
        if quad is None:
            pending, frames, previous = None, 0, None
            missing += 1
            cv2.putText(display, 'court not confirmed', (10, 30), 0, .7, (0, 165, 255), 2)
            cv2.imshow('football alignment', display)
            if missing < 3:
                continue
            if seen or searches >= 12:
                raise RuntimeError('足球区边线丢失或搜索失败，停止比赛')
            # 从投放朝向向右搜索，再向左回扫；次数不是角度标定。
            io.move('TURN_RIGHT' if searches < 4 else 'TURN_LEFT')
            searches += 1
            actions += 1
            missing = 0
            continue
        seen = True
        missing = 0
        action = alignment_action(quad, head.shape, mirrored)
        same = previous is not None and np.linalg.norm(quad-previous, axis=1).mean() < head.shape[1]*.035
        frames = frames+1 if action == pending and same else 1
        pending, previous = action, quad
        cv2.polylines(display, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(display, f'{action} {frames}/3', (10, 30), 0, .7, (0, 255, 0), 2)
        cv2.imshow('football alignment', display)
        if frames < 3:
            continue
        if action == 'WAIT':
            continue
        if action == 'DONE':
            return True
        io.move(action)
        actions += 1
        pending, frames, previous = None, 0, None
    raise RuntimeError('足球区对齐超时或超过动作上限，停止比赛')


def preview():
    """仅打开头部相机，不连接身体串口，不控制头部舵机。"""
    from roboteye import RobotEye
    settings = camera_settings()['head']
    eye = RobotEye(**settings, latest=True)
    print('仅预览足球区识别；不发送身体或头部动作。Q退出。')
    try:
        while True:
            ok, frame = eye.getImage()
            if not ok:
                raise RuntimeError('头部相机未返回图像')
            quad = find_court(frame)
            action = 'court not confirmed'
            if quad is not None:
                cv2.polylines(frame, [quad.astype(np.int32)], True, (0, 255, 0), 2)
                action = alignment_action(quad, frame.shape, settings['flip'] in ('1', '-1'))
            cv2.putText(frame, action, (10, 30), 0, .7, (0, 255, 0), 2)
            cv2.imshow('football alignment preview', frame)
            if cv2.waitKey(1) & 255 in (ord('q'), 27):
                break
    finally:
        eye.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    preview()
