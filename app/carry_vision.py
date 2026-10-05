"""选颜色找方块，抱起后送到指定二维码；预览不发送身体动作。"""
import argparse
from contextlib import ExitStack
import json
import time

import cv2
import numpy as np
from robot_config import ROOT
from roboteye import RobotEye
from dual_kick import camera_settings
from kick_shapes import Box, overlap

INITIAL_HEAD_POSITION = 129
BODY_SERIAL_PORT = '/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0'
REFERENCE_FILE = ROOT/'config/carry_reference.json'
COLORS = {
    'red': [(0,10),(170,179)],
    'green': [(35,85)],
    'blue': [(95,130)],
    'yellow': [(20,34)],
    'pink': [(140,169)],
}


def find_blocks(frame, color):
    """颜色限定身份，凸四至六边形限定方块外观；不声称恢复三维立方体。"""
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2],np.uint8)
    for low,high in COLORS[color]:
        mask |= cv2.inRange(hsv,(low,80,60),(high,255,255))
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    contours = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    blocks = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        if area < 250 or min(w,h) < 15 or not .5 <= w/h <= 2:
            continue
        if w*h > frame.shape[0]*frame.shape[1]*.35 or area/(w*h) < .55:
            continue
        hull = cv2.convexHull(contour)
        polygon = cv2.approxPolyDP(hull,.03*cv2.arcLength(hull,True),True)
        if 4 <= len(polygon) <= 6 and area/cv2.contourArea(hull) >= .8:
            blocks.append(Box(x,y,w,h))
    return blocks


def read_qr_codes(frame, detector):
    """只解码内容和位置，不执行二维码中的文本或网址。"""
    found,contents,points,_ = detector.detectAndDecodeMulti(frame)
    if not found or points is None:
        return []
    codes = []
    for content,corners in zip(contents,points):
        if not content:
            continue
        x,y,w,h = cv2.boundingRect(corners.astype(np.float32))
        codes.append((content,Box(x,y,w,h)))
    return codes


class StableTarget:
    def __init__(self):
        self.reset()

    def reset(self):
        self.box = None
        self.frames = 0

    def update(self, candidates):
        if len(candidates) != 1:
            self.reset()
            return None
        box = candidates[0]
        stable = self.box is not None and overlap(self.box,box) >= .65
        self.frames = self.frames+1 if stable else 1
        self.box = box
        return box


def steering(action, flip):
    if flip in ('1','-1'):
        action = {'SIDE_LEFT':'SIDE_RIGHT','SIDE_RIGHT':'SIDE_LEFT',
                  'LEFT_HOLDBOX':'RIGHT_HOLDBOX','RIGHT_HOLDBOX':'LEFT_HOLDBOX'}.get(action,action)
    # 沿用已经实机纠正的横移方向；抱物转向单独保留原映射。
    if action == 'SIDE_LEFT': return 'SIDE_RIGHT'
    if action == 'SIDE_RIGHT': return 'SIDE_LEFT'
    return action


class CarryPlanner:
    def __init__(self, reference, flip):
        self.reference = reference
        self.flip = flip
        self.phase = 'PICKUP'
        self.actions = 0
        self.started_at = time.monotonic()
        self.pending = None
        self.confirm_frames = 0
        self.reason = 'looking for selected block color'

    def reset_confirmation(self):
        self.pending = None
        self.confirm_frames = 0

    def decide(self, box, stable_frames, now=None):
        now = time.monotonic() if now is None else now
        if self.phase == 'DONE': return 'DONE'
        if self.actions >= 40 or now-self.started_at >= 180:
            self.reason = '40 actions / 180 seconds limit'
            return 'STOP'
        if box is None or stable_frames < 5:
            self.reset_confirmation()
            self.reason = 'target missing, multiple or unstable; hold body'
            return 'WAIT'
        target = Box(*self.reference['pickup' if self.phase == 'PICKUP' else 'drop'])
        dx,dy = box.cx-target.cx,box.cy-target.cy
        ratio = box.width/target.width
        if ratio > 1.3 or dy > .08:
            self.reason = 'past calibrated position; stop'
            return 'STOP'
        if abs(dx) > .06:
            if self.phase == 'PICKUP':
                action = 'SIDE_LEFT' if dx < 0 else 'SIDE_RIGHT'
            else:
                action = 'LEFT_HOLDBOX' if dx < 0 else 'RIGHT_HOLDBOX'
            action = steering(action,self.flip)
            self.reason = 'align block' if self.phase == 'PICKUP' else 'turn with block toward target QR'
        elif ratio < .8 or dy < -.07:
            action = 'UP_LITTLE' if self.phase == 'PICKUP' else 'UP_HOLDBOX'
            self.reason = 'approach calibrated position'
        else:
            action = 'HOLD_BOX' if self.phase == 'PICKUP' else 'DOWN_BOX'
            self.reason = 'pickup position confirmed' if self.phase == 'PICKUP' else 'drop position confirmed'
        self.confirm_frames = self.confirm_frames+1 if self.pending == action else 1
        self.pending = action
        return action if self.confirm_frames >= 5 else 'WAIT'

    def mark_sent(self, action):
        # 在发送前改变阶段：失败即退出，不重复抱取/放下。
        if action == 'HOLD_BOX': self.phase = 'DELIVER'
        if action == 'DOWN_BOX': self.phase = 'DONE'
        self.actions += 1
        self.reset_confirmation()


def run(color, target_qr, actions=False):
    settings = camera_settings()['head']
    reference = None
    if REFERENCE_FILE.exists():
        reference = json.loads(REFERENCE_FILE.read_text())
    if actions:
        if (reference is None or reference.get('version') != 1
                or reference.get('camera') != settings
                or reference.get('head_position') != INITIAL_HEAD_POSITION
                or reference.get('target_qr') != target_qr
                or 'pickup' not in reference or 'drop' not in reference):
            raise ValueError('先在预览中按 H 标定抱取位置、D 标定目标二维码放下位置')
    planner = CarryPlanner(reference,settings['flip']) if actions else None
    block_tracker,qr_tracker = StableTarget(),StableTarget()
    detector = cv2.QRCodeDetector()
    seen_contents = set()
    with ExitStack() as stack:
        eye = RobotEye(**settings,latest=True); stack.callback(eye.close)
        ok,frame = eye.getImage()
        if not ok: raise RuntimeError('Camera read failed before startup')
        if actions and reference.get('shape') != list(frame.shape[:2]):
            raise ValueError('画面尺寸变化，请重新标定搬运位置')
        if actions:
            from robotmove import RobotMove
            move = RobotMove(None,port=BODY_SERIAL_PORT); stack.callback(move.close)
        from Head import RobotHeadServoOnly
        head = RobotHeadServoOnly(hold=True); stack.callback(head.cleanup)
        head.turn_vertical(INITIAL_HEAD_POSITION)
        eye.discard_frames(1)
        stack.callback(cv2.destroyAllWindows)
        ready_at = time.monotonic()+.5
        print('目标颜色:',color,'目的地二维码:',target_qr)
        print('H=保存抱取位置；D=保存放下位置；Q=退出。预览不发送身体动作。')
        while True:
            ok,frame = eye.getImage()
            if not ok: raise RuntimeError('Camera read failed; no further actions')
            blocks = find_blocks(frame,color)
            block = block_tracker.update(blocks)
            codes = read_qr_codes(frame,detector)
            for content,_ in codes:
                if content not in seen_contents:
                    print('读到二维码:',content)
                    seen_contents.add(content)
            matches = [box for content,box in codes if content == target_qr]
            qr = qr_tracker.update(matches)
            display = frame.copy()
            for box in blocks:
                cv2.rectangle(display,(int(box.x),int(box.y)),(int(box.x+box.width),int(box.bottom)),(0,255,0),2)
            for content,box in codes:
                cv2.rectangle(display,(int(box.x),int(box.y)),(int(box.x+box.width),int(box.bottom)),(255,255,0),2)
                cv2.putText(display,content,(int(box.x),max(15,int(box.y)-5)),0,.5,(255,255,0),1)
            phase = planner.phase if actions else 'PREVIEW'
            reason = planner.reason if actions else 'H=pickup reference; D=drop reference; Q=quit'
            cv2.putText(display,f'{phase} color={color} blocks={len(blocks)} target QR={len(matches)}',(8,25),0,.5,(255,255,255),1)
            cv2.putText(display,reason,(8,48),0,.45,(255,255,255),1)
            cv2.imshow('carry',display)
            key = cv2.waitKey(1)&255
            if key in (ord('q'),27): return False
            if not actions and key in (ord('h'),ord('d')):
                box = block if key == ord('h') else qr
                tracker = block_tracker if key == ord('h') else qr_tracker
                if box is None or tracker.frames < 5:
                    print('需要唯一且稳定的目标，当前不能保存。');continue
                # 新建独立标定，不改头部和踢球标定；换二维码需重新标定。
                if (reference is None or reference.get('camera') != settings
                        or reference.get('head_position') != INITIAL_HEAD_POSITION
                        or reference.get('target_qr') != target_qr
                        or reference.get('shape') != list(frame.shape[:2])):
                    reference = dict(version=1,camera=settings,head_position=INITIAL_HEAD_POSITION,
                                     target_qr=target_qr,shape=list(frame.shape[:2]))
                normalized = box.normalized(frame.shape)
                reference['pickup' if key == ord('h') else 'drop'] = [normalized.x,normalized.y,normalized.width,normalized.height]
                REFERENCE_FILE.write_text(json.dumps(reference,ensure_ascii=False,indent=2))
                print('已保存:',REFERENCE_FILE)
            if not actions or time.monotonic() < ready_at: continue
            target = block if planner.phase == 'PICKUP' else qr
            tracker = block_tracker if planner.phase == 'PICKUP' else qr_tracker
            action = planner.decide(target.normalized(frame.shape) if target else None,tracker.frames)
            if action in ('STOP','DONE'):
                print(planner.reason);return action == 'DONE'
            if action == 'WAIT': continue
            planner.mark_sent(action)
            print('搬运动作:',action)
            move.robotMove(action)
            if action == 'DOWN_BOX':
                print('放下动作已执行；物块是否实际到位需要现场确认。');return True
            eye.discard_frames(1)
            block_tracker.reset();qr_tracker.reset()
            ready_at = time.monotonic()+.5


def choose_task(color=None, target_qr=None):
    if color is None:
        print('颜色：red红 / green绿 / blue蓝 / yellow黄 / pink粉')
        color = input('目标颜色: ').strip().lower()
    if color not in COLORS: raise ValueError('请选择 red、green、blue、yellow、pink')
    if target_qr is None: target_qr = input('目的地二维码完整内容: ')
    if not target_qr: raise ValueError('二维码内容不能为空')
    return color,target_qr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color',choices=COLORS)
    parser.add_argument('--target-qr')
    parser.add_argument('--actions',action='store_true')
    args = parser.parse_args()
    color,target_qr = choose_task(args.color,args.target_qr)
    raise SystemExit(0 if run(color,target_qr,args.actions) else 1)
