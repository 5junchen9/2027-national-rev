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
from handover_debug import Handover
from walk_kick import WalkKick
from ball_debug import distinct_from_background

INITIAL_HEAD_POSITION = 129
BODY_SERIAL_PORT = '/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0'
REFERENCE_FILE = ROOT/'config/carry_dual_reference.json'
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
        if not distinct_from_background(contour,hsv): continue
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
    settings = camera_settings()
    reference = json.loads(REFERENCE_FILE.read_text()) if REFERENCE_FILE.exists() else None
    if actions:
        if (reference is None or reference.get('version') != 2
                or reference.get('cameras') != settings
                or reference.get('head_position') != INITIAL_HEAD_POSITION
                or reference.get('target_qr') != target_qr
                or 'pickup' not in reference or 'drop' not in reference):
            raise ValueError('先用双摄预览按 H 标定腹部抱取位置、D 标定头部二维码放下位置')
    planner = CarryPlanner(reference,settings['belly']['flip']) if actions else None
    handover = Handover(dict(forward=INITIAL_HEAD_POSITION,down_sign=1,bounds=[85,180]))
    approach = WalkKick()
    head_tracker,belly_tracker,qr_tracker = StableTarget(),StableTarget(),StableTarget()
    detector = cv2.QRCodeDetector()
    seen_contents = set()
    belly_seen = 0
    with ExitStack() as stack:
        head_eye = RobotEye(**settings['head'],latest=True);stack.callback(head_eye.close)
        belly_eye = RobotEye(**settings['belly'],latest=True);stack.callback(belly_eye.close)
        okh,hf = head_eye.getImage();okb,bf = belly_eye.getImage()
        if not okh or not okb: raise RuntimeError('Camera read failed before startup')
        shapes = dict(head=list(hf.shape[:2]),belly=list(bf.shape[:2]))
        if actions and reference.get('shapes') != shapes:
            raise ValueError('画面尺寸变化，请重新标定双摄搬运位置')
        if actions:
            from robotmove import RobotMove
            move = RobotMove(None,port=BODY_SERIAL_PORT);stack.callback(move.close)
        from Head import RobotHeadServoOnly
        servo = RobotHeadServoOnly(hold=True);stack.callback(servo.cleanup)
        servo.turn_vertical(INITIAL_HEAD_POSITION)
        head_eye.discard_frames(1);belly_eye.discard_frames(1)
        stack.callback(cv2.destroyAllWindows)
        ready_at = time.monotonic()+.5
        print('目标颜色:',color,'目的地二维码:',target_qr)
        print('H=保存腹部抱取位置；D=保存头部二维码放下位置；Q=退出。')
        while True:
            okh,hf = head_eye.getImage();okb,bf = belly_eye.getImage()
            if not okh or not okb: raise RuntimeError('Camera read failed; no further actions')
            if dict(head=list(hf.shape[:2]),belly=list(bf.shape[:2])) != shapes:
                raise ValueError('Camera resolution changed; stop actions')
            head_busy = servo.is_moving()
            head_blocks,belly_blocks = find_blocks(hf,color),find_blocks(bf,color)
            head_block = head_tracker.update(head_blocks)
            belly_block = belly_tracker.update(belly_blocks)
            belly_seen = belly_seen+1 if belly_block is not None else 0
            if planner is not None and planner.phase == 'PICKUP':
                handover.step(None,belly_seen,hf.shape[0])
                if handover.phase == 'BELLY' and belly_block is None:
                    handover.phase = 'HEAD';handover.reset_follow()
            codes = read_qr_codes(hf,detector) if not actions or planner.phase != 'PICKUP' else []
            for content,_ in codes:
                if content not in seen_contents:
                    print('读到二维码:',content);seen_contents.add(content)
            matches = [box for content,box in codes if content == target_qr]
            qr = qr_tracker.update(matches)
            for name,frame,blocks in [('head',hf,head_blocks),('belly',bf,belly_blocks)]:
                display = frame.copy()
                for box in blocks:
                    cv2.rectangle(display,(int(box.x),int(box.y)),(int(box.x+box.width),int(box.bottom)),(0,255,0),2)
                if name == 'head':
                    for content,box in codes:
                        cv2.rectangle(display,(int(box.x),int(box.y)),(int(box.x+box.width),int(box.bottom)),(255,255,0),2)
                        cv2.putText(display,content,(int(box.x),max(15,int(box.y)-5)),0,.5,(255,255,0),1)
                phase = planner.phase if actions else 'PREVIEW'
                reason = planner.reason if actions else 'H=belly pickup; D=head QR drop; Q=quit'
                cv2.putText(display,f'{phase} {handover.phase} color={color} head={handover.angle}',(8,25),0,.5,(255,255,255),1)
                cv2.putText(display,reason,(8,48),0,.45,(255,255,255),1)
                cv2.imshow('carry '+name,display)
            key = cv2.waitKey(1)&255
            if key in (ord('q'),27): return False
            if not actions and key in (ord('h'),ord('d')):
                box = belly_block if key == ord('h') else qr
                tracker = belly_tracker if key == ord('h') else qr_tracker
                if box is None or tracker.frames < 5:
                    print('需要唯一且稳定的目标，当前不能保存。');continue
                if (reference is None or reference.get('version') != 2
                        or reference.get('cameras') != settings or reference.get('shapes') != shapes
                        or reference.get('head_position') != INITIAL_HEAD_POSITION
                        or reference.get('target_qr') != target_qr):
                    reference = dict(version=2,cameras=settings,head_position=INITIAL_HEAD_POSITION,
                                     target_qr=target_qr,shapes=shapes)
                frame = bf if key == ord('h') else hf
                normalized = box.normalized(frame.shape)
                reference['pickup' if key == ord('h') else 'drop'] = [normalized.x,normalized.y,normalized.width,normalized.height]
                REFERENCE_FILE.write_text(json.dumps(reference,ensure_ascii=False,indent=2))
                print('已保存:',REFERENCE_FILE)
            if not actions or head_busy or time.monotonic() < ready_at: continue
            if planner.actions >= 40 or time.monotonic()-planner.started_at >= 180:
                print('40 actions / 180 seconds limit');return False
            if planner.phase == 'PICKUP' and handover.phase == 'HEAD':
                box = (head_block.x,head_block.y,head_block.width,head_block.height) if head_block else None
                action = approach.decide('HEAD',box,None,head_tracker.frames,0,hf.shape[:2],settings['head']['flip'])
                # 任一相机有候选时保持头位；多目标等待，不盲选或继续扫描。
                if head_blocks or belly_blocks:
                    approach.reset_search()
                    if belly_blocks or action in ('REACQUIRE','LOWER_HEAD','RAISE_HEAD'): action = 'WAIT'
                planner.reason = approach.reason
                if action in ('REACQUIRE','LOWER_HEAD','RAISE_HEAD'):
                    if action != 'REACQUIRE':
                        direction = 1 if action == 'LOWER_HEAD' else -1
                        angle = max(handover.up_limit,min(handover.down,handover.angle+3*direction))
                        moved = angle != handover.angle
                        if moved:
                            handover.angle = angle;servo.begin_vertical(angle,settle_seconds=.18)
                        approach.mark_head_search(moved)
                    planner.reason = approach.reason
                    continue
            else:
                target = belly_block if planner.phase == 'PICKUP' else qr
                tracker = belly_tracker if planner.phase == 'PICKUP' else qr_tracker
                planner.flip = settings['belly' if planner.phase == 'PICKUP' else 'head']['flip']
                action = planner.decide(target.normalized(bf.shape if planner.phase == 'PICKUP' else hf.shape) if target else None,tracker.frames)
            if action in ('STOP','DONE'):
                print(planner.reason);return action == 'DONE'
            if action == 'WAIT': continue
            planner.mark_sent(action)
            approach.reset_observation()
            print('搬运动作:',action);move.robotMove(action)
            if action == 'DOWN_BOX':
                print('放下动作已执行；物块是否实际到位需要现场确认。');return True
            if action == 'HOLD_BOX':
                # 抱取完成后目标改为头部二维码，恢复与D标定相同的头位。
                handover.angle = INITIAL_HEAD_POSITION;servo.begin_vertical(INITIAL_HEAD_POSITION)
            head_eye.discard_frames(1);belly_eye.discard_frames(1)
            head_tracker.reset();belly_tracker.reset();qr_tracker.reset()
            belly_seen = 0;handover.reset_follow()
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
