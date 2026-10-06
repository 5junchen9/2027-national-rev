"""选颜色找方块，抱起后送到指定二维码；预览不发送身体动作。"""
import argparse
from contextlib import ExitStack
import json
import math
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

INITIAL_HEAD_POSITION = 123
DELIVERY_HEAD_POSITION = 120
TRACK_FRAMES = 2
PICKUP_FRAMES = 2
FINAL_FRAMES = 5  # 放下与保存标定仍确认5帧。
NEAR_BOTTOM = .88
TRANSFER_STEPS = 3  # 只限制近处目标丢失后的无目标小步，不限制看得见目标时的正常靠近。
PICKUP_TOP = .30  # 底边出画后，用可见上沿的试验线判断；需现场验证。
DROP_BOTTOM = .85  # 临时近处标准：120头位，二维码下沿到画面下方。
BODY_SERIAL_PORT = '/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0'
REFERENCE_FILE = ROOT/'config/carry_dual_reference.json'
COLORS = {
    'red': [(0,10),(170,179)],
    'green': [(35,85)],
    'blue': [(85,130)],
    'yellow': [(20,34)],
    'pink': [(140,169)],
}


def merge_box_parts(contours):
    """合并上下紧邻、宽度接近的盒盖和盒身；不合并左右并排物块。"""
    parts = [contour for contour in contours if cv2.contourArea(contour) >= 250]
    groups = []
    while parts:
        group = [parts.pop(0)]
        changed = True
        while changed:
            changed = False
            x,y,w,h = cv2.boundingRect(np.vstack(group))
            for index,contour in enumerate(parts):
                px,py,pw,ph = cv2.boundingRect(contour)
                horizontal = max(0,min(x+w,px+pw)-max(x,px))
                gap = max(0,max(y,py)-min(y+h,py+ph))
                vertical = max(0,min(y+h,py+ph)-max(y,py))
                containment = horizontal*vertical/min(w*h,pw*ph)
                if containment >= .85 or (horizontal >= .75*min(w,pw) and min(w,pw)/max(w,pw) >= .55
                        and gap <= .10*min(w,pw)):
                    group.append(parts.pop(index));changed = True
                    break
        groups.append(group)
    return groups


def find_blocks(frame, color, near=False):
    """远处排除横向出画的地垫；腹部允许大型盒面，但仍要求可见边缘。"""
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2],np.uint8)
    for low,high in COLORS[color]:
        mask |= cv2.inRange(hsv,(low,80,60),(high,255,255))
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    contours = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    blocks = []
    height,width = frame.shape[:2]
    for group in merge_box_parts(contours):
        points = np.vstack(group)
        area = sum(cv2.contourArea(contour) for contour in group)
        x,y,w,h = cv2.boundingRect(points)
        if min(w,h) < 15 or not .25 <= w/h <= 4 or area/(w*h) < .40:
            continue
        if near and y+h >= height-3 and area < height*width*.03:
            continue
        touches_side = x <= 5 or x+w >= width-5
        if not near and (touches_side or w*h > height*width*.85):
            continue
        # 全屏蓝色没有足够边缘信息，不能当作近处盒子。
        if x <= 2 and y <= 2 and x+w >= width-2 and y+h >= height-2:
            continue
        if not any(distinct_from_background(contour,hsv) for contour in group):
            continue
        hull = cv2.convexHull(points)
        polygon = cv2.approxPolyDP(hull,.02*cv2.arcLength(hull,True),True)
        clipped = touches_side or y <= 2 or y+h >= height-2
        min_corners = 3 if clipped else 4
        if min_corners <= len(polygon) <= 6 and area/cv2.contourArea(hull) >= .65:
            blocks.append(Box(x,y,w,h))
    return remove_nested_boxes(blocks)


def remove_nested_boxes(boxes):
    """同一物块的内外候选去重；按小框被包含比例，而非IoU判断。"""
    kept = []
    for box in sorted(boxes,key=lambda item:item.width*item.height,reverse=True):
        duplicate = False
        for outer in kept:
            horizontal = max(0,min(box.x+box.width,outer.x+outer.width)-max(box.x,outer.x))
            vertical = max(0,min(box.bottom,outer.bottom)-max(box.y,outer.y))
            if horizontal*vertical/(box.width*box.height) >= .85:
                duplicate = True
                break
        if not duplicate: kept.append(box)
    return kept


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
        stable = self.box is not None and overlap(self.box,box) >= .45
        self.frames = self.frames+1 if stable else 1
        self.box = box
        return box


class DestinationTracker(StableTarget):
    """精确解码确认身份；短暂漏读时必须在新画面里匹配到码图案才返回位置。"""
    def reset(self):
        super().reset()
        self.template = None
        self.valid_until = 0.0
        self.source = 'missing'

    def after_move(self):
        self.box = None
        self.frames = 0
        self.valid_until = time.monotonic()+3.0

    def update(self, candidates, frame=None, now=None):
        now = time.monotonic() if now is None else now
        self.source = 'missing'
        if len(candidates) == 1:
            box = super().update(candidates)
            self.source = 'decoded'
            self.valid_until = now+3.0
            if frame is not None:
                x,y,w,h = map(int,(box.x,box.y,box.width,box.height))
                patch = frame[max(0,y):y+h,max(0,x):x+w]
                if min(patch.shape[:2]) >= 15:
                    gray = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
                    if gray.std() >= 15: self.template = gray.copy()
            return box
        if len(candidates) > 1 or frame is None or self.template is None or now > self.valid_until:
            return super().update([])
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        best_score,best_box = 0.0,None
        for scale in (.85,1.0,1.15):
            template = cv2.resize(self.template,None,fx=scale,fy=scale)
            h,w = template.shape
            if h >= gray.shape[0] or w >= gray.shape[1]: continue
            scores = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
            _,score,_,position = cv2.minMaxLoc(scores)
            if score > best_score:
                best_score = score
                best_box = Box(position[0],position[1],w,h)
                x,y = position
                scores[max(0,y-h//2):y+h//2+1,max(0,x-w//2):x+w//2+1] = -1
                _,second,_,_ = cv2.minMaxLoc(scores)
                if second >= .88 and score-second < .08: best_box = None
        if best_score < .88 or best_box is None: return super().update([])
        self.source = 'tracked'
        self.valid_until = now+3.0
        return super().update([best_box])


def block_appearance(frame, box):
    """只统计目标框内饱和色像素的HSV分布，减少木地板和白贴纸影响。"""
    x,y,w,h = map(int,(box.x,box.y,box.width,box.height))
    patch = frame[max(0,y):y+h,max(0,x):x+w]
    if patch.size == 0: return None
    hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(0,80,60),(179,255,255))
    if cv2.countNonZero(mask) < 30: return None
    histogram = cv2.calcHist([hsv],[0,1],mask,[18,8],[0,180,0,256])
    return histogram/cv2.sumElems(histogram)[0]


def block_quality(frame, box, color):
    """返回颜色、形状分数；是规则匹配分，不是识别概率。"""
    x,y,w,h = map(int,(box.x,box.y,box.width,box.height))
    patch = frame[max(0,y):y+h,max(0,x):x+w]
    if patch.size == 0: return 0.0,0.0
    hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    hue_match = np.zeros(hsv.shape[:2],bool)
    for low,high in COLORS[color]:
        hue_match |= (hsv[:,:,0] >= low)&(hsv[:,:,0] <= high)
    saturation = np.clip((hsv[:,:,1].astype(float)-40)/140,0,1)
    brightness = np.clip((hsv[:,:,2].astype(float)-35)/125,0,1)
    evidence = (hue_match*saturation*brightness).ravel()
    # 同时看整体支持和最可靠的30%盒面，减少暗侧面、白贴纸影响。
    count = max(1,round(evidence.size*.3))
    strongest = np.partition(evidence,evidence.size-count)[-count:]
    color_score = float(.4*np.mean(evidence)+.6*np.mean(strongest))
    mask = (hue_match&(hsv[:,:,1] >= 80)&(hsv[:,:,2] >= 60)).astype(np.uint8)*255
    contours = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours: return color_score,0.0
    points = np.vstack(contours)
    area = sum(cv2.contourArea(contour) for contour in contours)
    hull_area = cv2.contourArea(cv2.convexHull(points))
    fill = min(1.0,area/(w*h))
    solidity = min(1.0,area/max(1,hull_area))
    aspect = min(w,h)/max(w,h)
    shape_score = (.4*fill+.3*solidity+.3*aspect)*min(1.0,aspect/.5)
    return color_score,shape_score


class BlockTracker(StableTarget):
    """动作后保留目标身份；旧框只供找回，不作为当前动作依据。"""
    def __init__(self, recover_head=False):
        self.recover_head = recover_head
        super().__init__()

    def reset(self):
        super().reset()
        self.reason = "no candidate"
        self.rejections = {}
        self.lost_since = None
        self.last_seen = None
        self.identity_box = None
        self.appearance = None
        self.motion = None
        self.scores = []

    def update(self, candidates, now=None, frame=None, color=None):
        now = time.monotonic() if now is None else now
        previous = self.box if self.box is not None else self.identity_box
        candidates = remove_nested_boxes(candidates)
        ranked = []
        self.scores = []
        self.rejections = {}
        self.reason = "no candidate"
        for candidate in candidates:
            quality = 1.0
            if frame is not None and color is not None:
                color_score,shape_score = block_quality(frame,candidate,color)
                if self.recover_head and candidate.width/candidate.height > 1.5 and shape_score < .70:
                    self.rejections[candidate] = 'wide floor-like region'
                    self.scores.append((candidate,0.0,color_score,shape_score))
                    continue
                quality = .65*color_score+.35*shape_score
                if previous is None:
                    threshold = .35 if self.recover_head else .45
                    reliable = color_score >= .12 and quality >= threshold
                    if self.recover_head: reliable = reliable and shape_score >= .45
                else:
                    # 已锁定先保留有颜色和形状依据的候选，再用身份关联判断。
                    reliable = color_score >= .06 and shape_score >= .25
                if not reliable:
                    self.rejections[candidate] = "color/shape weak"
                    self.scores.append((candidate,quality,color_score,shape_score))
                    continue
            else:
                color_score,shape_score = 1.0,1.0
            if previous is None:
                score = quality
                if self.recover_head and frame is not None:
                    # 小色块可能来自背景；首次选择优先完整、较大的盒面。
                    area_fraction = candidate.width*candidate.height/(frame.shape[0]*frame.shape[1])
                    score *= min(1.0,area_fraction/.02)
                    if score < .35:
                        self.rejections[candidate] = "small/weak candidate"
                        self.scores.append((candidate,score,color_score,shape_score))
                        continue
            else:
                wr = candidate.width/previous.width
                hr = candidate.height/previous.height
                if self.motion is None:
                    if overlap(previous,candidate) < .25 or not (.5 <= wr <= 2 and .5 <= hr <= 2):
                        self.rejections[candidate] = "position/size mismatch"
                        self.scores.append((candidate,0.0,color_score,shape_score));continue
                else:
                    if not (.4 <= wr <= 2.5 and .4 <= hr <= 2.5 and .5 <= wr/hr <= 2):
                        self.rejections[candidate] = "size mismatch"
                        self.scores.append((candidate,0.0,color_score,shape_score));continue
                    x_limit = max(40,previous.width*.8) if self.motion == 'head' else previous.width*2
                    if abs(candidate.cx-previous.cx) > x_limit:
                        self.rejections[candidate] = "position mismatch"
                        self.scores.append((candidate,0.0,color_score,shape_score));continue
                distance = 0.0
                if frame is not None and self.appearance is not None:
                    appearance = block_appearance(frame,candidate)
                    if appearance is None:
                        self.rejections[candidate] = "appearance missing"
                        continue
                    distance = cv2.compareHist(self.appearance,appearance,cv2.HISTCMP_BHATTACHARYYA)
                    if distance > .5:
                        self.rejections[candidate] = "appearance mismatch"
                        self.scores.append((candidate,0.0,color_score,shape_score));continue
                cost = abs(math.log(wr))+abs(math.log(hr))+distance*2
                cost += abs(candidate.cx-previous.cx)/previous.width*.3
                if self.motion != 'head':
                    cost += abs(candidate.cy-previous.cy)/previous.height*.3
                association = math.exp(-cost)
                # 已锁定时关联占主要权重，避免另一个更亮的盒子抢锁。
                score = .65*association+.35*quality
                if association < .25 or score < .35:
                    self.rejections[candidate] = "association weak"
                    self.scores.append((candidate,score,color_score,shape_score));continue
            self.scores.append((candidate,score,color_score,shape_score))
            ranked.append((score,candidate))
        ranked.sort(key=lambda item:item[0],reverse=True)
        lead_required = .12 if previous is None else .18
        if not ranked or (len(ranked) > 1 and ranked[0][0]-ranked[1][0] < lead_required):
            self.frames = 0
            self.reason = "ambiguous candidates" if ranked else ", ".join(sorted(set(self.rejections.values()))) or "no candidate"
            if previous is not None:
                if self.lost_since is None: self.lost_since = now
                if self.recover_head and now-self.lost_since >= 2.0:
                    # 只在头部找物阶段解锁；腹部接手后不换物块。
                    self.box = self.identity_box = self.appearance = None
                    self.motion = None
                    self.last_seen = self.lost_since = None
                    self.reason = "old lock released; reconfirm target"
                    return None
            if self.last_seen is not None and now-self.last_seen >= .5:
                self.box = None
                if self.motion is None: self.motion = 'body'
            return None
        self.lost_since = None
        self.reason = "confirming target"
        box = ranked[0][1]
        stable = self.box is not None and overlap(self.box,box) >= .45
        self.frames = self.frames+1 if stable else 1
        self.box = self.identity_box = box
        self.last_seen = now
        if self.appearance is None and frame is not None:
            self.appearance = block_appearance(frame,box)
        if self.frames >= TRACK_FRAMES: self.motion = None
        return box

    def after_body_move(self):
        self.frames = 0
        self.motion = 'body'
        self.scores = []

    def after_head_move(self):
        self.frames = 0
        self.motion = 'head'
        self.scores = []


def clipped_edges(box):
    """归一化框触及哪些画面边缘：左、上、右、下。"""
    return [box.x <= .01,box.y <= .01,
            box.x+box.width >= .99,box.bottom >= .99]


def near_head_target(box):
    """归一化框须居中、靠近下沿且足够大；下沿位置本身不能代表距离。"""
    return (abs(box.cx-.5) <= .08 and box.bottom >= NEAR_BOTTOM
            and box.width >= .30 and box.width*box.height >= .10)


def region_pickup_action(box, reference, flip):
    """H只提供横向位置和近处区域；不比较不同尺寸盒子的宽高。"""
    target = Box(*reference['pickup'])
    left,top,right,bottom = reference['pickup_clipped']
    if left and not right:
        dx = box.x+box.width-target.x-target.width
    elif right and not left:
        dx = box.x-target.x
    else:
        dx = box.cx-target.cx
    if abs(dx) > .10:
        return steering('SIDE_LEFT' if dx < 0 else 'SIDE_RIGHT',flip)
    target_bottom = .90 if bottom else target.bottom
    if clipped_edges(box)[3]:
        return 'UP_LITTLE' if box.y > PICKUP_TOP else 'HOLD_BOX'
    if not clipped_edges(box)[3] and box.bottom > target_bottom+.12: return 'STOP'
    if box.bottom < target_bottom-.08: return 'UP_LITTLE'
    return 'HOLD_BOX'



def pickup_action(box, reference, flip):
    """H提供大致范围；底边出画时根据可见盒面判断，不把画面边缘当真实底边。"""
    if reference.get('pickup_mode') == 'visible_region':
        return region_pickup_action(box,reference,flip)
    target = Box(*reference['pickup'])
    dx = box.cx-target.cx
    if abs(dx) > .10:
        return steering('SIDE_LEFT' if dx < 0 else 'SIDE_RIGHT',flip)
    if clipped_edges(box)[3]:
        # 不再将底边出画等同于可抱；每小步重新观察可见上沿。
        return 'UP_LITTLE' if box.y > PICKUP_TOP else 'HOLD_BOX'
    if box.bottom > target.bottom+.12: return 'STOP'
    if box.bottom < target.bottom-.08: return 'UP_LITTLE'
    return 'HOLD_BOX'


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
        self.delivery_aligned = False
        self.clipped_steps = 0
        self.pending_clipped = False
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
        if box is None or stable_frames < 1:
            self.reset_confirmation()
            self.reason = 'target missing, multiple or unstable; hold body'
            return 'WAIT'
        if self.phase == 'PICKUP':
            action = pickup_action(box,self.reference,self.flip)
            self.pending_clipped = clipped_edges(box)[3]
            self.reason = (f'partial box top={box.y:.0%}, pickup line=30%; '+action
                           if self.pending_clipped else 'belly pickup: relaxed H reference; '+action)
            if self.pending_clipped and action == 'UP_LITTLE' and self.clipped_steps >= 6:
                self.reason = '6 belly approach steps without reaching pickup line; stop'
                return 'STOP'
            if action in ('WAIT','STOP'):
                self.reset_confirmation()
                return action
        else:
            # 旧129头位的D参考不能用于新的120头位。
            drop = self.reference.get('drop')
            if drop is not None and self.reference.get('drop_head_position') == DELIVERY_HEAD_POSITION:
                target = Box(*drop)
                target_x, target_bottom = target.cx, target.bottom
            else:
                target_x, target_bottom = .5, DROP_BOTTOM
            dx = box.cx-target_x
            tolerance = .16 if self.delivery_aligned else .12
            if abs(dx) > tolerance:
                self.delivery_aligned = False
                action = 'LEFT_HOLDBOX' if dx < 0 else 'RIGHT_HOLDBOX'
                action = steering(action,self.flip)
                self.reason = 'turn toward target QR'
            elif box.bottom < target_bottom-.03:
                self.delivery_aligned = True
                action = 'UP_HOLDBOX'
                self.reason = 'approach target bottom reference'
            else:
                self.delivery_aligned = True
                action = 'DOWN_BOX'
                self.reason = 'QR near bottom; release'
        self.confirm_frames = self.confirm_frames+1 if self.pending == action else 1
        self.pending = action
        required = FINAL_FRAMES if action == 'DOWN_BOX' else PICKUP_FRAMES if action == 'HOLD_BOX' else TRACK_FRAMES
        return action if stable_frames >= TRACK_FRAMES and self.confirm_frames >= required else 'WAIT'

    def approach(self, box, stable_frames):
        self.pending_clipped = False
        if box is None or stable_frames < 1:
            self.reset_confirmation()
            self.reason = 'head target unstable; hold body'
            return 'WAIT'
        dx = box.cx-.5
        action = 'SIDE_LEFT' if dx < -.08 else 'SIDE_RIGHT' if dx > .08 else 'UP_LITTLE'
        action = steering(action,self.flip)
        self.confirm_frames = self.confirm_frames+1 if self.pending == action else 1
        self.pending = action
        self.reason = 'head block: align then approach'
        return action if stable_frames >= TRACK_FRAMES and self.confirm_frames >= TRACK_FRAMES else 'WAIT'

    def mark_sent(self, action):
        if action == 'UP_LITTLE' and self.pending_clipped: self.clipped_steps += 1
        # 在发送前改变阶段：失败即退出，不重复抱取/放下。
        if action == 'HOLD_BOX': self.phase = 'DELIVER'
        if action == 'DOWN_BOX': self.phase = 'DONE'
        self.actions += 1
        self.reset_confirmation()


def run(color, target_qr, actions=False, robot=None, search_right_actions=5, deadline=None,
        qr_reader=None):
    settings = camera_settings()
    reference = json.loads(REFERENCE_FILE.read_text()) if REFERENCE_FILE.exists() else None
    if actions:
        if (reference is None or reference.get('version') not in (2,3)
                or reference.get('cameras') != settings
                or reference.get('target_qr') != target_qr
                or 'pickup' not in reference):
            raise ValueError('先用双摄预览按 H 标定腹部抱取位置；D可选，用120头位标定放下位置')
    planner = CarryPlanner(reference,settings['belly']['flip']) if actions else None
    handover = Handover(dict(forward=INITIAL_HEAD_POSITION,down_sign=1,bounds=[85,180]))
    approach = WalkKick()
    head_tracker,belly_tracker,qr_tracker = BlockTracker(recover_head=True),BlockTracker(),DestinationTracker()
    detector = cv2.QRCodeDetector()
    seen_contents = set()
    search_turns = 0
    qr_missing_since = None
    transferring = False
    transfer_steps = 0
    last_qr_x = None
    with ExitStack() as stack:
        head_eye = RobotEye(**settings['head'],latest=True);stack.callback(head_eye.close)
        belly_eye = RobotEye(**settings['belly'],latest=True);stack.callback(belly_eye.close)
        okh,hf = head_eye.getImage();okb,bf = belly_eye.getImage()
        if not okh or not okb: raise RuntimeError('Camera read failed before startup')
        shapes = dict(head=list(hf.shape[:2]),belly=list(bf.shape[:2]))
        if actions and reference.get('shapes') != shapes:
            raise ValueError('画面尺寸变化，请重新标定双摄搬运位置')
        if actions:
            move = robot
            if move is None:
                from robotmove import RobotMove
                move = RobotMove(None,port=BODY_SERIAL_PORT);stack.callback(move.close)
        from Head import RobotHeadServoOnly
        servo = RobotHeadServoOnly(hold=True);stack.callback(servo.cleanup)
        servo.turn_vertical(INITIAL_HEAD_POSITION)
        head_eye.discard_frames(1);belly_eye.discard_frames(1)
        stack.callback(cv2.destroyAllWindows)
        ready_at = time.monotonic()+.5
        print('目标颜色:',color,'目的地二维码:',target_qr)
        print('候选显示score总分、C颜色分、S形状分；这是匹配评分，不是识别概率。')
        print('H=123头位/保存腹部抱取位置；D=120头位/保存二维码放下位置；Q=退出。')
        print('腹部接手后锁定阶段；漏检暂停身体，保留关联0.5秒。H可保存实际可抱位置的局部色块。')
        print('接近、腹部接手和抱起确认2帧；放下和保存参考仍确认5帧。')
        print('H仅参考横向位置和近处下沿；不比较盒子尺寸，不追加面积或形状分抱取门槛。')
        print('腹部底边出画后，上沿到画面上方30%才抱取；每次小步后重新观察，最多6次。30%需现场验证。')
        print('交接须目标居中、下沿到88%、宽度至少30%、框面积至少10%。')
        print('看得见目标时正常靠近；近处丢失后固定头位，最多惯性前进3次UP_LITTLE，腹部未接手则停止。')
        print('H或D切换头位后，请等画面稳定再按一次保存。未标定D时按二维码下沿85%放下。')
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                print('比赛总期限到达，停止搬运');return False
            okh,hf = head_eye.getImage();okb,bf = belly_eye.getImage()
            if not okh or not okb: raise RuntimeError('Camera read failed; no further actions')
            if dict(head=list(hf.shape[:2]),belly=list(bf.shape[:2])) != shapes:
                raise ValueError('Camera resolution changed; stop actions')
            head_busy = servo.is_moving()
            head_blocks,belly_blocks = find_blocks(hf,color),find_blocks(bf,color,near=True)
            if head_busy:
                head_tracker.after_head_move()
                head_block = None
            else:
                head_block = head_tracker.update(head_blocks,frame=hf,color=color)
            belly_block = belly_tracker.update(belly_blocks,frame=bf,color=color)
            bottom_clipped = belly_block is not None and belly_block.bottom >= bf.shape[0]-3
            local_pickup = reference is not None and reference.get('pickup_mode') == 'visible_region'
            if planner is not None and planner.phase == 'PICKUP':
                if belly_tracker.frames >= TRACK_FRAMES:
                    handover.phase = 'BELLY'
                    transferring = False
                # 腹部接手后锁定阶段，短暂丢失不退回头部前进。
            codes = []
            if not actions or planner.phase != 'PICKUP':
                codes = qr_reader(hf) if qr_reader is not None else read_qr_codes(hf,detector)
            for content,_ in codes:
                if content not in seen_contents:
                    print('读到二维码:',content);seen_contents.add(content)
            matches = [box for content,box in codes if content == target_qr]
            qr = qr_tracker.update(matches,frame=hf)
            if head_busy:
                head_tracker.after_head_move();qr_tracker.reset()
                head_block = qr = None
            if qr is not None:
                qr_missing_since = None
                search_turns = 0
                last_qr_x = qr.cx/hf.shape[1]
            for name,frame,blocks in [('head',hf,head_blocks),('belly',bf,belly_blocks)]:
                display = frame.copy()
                for box in blocks:
                    cv2.rectangle(display,(int(box.x),int(box.y)),(int(box.x+box.width),int(box.bottom)),(0,255,0),2)
                tracker = head_tracker if name == 'head' else belly_tracker
                for box,score,color_score,shape_score in tracker.scores:
                    label = f'score={score:.2f} C={color_score:.2f} S={shape_score:.2f}'
                    if box in tracker.rejections: label += ' '+tracker.rejections[box]
                    cv2.putText(display,label,(int(box.x),max(90,int(box.y)-5)),0,.4,(0,255,255),1)
                if name == 'head':
                    for content,box in codes:
                        cv2.rectangle(display,(int(box.x),int(box.y)),(int(box.x+box.width),int(box.bottom)),(255,255,0),2)
                        cv2.putText(display,content,(int(box.x),max(15,int(box.y)-5)),0,.5,(255,255,0),1)
                phase = planner.phase if actions else 'PREVIEW'
                reason = planner.reason if actions else 'H=pickup at 123; D=QR drop at 120; Q=quit'
                cv2.putText(display,f'{phase} {handover.phase} color={color} head={handover.angle}',(8,25),0,.5,(255,255,255),1)
                if name == 'belly' and bottom_clipped and not local_pickup and not actions:
                    reason = 'partial block: H can save visible pickup region'
                cv2.putText(display,reason,(8,48),0,.45,(255,255,255),1)
                if name == 'belly':
                    status = f'belly: blocks={len(belly_blocks)} stable={belly_tracker.frames}/{TRACK_FRAMES}'
                else:
                    status = f'head: blocks={len(head_blocks)} lock={int(head_tracker.identity_box is not None)} frames={head_tracker.frames}/{TRACK_FRAMES} QR={len(matches)} {qr_tracker.source}'
                    if actions and planner.phase == 'DELIVER':
                        status = f'head: QR decoded={len(matches)} {qr_tracker.source} stable={qr_tracker.frames}/{FINAL_FRAMES}'
                if name == 'head':
                    detail = 'target QR: '+qr_tracker.source if actions and planner.phase == 'DELIVER' else head_tracker.reason
                    cv2.putText(display,detail,(8,92),0,.4,(0,255,255),1)
                selected = belly_block if name == 'belly' else head_block
                if selected is not None:
                    cv2.rectangle(display,(int(selected.x),int(selected.y)),
                                  (int(selected.x+selected.width),int(selected.bottom)),(0,255,255),3)
                cv2.putText(display,status,(8,70),0,.5,(255,255,255),1)
                if name == 'belly' and reference is not None and 'pickup' in reference:
                    target = Box(*reference['pickup'])
                    edge = PICKUP_TOP if bottom_clipped else .90 if reference.get('pickup_mode') == 'visible_region' and reference['pickup_clipped'][3] else target.bottom
                    y = round(edge*display.shape[0])
                    cv2.line(display,(0,y),(display.shape[1]-1,y),(0,255,255),1)
                if name == 'head' and handover.angle == DELIVERY_HEAD_POSITION:
                    drop_bottom = DROP_BOTTOM
                    if reference is not None and reference.get('drop_head_position') == DELIVERY_HEAD_POSITION:
                        drop_bottom = Box(*reference['drop']).bottom
                    y = round(drop_bottom*display.shape[0])
                    cv2.line(display,(0,y),(display.shape[1]-1,y),(0,255,255),1)
                cv2.imshow('carry '+name,display)
            key = cv2.waitKey(1)&255
            if key in (ord('q'),27): return False
            if not actions and key in (ord('h'),ord('d')):
                angle = INITIAL_HEAD_POSITION if key == ord('h') else DELIVERY_HEAD_POSITION
                if handover.angle != angle:
                    handover.angle = angle;servo.begin_vertical(angle)
                    head_tracker.after_head_move();qr_tracker.reset()
                    print('已切换头位到',angle,'；等待稳定后再按同一个键保存。')
                    continue
                if head_busy:
                    print('头部仍在移动，请等画面稳定。');continue
                box = belly_block if key == ord('h') else qr
                tracker = belly_tracker if key == ord('h') else qr_tracker
                if box is None or tracker.frames < 5:
                    count = len(belly_blocks) if key == ord('h') else len(matches)
                    if key == ord('h') and bottom_clipped:
                        print('物块底边出画，不能保存抱取参考。')
                    print('腹部物块' if key == ord('h') else '头部目标二维码',
                          '候选数=',count,'稳定帧=',tracker.frames,'/5；当前不能保存。');continue
                if (reference is None or reference.get('version') not in (2,3)
                        or reference.get('cameras') != settings or reference.get('shapes') != shapes
                        or reference.get('target_qr') != target_qr):
                    reference = dict(version=3,cameras=settings,head_position=INITIAL_HEAD_POSITION,
                                     target_qr=target_qr,shapes=shapes)
                frame = bf if key == ord('h') else hf
                normalized = box.normalized(frame.shape)
                reference['version'] = 3
                if key == ord('d'): reference['drop_head_position'] = DELIVERY_HEAD_POSITION
                if key == ord('h'):
                    reference['pickup_mode'] = 'visible_region'
                    reference['pickup_clipped'] = clipped_edges(normalized)
                reference['pickup' if key == ord('h') else 'drop'] = [normalized.x,normalized.y,normalized.width,normalized.height]
                REFERENCE_FILE.write_text(json.dumps(reference,ensure_ascii=False,indent=2))
                print('已保存:',REFERENCE_FILE)
            if not actions or head_busy or time.monotonic() < ready_at: continue
            if planner.actions >= 40 or time.monotonic()-planner.started_at >= 180:
                print('40 actions / 180 seconds limit');return False
            if planner.phase == 'DELIVER' and qr is None and search_right_actions:
                # 短暂漏读先等，随后向最后看到二维码的一侧有限找回；不按旧框前进或放下。
                if qr_missing_since is None: qr_missing_since = time.monotonic()
                planner.reset_confirmation()
                planner.reason = 'QR missing: hold and reacquire'
                if len(matches) > 1 or time.monotonic()-qr_missing_since < 2.0: continue
                if search_turns >= search_right_actions:
                    print('连续有限搜索后仍未找回目的地二维码，停止');return False
                search_action = 'LEFT_HOLDBOX' if last_qr_x is not None and last_qr_x < .5 else 'RIGHT_HOLDBOX'
                search_action = steering(search_action,settings['head']['flip'])
                move.robotMove(search_action)
                search_turns += 1;planner.mark_sent(search_action)
                head_eye.discard_frames(1);belly_eye.discard_frames(1)
                qr_tracker.after_move();qr_missing_since = None
                ready_at = time.monotonic()+.5
                continue
            if planner.phase == 'PICKUP':
                if handover.phase == 'HEAD' and head_block is not None and head_tracker.frames >= TRACK_FRAMES:
                    box = head_block.normalized(hf.shape)
                    transferring = near_head_target(box)
            if planner.phase == 'PICKUP' and handover.phase == 'HEAD':
                planner.flip = settings['head']['flip']
                if belly_blocks:
                    # 腹部第一帧就开始累计位置确认，第二帧接手时不再额外等待。
                    planner.flip = settings['belly']['flip']
                    action = planner.decide(belly_block.normalized(bf.shape) if belly_block else None,
                                            belly_tracker.frames)
                elif head_block is not None:
                    # 当前目标仍可见，按视觉确认靠近，不消耗丢失后的惯性步数。
                    action = planner.approach(head_block.normalized(hf.shape),head_tracker.frames)
                elif transferring:
                    # 只在已确认近处且居中的目标后启用；短暂丢失也保持头位。
                    if transfer_steps >= TRANSFER_STEPS:
                        print('近处目标丢失后已惯性前进3次，腹部仍未接手，停止避免撞箱子');return False
                    action = 'UP_LITTLE'
                    planner.reason = f'camera transfer: forward {transfer_steps+1}/{TRANSFER_STEPS}; head fixed'
                    transfer_steps += 1
                elif head_block is None and not belly_blocks:
                    action = approach.search_action(time.monotonic())
                    planner.reason = approach.reason.replace('ball','block')
                else:
                    action = planner.approach(head_block.normalized(hf.shape) if head_block else None,head_tracker.frames)
                # 未通过关联的候选不能阻止头部搜索；身体仍等待重新确认。
                if head_block is not None or belly_blocks:
                    approach.reset_search()
                    if belly_blocks or action in ('REACQUIRE','LOWER_HEAD','RAISE_HEAD'): action = 'WAIT'
                if action in ('REACQUIRE','LOWER_HEAD','RAISE_HEAD'):
                    if action != 'REACQUIRE':
                        direction = 1 if action == 'LOWER_HEAD' else -1
                        angle = max(INITIAL_HEAD_POSITION-12,min(handover.down,handover.angle+3*direction))
                        moved = angle != handover.angle
                        if not moved: approach.search_head_direction *= -1
                        if moved:
                            handover.angle = angle;servo.begin_vertical(angle,settle_seconds=.18)
                            head_tracker.after_head_move()
                        approach.mark_head_search(moved)
                    planner.reason = approach.reason.replace('ball','block')
                    continue
            else:
                target = belly_block if planner.phase == 'PICKUP' else qr
                tracker = belly_tracker if planner.phase == 'PICKUP' else qr_tracker
                planner.flip = settings['belly' if planner.phase == 'PICKUP' else 'head']['flip']
                action = planner.decide(target.normalized(bf.shape if planner.phase == 'PICKUP' else hf.shape) if target else None,
                                        tracker.frames)
            if action in ('STOP','DONE'):
                print(planner.reason);return action == 'DONE'
            if action == 'WAIT': continue
            planner.mark_sent(action)
            approach.reset_observation()
            print('搬运动作:',action);move.robotMove(action)
            if action == 'DOWN_BOX':
                print('放下动作已执行；物块是否实际到位需要现场确认。');return True
            if action == 'HOLD_BOX':
                # 抱取完成后抬头到120，重新识别目的地二维码。
                handover.phase = 'HEAD'
                handover.angle = DELIVERY_HEAD_POSITION;servo.begin_vertical(DELIVERY_HEAD_POSITION)
            head_eye.discard_frames(1);belly_eye.discard_frames(1)
            if action == 'HOLD_BOX': head_tracker.after_head_move()
            else: head_tracker.after_body_move()
            belly_tracker.after_body_move();qr_tracker.after_move()
            handover.reset_follow()
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
