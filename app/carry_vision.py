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
from ball_debug import distinct_from_background

INITIAL_HEAD_POSITION = 123
DELIVERY_HEAD_POSITION = 120
TRACK_FRAMES = 2
PICKUP_FRAMES = 2
FINAL_FRAMES = 5  # 保存标定仍确认5帧。
NEAR_BOTTOM = .88
TRANSFER_STEPS = 3  # 只限制近处目标丢失后的无目标小步，不限制看得见目标时的正常靠近。
PICKUP_TOP = .20  # 从底部算80%，即从顶部算20%；上沿到线或在线上方才抱。
QR_WAIT_SECONDS = 5.0
DELIVERY_OBSERVE_SECONDS = 1.2  # 二维码搬运每个身体动作结束后，等待画面稳定。
RUN_SECONDS = 300
HEAD_STEP = 6  # 舵机控制单位，不是物理角度。
HEAD_SEARCH_STEP = 3
HEAD_SEARCH_MIN = 120
HEAD_SEARCH_MAX = 140
HEAD_SEARCH_WAIT = 1.5
DROP_FRAMES = 3
DROP_SIZE_MIN = .90  # 实测投放参考的宽、高允许小10%，停止在参考附近。
DROP_SIZE_MAX = 1.20  # 超过参考20%时停止，避免继续靠近立牌。
DROP_POSITION_TOLERANCE = .06
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
        partial_head = y+h >= height-3 and w >= width*.15 and w*h >= height*width*.04
        if not near and ((touches_side and not partial_head) or w*h > height*width*.85):
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


def block_appearance(frame, box, color=None):
    """腹部只比较所选颜色的色相；不让饱和度和框内地板比例主导身份。"""
    x,y,w,h = map(int,(box.x,box.y,box.width,box.height))
    patch = frame[max(0,y):y+h,max(0,x):x+w]
    if patch.size == 0: return None
    hsv = cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(0,80,60),(179,255,255))
    if cv2.countNonZero(mask) < 30: return None
    if color is not None:
        selected = np.zeros(hsv.shape[:2],np.uint8)
        for low,high in COLORS[color]:
            selected |= cv2.inRange(hsv,(low,80,60),(high,255,255))
        mask &= selected
        if cv2.countNonZero(mask) < 30: return None
        histogram = cv2.calcHist([hsv],[0],mask,[18],[0,180])
        # 相邻色相区间平滑，减少蓝色在直方图格子边界跳变的影响。
        histogram = .5*histogram+.25*np.roll(histogram,1,axis=0)+.25*np.roll(histogram,-1,axis=0)
    else:
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
    def __init__(self, recover_head=False, near=False):
        self.recover_head = recover_head
        self.near = near
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
        self.recovery = StableTarget()

    def update(self, candidates, now=None, frame=None, color=None):
        now = time.monotonic() if now is None else now
        previous = self.box if self.box is not None else self.identity_box
        candidates = remove_nested_boxes(candidates)
        ranked = []
        changed_appearance = set()
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
                partial = (self.near and frame is not None
                           and (candidate.bottom >= frame.shape[0]-3 or previous.bottom >= frame.shape[0]-3))
                if partial:
                    # 近处盒面放大或出画时，宽高比不能继续作为身份门槛。
                    horizontal = max(0,min(previous.x+previous.width,candidate.x+candidate.width)
                                     -max(previous.x,candidate.x))
                    if horizontal < .5*min(previous.width,candidate.width):
                        self.rejections[candidate] = "position mismatch"
                        self.scores.append((candidate,0.0,color_score,shape_score));continue
                elif self.motion is None:
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
                    appearance = block_appearance(frame,candidate,color if self.near else None)
                    if appearance is None:
                        self.rejections[candidate] = "appearance missing"
                        continue
                    distance = cv2.compareHist(self.appearance,appearance,cv2.HISTCMP_BHATTACHARYYA)
                    if distance > .5:
                        horizontal = max(0,min(previous.x+previous.width,candidate.x+candidate.width)
                                         -max(previous.x,candidate.x))
                        vertical = max(0,min(previous.bottom,candidate.bottom)-max(previous.y,candidate.y))
                        coverage = horizontal*vertical/min(previous.width*previous.height,
                                                           candidate.width*candidate.height)
                        same_region = (coverage >= .5 and
                                       abs(candidate.cx-previous.cx) <= .35*max(previous.width,candidate.width))
                        if not (self.near and color is not None and len(candidates) == 1
                                and color_score >= .45 and shape_score >= .45 and same_region):
                            self.rejections[candidate] = "appearance mismatch"
                            self.scores.append((candidate,0.0,color_score,shape_score));continue
                        # 新画面强颜色证据和位置连续时重新确认，仍需当前框稳定2帧。
                        changed_appearance.add(candidate)
                        distance = .35
                cost = distance*2 if partial else abs(math.log(wr))+abs(math.log(hr))+distance*2
                cost += abs(candidate.cx-previous.cx)/max(previous.width,candidate.width)*.3
                if self.motion != 'head' and not partial:
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
            # 腹部旧位置失配时，用当前单个强候选连续3帧重新建立位置参考。
            # 不退回头部、不按旧框走，也不接受多个候选或弱颜色区域。
            recover_position = (self.near and frame is not None and color is not None
                                and len(candidates) == 1 and self.reason in
                                ('position mismatch','position/size mismatch','size mismatch','association weak'))
            if recover_position:
                candidate = candidates[0]
                color_score,shape_score = block_quality(frame,candidate,color)
                strong = (color_score >= .45 and shape_score >= .45
                          and candidate.width*candidate.height >= frame.shape[0]*frame.shape[1]*.03)
                if strong:
                    self.recovery.update([candidate])
                    count = self.recovery.frames
                    self.reason = f'belly position reconfirm {count}/3; hold body'
                    if count >= 3:
                        self.box = self.identity_box = candidate
                        self.frames = count
                        self.appearance = block_appearance(frame,candidate,color)
                        self.last_seen = now
                        self.lost_since = self.motion = None
                        self.rejections = {}
                        self.scores = [(candidate,.65*color_score+.35*shape_score,color_score,shape_score)]
                        self.recovery.reset()
                        self.reason = 'belly position recovered; current target confirmed'
                        return candidate
                else:
                    self.recovery.reset()
            else:
                self.recovery.reset()
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
        self.recovery.reset()
        self.lost_since = None
        self.reason = "confirming target"
        box = ranked[0][1]
        stable = box not in changed_appearance and self.box is not None and overlap(self.box,box) >= .45
        self.frames = self.frames+1 if stable else 1
        self.box = self.identity_box = box
        self.last_seen = now
        if frame is not None and (self.appearance is None or box in changed_appearance or self.near and self.frames >= TRACK_FRAMES):
            self.appearance = block_appearance(frame,box,color if self.near else None)
        if box in changed_appearance:
            self.reason = 'box appearance changed; reconfirm current region'
        if self.frames >= TRACK_FRAMES: self.motion = None
        return box

    def after_body_move(self):
        self.frames = 0
        self.motion = 'body'
        self.scores = []
        self.recovery.reset()

    def after_head_move(self):
        self.frames = 0
        self.motion = 'head'
        self.scores = []
        self.recovery.reset()


def clipped_edges(box):
    """归一化框触及哪些画面边缘：左、上、右、下。"""
    return [box.x <= .01,box.y <= .01,
            box.x+box.width >= .99,box.bottom >= .99]


def near_head_target(box):
    """归一化框须居中、靠近下沿且足够大；下沿位置本身不能代表距离。"""
    return (abs(box.cx-.5) <= .08 and box.bottom >= NEAR_BOTTOM
            and box.width >= .30 and box.width*box.height >= .10)


def follow_head_angle(box, angle):
    """稳定目标临近上下边缘时转头；每次转头后须重新确认目标。"""
    if box.cy > .65: return min(HEAD_SEARCH_MAX,angle+HEAD_STEP)
    if box.cy < .25: return max(HEAD_SEARCH_MIN,angle-HEAD_STEP)
    return angle


def region_pickup_action(box, reference, flip):
    """H只提供横向对齐参考；距离统一按方块上沿判断。"""
    target = Box(*reference['pickup'])
    left,_,right,_ = reference['pickup_clipped']
    if left and not right:
        dx = box.x+box.width-target.x-target.width
    elif right and not left:
        dx = box.x-target.x
    else:
        dx = box.cx-target.cx
    if abs(dx) > .10:
        return steering('SIDE_LEFT' if dx < 0 else 'SIDE_RIGHT',flip)
    return 'UP_LITTLE' if box.y > PICKUP_TOP else 'HOLD_BOX'



def pickup_action(box, reference, flip):
    """横向对齐后，上沿到达顶部20%横线才抱；旧H下沿不终止接近。"""
    if reference.get('pickup_mode') == 'visible_region':
        return region_pickup_action(box,reference,flip)
    target = Box(*reference['pickup'])
    dx = box.cx-target.cx
    if abs(dx) > .10:
        return steering('SIDE_LEFT' if dx < 0 else 'SIDE_RIGHT',flip)
    return 'UP_LITTLE' if box.y > PICKUP_TOP else 'HOLD_BOX'


def steering(action, flip):
    if flip in ('1','-1'):
        action = {'SIDE_LEFT':'SIDE_RIGHT','SIDE_RIGHT':'SIDE_LEFT',
                  'LEFT_HOLDBOX':'RIGHT_HOLDBOX','RIGHT_HOLDBOX':'LEFT_HOLDBOX'}.get(action,action)
    # 沿用已经实机纠正的横移方向；抱物转向单独保留原映射。
    if action == 'SIDE_LEFT': return 'SIDE_RIGHT'
    if action == 'SIDE_RIGHT': return 'SIDE_LEFT'
    return action


def drop_reference(reference):
    """只接受120头位、完整画面内的D参考，不用默认横线代替投放位置。"""
    if not reference or reference.get('drop_head_position') != DELIVERY_HEAD_POSITION:
        return None
    values = reference.get('drop')
    if not isinstance(values,(list,tuple)) or len(values) != 4:
        return None
    if not all(isinstance(value,(int,float)) and math.isfinite(value) for value in values):
        return None
    box = Box(*values)
    if box.width <= 0 or box.height <= 0 or box.x < 0 or box.y < 0 or box.x+box.width > 1 or box.bottom > 1:
        return None
    return box


class CarryPlanner:
    def __init__(self, reference, flip, use_original_drop=False):
        self.reference = reference
        self.flip = flip
        self.use_original_drop = use_original_drop
        self.phase = 'PICKUP'
        self.actions = 0
        self.started_at = time.monotonic()
        self.pending = None
        self.confirm_frames = 0
        self.delivery_aligned = False
        self.belly_approach_steps = 0
        self.pending_belly_approach = False
        self.reason = 'looking for selected block color'

    def reset_confirmation(self):
        self.pending = None
        self.confirm_frames = 0

    def decide(self, box, stable_frames, now=None, decoded=True):
        now = time.monotonic() if now is None else now
        if self.phase == 'DONE': return 'DONE'
        if self.actions >= 40 or now-self.started_at >= RUN_SECONDS:
            self.reason = f'40 actions / {RUN_SECONDS} seconds limit'
            return 'STOP'
        if box is None or stable_frames < 1:
            self.reset_confirmation()
            self.reason = 'target missing, multiple or unstable; hold body'
            return 'WAIT'
        if self.phase == 'PICKUP':
            action = pickup_action(box,self.reference,self.flip)
            self.pending_belly_approach = True
            self.reason = f'belly top={box.y:.1%}, pickup line=20% (80% from bottom); '+action
            if action == 'UP_LITTLE' and self.belly_approach_steps >= 6:
                self.reason = '6 belly approach steps without reaching pickup line; stop'
                return 'STOP'
        else:
            target = drop_reference(self.reference)
            if target is None and self.use_original_drop:
                # 旧版只有画面下沿阈值，不能据此推算真实投放距离。
                dx = box.cx-.5
                tolerance = .16 if self.delivery_aligned else .12
                if abs(dx) > tolerance:
                    self.delivery_aligned = False
                    action = steering('LEFT_HOLDBOX' if dx < 0 else 'RIGHT_HOLDBOX',self.flip)
                elif box.bottom < .60-.03:
                    self.delivery_aligned = True
                    action = 'UP_HOLDBOX'
                elif not decoded:
                    self.reset_confirmation()
                    self.reason = 'original drop line reached; need fresh exact QR decode'
                    return 'WAIT'
                else:
                    self.delivery_aligned = True
                    action = 'DOWN_BOX'
                self.reason = f'original drop rule: QR bottom={box.bottom:.0%}; '+action
            elif target is None:
                self.reason = 'STOP: valid D placement reference at head120 required'
                return 'STOP'
            else:
                return self.decide_placement(box,stable_frames,target,decoded)
        return self.confirm_action(action,stable_frames)

    def decide_placement(self, box, stable_frames, target, decoded):
        width_ratio = box.width/target.width
        height_ratio = box.height/target.height
        dx = box.cx-target.cx
        tolerance = .16 if self.delivery_aligned else .12
        near = min(width_ratio,height_ratio)+1e-6 >= DROP_SIZE_MIN
        if max(width_ratio,height_ratio) > DROP_SIZE_MAX:
            self.reason = 'STOP: QR larger than placement reference; too close or view changed'
            return 'STOP'
        if not .80 <= width_ratio/height_ratio <= 1.25:
            self.reset_confirmation()
            self.reason = 'QR shape differs from placement reference; hold body'
            return 'WAIT'
        if near:
            tolerance = DROP_POSITION_TOLERANCE
        if abs(dx) > tolerance:
            self.delivery_aligned = False
            action = 'LEFT_HOLDBOX' if dx < 0 else 'RIGHT_HOLDBOX'
            action = steering(action,self.flip)
            self.reason = 'turn toward target QR'
        elif not near:
            self.delivery_aligned = True
            action = 'UP_HOLDBOX'
            self.reason = f'approach placement size: W={width_ratio:.0%} H={height_ratio:.0%}'
        elif abs(box.cy-target.cy) > DROP_POSITION_TOLERANCE:
            self.reset_confirmation()
            self.reason = 'placement size reached but vertical position differs; hold body'
            return 'WAIT'
        elif not decoded:
            self.reset_confirmation()
            self.reason = 'placement near; need fresh exact QR decode before release'
            return 'WAIT'
        else:
            self.delivery_aligned = True
            action = 'DOWN_BOX'
            self.reason = f'placement reference reached: W={width_ratio:.0%} H={height_ratio:.0%}; release'
        return self.confirm_action(action,stable_frames)

    def confirm_action(self, action, stable_frames):
        self.confirm_frames = self.confirm_frames+1 if self.pending == action else 1
        self.pending = action
        required = DROP_FRAMES if action == 'DOWN_BOX' else PICKUP_FRAMES if action == 'HOLD_BOX' else TRACK_FRAMES
        return action if stable_frames >= TRACK_FRAMES and self.confirm_frames >= required else 'WAIT'

    def approach(self, box, stable_frames):
        self.pending_belly_approach = False
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
        if action == 'UP_LITTLE' and self.pending_belly_approach: self.belly_approach_steps += 1
        # 在发送前改变阶段：失败即退出，不重复抱取/放下。
        if action == 'HOLD_BOX': self.phase = 'DELIVER'
        if action == 'DOWN_BOX': self.phase = 'DONE'
        self.actions += 1
        self.reset_confirmation()


def run(color, target_qr, actions=False, robot=None, search_right_actions=5, deadline=None,
        qr_reader=None, use_original_drop=False, eyes=None, servo=None):
    settings = camera_settings()
    reference = json.loads(REFERENCE_FILE.read_text()) if REFERENCE_FILE.exists() else None
    if actions:
        if (reference is None or reference.get('version') not in (2,3)
                or reference.get('cameras') != settings
                or reference.get('target_qr') != target_qr
                or 'pickup' not in reference):
            raise ValueError('双摄配置或H参考不匹配，请先恢复原相机配置并核对H；实际投放还需120头位D参考')
        if drop_reference(reference) is None and not use_original_drop:
            raise ValueError('实际投放需要120头位的D参考：在物块实际应放下的位置保存D；保留已有H标定')
    planner = CarryPlanner(reference,settings['belly']['flip'],use_original_drop) if actions else None
    handover = Handover(dict(forward=INITIAL_HEAD_POSITION,down_sign=1,
                             bounds=[HEAD_SEARCH_MIN,HEAD_SEARCH_MAX]))
    handover.down = HEAD_SEARCH_MAX  # 只覆盖搬运范围，不改变球类任务的共享默认值。
    head_tracker,belly_tracker,qr_tracker = BlockTracker(recover_head=True),BlockTracker(near=True),DestinationTracker()
    detector = cv2.QRCodeDetector()
    seen_contents = set()
    search_turns = 0
    qr_missing_since = None
    transferring = False
    transfer_steps = 0
    head_search_direction = 1
    last_qr_x = None
    with ExitStack() as stack:
        if eyes is None:
            head_eye = RobotEye(**settings['head'],latest=True);stack.callback(head_eye.close)
            belly_eye = RobotEye(**settings['belly'],latest=True);stack.callback(belly_eye.close)
        else:
            head_eye, belly_eye = eyes
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
        if servo is None:
            servo = RobotHeadServoOnly(hold=True);stack.callback(servo.cleanup)
        servo.turn_vertical(INITIAL_HEAD_POSITION)
        head_eye.discard_frames(1);belly_eye.discard_frames(1)
        stack.callback(cv2.destroyAllWindows)
        ready_at = time.monotonic()+.5
        print('目标颜色:',color,'目的地二维码:',target_qr)
        print('候选显示score总分、C颜色分、S形状分；这是匹配评分，不是识别概率。')
        print('H=123头位/保存腹部抱取位置；D=120头位/保存二维码放下位置；Q=退出。')
        print('腹部确认后独占接近控制；旧位置失配时单个强目标稳定3帧重新确认。H可保存局部参考。')
        print('未确认腹部候选不阻断可靠头部靠近；抱取完成后仅头部二维码识别参与动作，腹部仅预览。')
        print('接近、腹部接手和抱起确认2帧；放下确认3帧，保存参考仍确认5帧。')
        print('H仅参考横向位置；不比较盒子尺寸，不追加面积或形状分抱取门槛。')
        print('腹部目标横向对齐后，上沿到达或高于顶部20%横线（y<=20%）才抱取；每次小步后重新观察，最多6次。上方20%线需现场验证。')
        print('可见目标跟随每次6单位；丢失搜索每次3单位、观察1.5秒，范围120至140。')
        print('二维码搬运每次身体动作结束后额外观察1.2秒，再决定下一步。')
        print('二维码丢失后每个位置原地尝试5秒，再有限转向搜索；搬运总上限300秒、40次身体动作。')
        print('交接须目标居中、下沿到88%、宽度至少30%、框面积至少10%。')
        print('看得见目标时正常靠近；近处丢失后先低头找回，到低头边界仍丢失才最多惯性前进3次UP_LITTLE，腹部未接手则停止。')
        print('实际投放用D位置与宽高参考：宽高达到参考90%至120%、中心接近，且目标二维码连续解码3帧才放下；超过120%停止。')
        if use_original_drop and drop_reference(reference) is None:
            print('调试沿用H：缺D时使用旧版下沿60%参考，提前3%触发、目标二维码解码确认3帧；实际投放位置尚未验证。')
        else:
            print('在实机确认适合放下的位置保存D，头位120；已有正确D可沿用。缺D不启用身体搬运。')
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                print('比赛总期限到达，停止搬运');return False
            okh,hf = head_eye.getImage();okb,bf = belly_eye.getImage()
            if not okh or not okb: raise RuntimeError('Camera read failed; no further actions')
            if dict(head=list(hf.shape[:2]),belly=list(bf.shape[:2])) != shapes:
                raise ValueError('Camera resolution changed; stop actions')
            head_busy = servo.is_moving()
            pickup_active = not actions or planner.phase == 'PICKUP'
            head_blocks = belly_blocks = []
            head_block = belly_block = None
            if pickup_active:
                # 腹部接手后关闭头部找方块；抱起后两路都不再检测方块。
                if not actions or handover.phase == 'HEAD':
                    head_blocks = find_blocks(hf,color)
                    if head_busy:
                        head_tracker.after_head_move()
                    else:
                        head_block = head_tracker.update(head_blocks,frame=hf,color=color)
                belly_blocks = find_blocks(bf,color,near=True)
                belly_block = belly_tracker.update(belly_blocks,frame=bf,color=color)
            bottom_clipped = any(box.bottom >= bf.shape[0]-3 for box in belly_blocks)
            local_pickup = reference is not None and reference.get('pickup_mode') == 'visible_region'
            if planner is not None and planner.phase == 'PICKUP':
                if belly_tracker.frames >= TRACK_FRAMES:
                    handover.phase = 'BELLY'
                    transferring = False
                    head_tracker.reset()
                    head_block = None
                    head_blocks = []
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
                    status = (f'belly: blocks={len(belly_blocks)} stable={belly_tracker.frames}/{TRACK_FRAMES}'
                              if pickup_active else 'belly: preview only; block detection off')
                else:
                    status = f'head: blocks={len(head_blocks)} lock={int(head_tracker.identity_box is not None)} frames={head_tracker.frames}/{TRACK_FRAMES} QR={len(matches)} {qr_tracker.source}'
                    if actions and planner.phase == 'PICKUP' and handover.phase == 'BELLY':
                        status = 'head: block detection off; belly controls pickup'
                    if actions and planner.phase == 'DELIVER':
                        status = f'head: QR decoded={len(matches)} {qr_tracker.source} stable={qr_tracker.frames}/{DROP_FRAMES}'
                if name == 'head':
                    detail = ('target QR: '+qr_tracker.source if actions and planner.phase == 'DELIVER'
                              else 'belly controls pickup; head block detection off'
                              if actions and handover.phase == 'BELLY' else head_tracker.reason)
                    cv2.putText(display,detail,(8,92),0,.4,(0,255,255),1)
                selected = belly_block if name == 'belly' else head_block
                if selected is not None:
                    cv2.rectangle(display,(int(selected.x),int(selected.y)),
                                  (int(selected.x+selected.width),int(selected.bottom)),(0,255,255),3)
                cv2.putText(display,status,(8,70),0,.5,(255,255,255),1)
                if name == 'belly' and pickup_active and reference is not None and 'pickup' in reference:
                    edge = PICKUP_TOP
                    y = round(edge*display.shape[0])
                    cv2.line(display,(0,y),(display.shape[1]-1,y),(0,255,255),1)
                    line_label = 'pickup top: y <= 20% (80% from bottom)'
                    cv2.putText(display,line_label,(8,max(110,y-6)),0,.4,(0,255,255),1)
                if name == 'head' and handover.angle == DELIVERY_HEAD_POSITION:
                    target = drop_reference(reference)
                    if target is not None:
                        height,width = display.shape[:2]
                        cv2.rectangle(display,(round(target.x*width),round(target.y*height)),
                                      (round((target.x+target.width)*width),round(target.bottom*height)),(0,255,255),1)
                        if qr is not None:
                            current = qr.normalized(display.shape)
                            cv2.putText(display,f'D size W={current.width/target.width:.0%} H={current.height/target.height:.0%}',
                                        (8,115),0,.5,(0,255,255),1)
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
                if key == ord('d') and (qr_tracker.source != 'decoded' or len(matches) != 1):
                    print('D需当前画面精确解码到唯一目标二维码，不能只用模板跟踪保存。');continue
                if (reference is None or reference.get('version') not in (2,3)
                        or reference.get('cameras') != settings or reference.get('shapes') != shapes
                        or reference.get('target_qr') != target_qr):
                    if key == ord('d') and reference is not None:
                        print('现有标定的相机/尺寸/目标不匹配，D未保存；先恢复原配置，保留原H参考。');continue
                    reference = dict(version=3,cameras=settings,head_position=INITIAL_HEAD_POSITION,
                                     target_qr=target_qr,shapes=shapes)
                frame = bf if key == ord('h') else hf
                normalized = box.normalized(frame.shape)
                if key == ord('d') and (normalized.x <= .01 or normalized.y <= .01
                                       or normalized.x+normalized.width >= .99 or normalized.bottom >= .99):
                    print('D需完整二维码，离画面边缘留出距离后再保存。');continue
                reference['version'] = 3
                if key == ord('d'): reference['drop_head_position'] = DELIVERY_HEAD_POSITION
                if key == ord('h'):
                    reference['pickup_mode'] = 'visible_region'
                    reference['pickup_clipped'] = clipped_edges(normalized)
                reference['pickup' if key == ord('h') else 'drop'] = [normalized.x,normalized.y,normalized.width,normalized.height]
                REFERENCE_FILE.write_text(json.dumps(reference,ensure_ascii=False,indent=2))
                print('已保存:',REFERENCE_FILE)
            if not actions or head_busy or time.monotonic() < ready_at: continue
            if planner.actions >= 40 or time.monotonic()-planner.started_at >= RUN_SECONDS:
                print(f'40 actions / {RUN_SECONDS} seconds limit');return False
            if planner.phase == 'DELIVER' and qr is None and search_right_actions:
                # 短暂漏读先等，随后向最后看到二维码的一侧有限找回；不按旧框前进或放下。
                if qr_missing_since is None: qr_missing_since = time.monotonic()
                planner.reset_confirmation()
                planner.reason = 'QR missing: hold and reacquire'
                if len(matches) > 1 or time.monotonic()-qr_missing_since < QR_WAIT_SECONDS: continue
                if search_turns >= search_right_actions:
                    print('连续有限搜索后仍未找回目的地二维码，停止');return False
                search_action = 'LEFT_HOLDBOX' if last_qr_x is not None and last_qr_x < .5 else 'RIGHT_HOLDBOX'
                search_action = steering(search_action,settings['head']['flip'])
                move.robotMove(search_action)
                search_turns += 1;planner.mark_sent(search_action)
                head_eye.discard_frames(1);belly_eye.discard_frames(1)
                qr_tracker.after_move();qr_missing_since = None
                ready_at = time.monotonic()+DELIVERY_OBSERVE_SECONDS
                continue
            if planner.phase == 'PICKUP':
                if handover.phase == 'HEAD' and head_block is not None and head_tracker.frames >= TRACK_FRAMES:
                    box = head_block.normalized(hf.shape)
                    transferring = transferring or near_head_target(box)
            if planner.phase == 'PICKUP' and handover.phase == 'HEAD':
                planner.flip = settings['head']['flip']
                if head_block is not None:
                    if head_tracker.frames >= TRACK_FRAMES:
                        angle = follow_head_angle(head_block.normalized(hf.shape),handover.angle)
                        if angle != handover.angle:
                            # 跟随提前continue会跳过下方搜索方向重置；在这里同步重置。
                            head_search_direction = 1
                            handover.angle = angle
                            servo.begin_vertical(angle,settle_seconds=.18)
                            head_tracker.after_head_move();planner.reset_confirmation()
                            planner.reason = 'block near image edge; adjust head then reconfirm'
                            ready_at = time.monotonic()+.6
                            continue
                    # 当前目标仍可见，按视觉确认靠近，不消耗丢失后的惯性步数。
                    action = planner.approach(head_block.normalized(hf.shape),head_tracker.frames)
                elif belly_blocks:
                    action = 'WAIT'
                    planner.reset_confirmation()
                    planner.reason = belly_tracker.reason+'; waiting for belly confirmation'
                elif transferring:
                    # 已确认近处目标丢失，先低头找回；到边界才沿用有限交接小步。
                    if handover.angle < handover.down:
                        handover.angle = min(handover.down,handover.angle+HEAD_SEARCH_STEP)
                        servo.begin_vertical(handover.angle,settle_seconds=.5)
                        head_tracker.after_head_move();planner.reset_confirmation()
                        planner.reason = 'near block lost; lower head then observe 1.5s'
                        ready_at = time.monotonic()+HEAD_SEARCH_WAIT
                        continue
                    if transfer_steps >= TRANSFER_STEPS:
                        print('近处目标丢失后已惯性前进3次，腹部仍未接手，停止避免撞箱子');return False
                    action = 'UP_LITTLE'
                    planner.reason = f'camera transfer: forward {transfer_steps+1}/{TRANSFER_STEPS}; head fixed'
                    transfer_steps += 1
                elif head_block is None and not belly_blocks:
                    action = 'LOWER_HEAD' if head_search_direction > 0 else 'RAISE_HEAD'
                    planner.reason = 'block lost; search by 3 units, observe 1.5s; hold body'
                else:
                    action = planner.approach(head_block.normalized(hf.shape) if head_block else None,head_tracker.frames)
                # 未通过关联的候选不能阻止头部搜索；身体仍等待重新确认。
                if head_block is not None or belly_blocks:
                    head_search_direction = 1
                    if action in ('REACQUIRE','LOWER_HEAD','RAISE_HEAD'): action = 'WAIT'
                if action in ('LOWER_HEAD','RAISE_HEAD'):
                    angle = max(HEAD_SEARCH_MIN,min(handover.down,handover.angle+HEAD_SEARCH_STEP*head_search_direction))
                    if angle in (HEAD_SEARCH_MIN,handover.down): head_search_direction *= -1
                    if angle != handover.angle:
                        handover.angle = angle;servo.begin_vertical(angle,settle_seconds=.5)
                        head_tracker.after_head_move()
                    ready_at = time.monotonic()+HEAD_SEARCH_WAIT
                    continue
            else:
                target = belly_block if planner.phase == 'PICKUP' else qr
                tracker = belly_tracker if planner.phase == 'PICKUP' else qr_tracker
                planner.flip = settings['belly' if planner.phase == 'PICKUP' else 'head']['flip']
                action = planner.decide(target.normalized(bf.shape if planner.phase == 'PICKUP' else hf.shape) if target else None,
                                        tracker.frames, decoded=qr_tracker.source == 'decoded')
                if planner.phase == 'PICKUP' and target is None:
                    planner.reason = belly_tracker.reason+'; hold body'
            if action in ('STOP','DONE'):
                print(planner.reason);return action == 'DONE'
            if action == 'WAIT': continue
            planner.mark_sent(action)
            print('搬运动作:',action);move.robotMove(action)
            if action == 'DOWN_BOX':
                print('放下动作已执行；物块是否实际到位需要现场确认。');return True
            if action == 'HOLD_BOX':
                # 抱取完成后抬头到120，重新识别目的地二维码。
                handover.phase = 'HEAD'
                handover.angle = DELIVERY_HEAD_POSITION;servo.begin_vertical(DELIVERY_HEAD_POSITION)
            head_eye.discard_frames(1);belly_eye.discard_frames(1)
            if action == 'HOLD_BOX':
                head_tracker.reset();belly_tracker.reset()
            else: head_tracker.after_body_move()
            belly_tracker.after_body_move();qr_tracker.after_move()
            handover.reset_follow()
            wait_seconds = DELIVERY_OBSERVE_SECONDS if planner.phase == 'DELIVER' else .5
            ready_at = time.monotonic()+wait_seconds


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
