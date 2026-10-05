"""双摄跟踪与交接；--actions 启用对齐球门并行走带球。"""
from contextlib import ExitStack
import argparse
from collections import deque
import json
import math
from statistics import median
import time

import cv2
from robot_config import ROOT
from roboteye import RobotEye
from ball_debug import PatchTracker
from dual_kick import camera_settings
from kick_shapes import Box
from goal_debug import GoalTracker
from walk_kick import WalkKick, MIN_BALL_SCORE

BODY_SERIAL_PORT = "/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0"
# 修改这里即可调整本流程的初始位置和接管后的回正位置。
INITIAL_HEAD_POSITION = 129


def start_robot():
    """建立身体连接并发送初始站姿；后续可由 main() 调用并复用。"""
    from robotmove import RobotMove
    print('Startup robot: handshake, then STAND.')
    # RobotMove 的构造函数已包含握手和 stand()，不要重复发站姿。
    return RobotMove(None,port=BODY_SERIAL_PORT)


def same_view(saved, current):
    # FPS changes timing, not the calibrated pixel positions or camera orientation.
    return (isinstance(saved, dict) and
            {k:v for k,v in saved.items() if k != 'fps'} ==
            {k:v for k,v in current.items() if k != 'fps'})


def validate(head, foot, settings):
    def number(value):
        return type(value) in (int,float) and math.isfinite(value)
    if head.get('version') != 1 or head.get('gpio_bcm') != 27:
        raise ValueError('Need saved head_calibration.json for BCM27')
    f,d = head.get('forward'),head.get('down_sign')
    if type(f) is not int or d not in (-1,1) or not 85 <= f <= 180:
        raise ValueError('Invalid measured head poses')
    bounds = head.get('bounds')
    if (not isinstance(bounds,list) or len(bounds) != 2
            or not all(number(n) for n in bounds)
            or not 85 <= bounds[0] <= bounds[1] <= 180
            or (d == 1 and f < bounds[0]) or (d == -1 and f > bounds[1])):
        raise ValueError('Head poses outside calibrated bounds')
    if foot.get('version') != 2 or foot.get('method') != 'yellow_green_visible_region':
        raise ValueError('Need color-mode right_foot_reference.json version 2')
    cameras = head.get('cameras',{})
    if (not isinstance(cameras,dict) or set(cameras) != set(settings)
            or not all(same_view(cameras[name],view) for name,view in settings.items())
            or not same_view(foot.get('camera'),settings['belly'])):
        raise ValueError('Camera view changed; restore calibrated device/backend/flip')
    area = foot.get('visible_patch_box')
    if (not isinstance(area,list) or len(area) != 4 or not all(number(n) for n in area)
            or not 0 <= area[0] < 1 or not 0 <= area[1] < 1
            or not 0 < area[2] <= 1-area[0]+1e-6 or not 0 < area[3] <= 1-area[1]+1e-6):
        raise ValueError('Invalid measured right-foot visible box')
    for shape in [head.get('shapes',{}).get('head'),head.get('shapes',{}).get('belly'),foot.get('shape')]:
        if not isinstance(shape,list) or len(shape) != 2 or not all(type(n) is int and n>0 for n in shape):
            raise ValueError('Invalid calibrated resolution')
    if foot['shape'] != head['shapes']['belly']:
        raise ValueError('Belly resolution mismatch between calibrations')
    color = foot.get('color',{})
    for key,low,high in [('hue',0,179),('hue_width',1,45),('min_s',0,255),('min_v',0,255)]:
        if not number(color.get(key)) or not low <= color[key] <= high:
            raise ValueError('Invalid ball color: '+key)


def foot_match_details(box, shape, reference, edges):
    if box is None:
        return dict(match=False,reason='ball not detected',detail='')
    x,y,w,h = box
    height,width = shape
    rx,ry,rw,rh = reference['visible_patch_box']
    target_edges = set(reference.get('clipped_edges',[]))
    dx = (x+w/2)/width-rx-rw/2
    dy = (y+h/2)/height-ry-rh/2
    x_ok,y_ok = abs(dx) <= .06,abs(dy) <= .06
    # A clipped reference has no reliable center along the clipping axis.
    # Compare its opposite visible edge and allow a small gap at the image border.
    if 'left' in target_edges:
        dx = (x+w)/width-rx-rw
        x_ok = x/width <= .10 and abs(dx) <= .07
    if 'right' in target_edges:
        dx = x/width-rx
        x_ok = (x+w)/width >= .90 and abs(dx) <= .07
    if 'top' in target_edges:
        dy = (y+h)/height-ry-rh
        y_ok = y/height <= .10 and abs(dy) <= .07
    if 'bottom' in target_edges:
        dy = y/height-ry
        y_ok = (y+h)/height >= .90 and abs(dy) <= .07
    width_ratio,height_ratio = (w/width)/rw,(h/height)/rh
    size_ok = .7 <= width_ratio <= 1.35 and .7 <= height_ratio <= 1.35
    failures = [name for name,ok in [('X position',x_ok),('Y position/border',y_ok),('visible size',size_ok)] if not ok]
    return dict(match=not failures,reason=', '.join(failures) if failures else 'geometry matched',
                dx=dx,dy=dy,x_ok=x_ok,y_ok=y_ok,width_ratio=width_ratio,height_ratio=height_ratio,
                detail=f'X err={dx*width:+.0f}px Y err={dy*height:+.0f}px W={width_ratio:.0%} H={height_ratio:.0%}')


def at_right_foot(box, shape, reference, edges):
    return foot_match_details(box,shape,reference,edges)['match']


class Handover:
    def __init__(self, profile):
        self.forward = profile['forward']
        self.direction = profile['down_sign']
        self.angle = self.forward
        self.phase = 'HEAD'
        bounds = profile['bounds']
        # 取消旧标定的低头终点（137），保留底层驱动指令范围。
        self.down = 180 if self.direction == 1 else 85
        self.up_limit = max(85,bounds[0]) if self.direction == 1 else min(180,bounds[1])
        self.follow_direction = 0
        self.follow_confirm = 0
        self.confirm = 0
        self.positions = deque(maxlen=3)
        self.follow_since = None
        self.last_move_at = -math.inf
        self.last_move_direction = 0

    def reset_follow(self):
        self.positions.clear()
        self.follow_direction = self.follow_confirm = 0
        self.follow_since = None
        self.confirm = 0

    def step(self, head_box, belly_frames, height, now=None):
        if self.phase != 'HEAD': return None
        now = time.monotonic() if now is None else now
        if head_box is not None:
            self.positions.append((head_box[1]+head_box[3]/2)/height)
            cy = median(self.positions)
        else:
            self.positions.clear()
            cy = None
        # 腹部稳定接住近处球即可交接，不要求头部先到固定角度。
        if belly_frames >= 5:
            self.confirm += 1
            if self.confirm >= 3:
                self.phase = 'BELLY'
                self.angle = self.forward
                return self.forward
            # 确认交接时暂停追头，避免低头动作把交接计数又清零。
            return None
        else: self.confirm = 0
        direction = 1 if cy is not None and cy > .65 else -1 if cy is not None and cy < .35 else 0
        if direction == 0:
            self.follow_direction = self.follow_confirm = 0
            self.follow_since = None
            return None
        if direction != self.follow_direction:
            self.follow_since = now
        self.follow_confirm = self.follow_confirm+1 if direction == self.follow_direction else 1
        self.follow_direction = direction
        # Frame count alone becomes over-sensitive as processing FPS improves.
        # Require a sustained error and extra time before reversing direction.
        interval = .4 if self.last_move_direction and direction != self.last_move_direction else .18
        if (self.follow_confirm < 3 or now-self.follow_since < .10
                or now-self.last_move_at < interval): return None
        low,high = sorted((self.up_limit,self.down))
        # Larger image error takes 2-3 units; close to center stays at one unit.
        units = min(3,max(1,round(abs(cy-.5)*8)))
        candidate = max(low,min(high,self.angle+direction*self.direction*units))
        if low <= candidate <= high and candidate != self.angle:
            self.angle = candidate
            self.last_move_at = now
            self.last_move_direction = direction
            self.reset_follow()
            return candidate
        return None


def run(forward=INITIAL_HEAD_POSITION, fps=30, settle=.18, actions=False):
    if not 0 < fps <= 120 or not .08 <= settle <= 2.0:
        raise ValueError('fps must be >0..120; settle must be 0.08..2.0s')
    head_profile = json.loads((ROOT/'config/head_calibration.json').read_text(encoding='utf-8-sig'))
    foot = json.loads((ROOT/'config/right_foot_reference.json').read_text(encoding='utf-8-sig'))
    # Apply the user's new limits in memory; preserve the field calibration files.
    head_profile.update(forward=forward)
    settings = camera_settings()
    for view in settings.values(): view['fps'] = fps
    validate(head_profile,foot,settings)  # All checks precede physical control.
    state = Handover(head_profile)
    body = WalkKick() if actions else None
    ht,bt = PatchTracker(),PatchTracker()
    gt = GoalTracker()
    for key in ('hue','hue_width','min_s','min_v'): setattr(bt,key,foot['color'][key])
    print(f'Startup/return={forward}; head command limits={sorted((state.up_limit,state.down))}; requested FPS={fps}.')
    print('Smoothed ball feedback: below 65%=lower, above 35%=raise; center/lost=hold.')
    print(f'1-3 units after >=3 frames and 0.10s; interval>=0.18s; reverse>=0.4s; settle={settle}s.')
    print('GOAL ALIGN + WALK enabled; no kick action.' if actions else 'Head only; no body serial, walking or kick.')
    if actions: print('Fixed robot serial:',BODY_SERIAL_PORT)
    print('G=restart head phase; R=unlock trackers; Q=quit. Belly ball follows image center.')
    print('Align goal and ball, walk one action, then observe again.' if actions else 'Move ball by hand from head view into belly view.')
    if actions: print('Loss recovery: visual search -> repeat 4 down / 4 up head moves, 1.5s per pose; Q to quit.')
    print('Saved calibrations are read, not overwritten.')
    print('S=select HEAD ball; B=select BELLY ball. Drag a green ball region, ENTER=confirm, C=cancel.')
    print('Selection freezes the image and head control; R starts a fresh search.')
    print('Nonblocking head moves keep preview live. Timing shows processing FPS, camera-read and detection time.')
    enabled = False
    move = None
    body_ready_at = 0.0
    with ExitStack() as stack:
        def close_windows():
            try: cv2.destroyAllWindows()
            except cv2.error: pass
        stack.callback(close_windows)
        head = RobotEye(**settings['head'],latest=True); stack.callback(head.close)
        belly = RobotEye(**settings['belly'],latest=True); stack.callback(belly.close)
        # Verify both saved image sizes before applying the requested forward pose.
        okh,initial_head = head.getImage(); okb,initial_belly = belly.getImage()
        if (not okh or not okb or list(initial_head.shape[:2]) != head_profile['shapes']['head']
                or list(initial_belly.shape[:2]) != foot['shape']):
            raise ValueError('Cannot verify camera resolutions before startup head reset')
        if actions:
            move = start_robot(); stack.callback(move.close)
            # 启动站姿后必须读取新画面，再做行走判断。
            body_ready_at = time.monotonic()+.5
        from Head import RobotHeadServoOnly
        servo = RobotHeadServoOnly(hold=True); stack.callback(servo.cleanup)
        servo.turn_vertical(state.forward)
        enabled = True
        head.discard_frames(1); belly.discard_frames(1)
        measured_at = time.monotonic()
        measured_frames = 0
        read_total = detect_total = 0.0
        timing = 'measuring processing FPS...'
        while True:
            read_at = time.monotonic()
            okh,hf = head.getImage(); okb,bf = belly.getImage()
            read_total += time.monotonic()-read_at
            if not okh or not okb: raise RuntimeError('Camera read failed; no further head commands')
            if list(hf.shape[:2]) != head_profile['shapes']['head'] or list(bf.shape[:2]) != foot['shape']:
                raise ValueError('Resolution changed from saved calibration')
            head_busy = servo.is_moving()
            detect_at = time.monotonic()
            # Moving-camera frames are for preview; use settled frames for
            # target decisions while the belly continues to track throughout.
            hb = None if head_busy or state.phase != 'HEAD' else ht.update(hf)
            bb = bt.update(bf)
            if ht.score < MIN_BALL_SCORE: hb = None
            if bt.score < MIN_BALL_SCORE: bb = None
            if enabled and not head_busy:
                # 交接看连续识别次数；半球进画面时大小变化，不要求位置已静止。
                angle = state.step(hb,bt.frames if bb is not None else 0,hf.shape[0])
                if angle is not None:
                    if state.phase == 'HEAD':
                        servo.begin_vertical(angle,settle_seconds=settle)
                    else:
                        servo.begin_vertical(angle)
                    head_busy = True
                    ht.notify_pitch_change()  # Keep horizontal identity gate when head tilts.
                    print('Head:',angle,'phase:',state.phase)
                    if state.phase == 'BELLY':
                        print('Belly took over; head returned to forward pose.')
            status = 'ALIGN GOAL + BALL' if state.phase == 'BELLY' else 'HEAD FOLLOW'
            if not enabled: status = 'G TO ENABLE HEAD'
            # 俯仰移动中不确认球门；腹部接管后头部回正，持续看门。
            goal = None
            if not head_busy:
                goal = gt.update(hf)
            else:
                gt.reset()
            detect_total += time.monotonic()-detect_at
            measured_frames += 1
            elapsed = time.monotonic()-measured_at
            if elapsed >= 1.0:
                timing = (f'view={measured_frames/elapsed:.1f}fps '
                          f'read={1000*read_total/measured_frames:.0f}ms '
                          f'detect={1000*detect_total/measured_frames:.0f}ms')
                measured_at = time.monotonic()
                measured_frames = 0
                read_total = detect_total = 0.0
            for name,frame,box in [('head',hf,hb),('belly',bf,bb)]:
                display = frame.copy()
                if box:
                    x,y,w,h = box; cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
                if name == 'belly':
                    width,height = frame.shape[1],frame.shape[0]
                    for fraction in (.42,.5,.58):
                        cv2.line(display,(round(width*fraction),0),(round(width*fraction),height-1),(255,0,255),1)
                if name == 'head' and isinstance(goal,Box):
                    cv2.rectangle(display,(int(goal.x),int(goal.y)),(int(goal.x+goal.width),int(goal.bottom)),(255,255,0),2)
                if name == 'head':
                    cv2.line(display,(frame.shape[1]//2,0),(frame.shape[1]//2,frame.shape[0]-1),(255,255,0),1)
                    cv2.putText(display,f'{gt.reason} stable={gt.stable_frames}/5',(8,226),0,.43,(255,255,0),1)
                if name == 'head' and state.phase == 'HEAD':
                    for fraction in (.35,.65):
                        cv2.line(display,(0,round(frame.shape[0]*fraction)),
                                 (frame.shape[1]-1,round(frame.shape[0]*fraction)),(0,165,255),1)
                cv2.putText(display,status+f' head={state.angle}',(8,25),0,.55,(0,255,0),1)
                cv2.putText(display,body.reason if actions else 'NO WALK / NO KICK',(8,48),0,.5,(255,255,255),1)
                tracker = ht if name == 'head' else bt
                cv2.putText(display,tracker.reason,(8,116),0,.43,(255,255,255),1)
                minimum = MIN_BALL_SCORE
                cv2.putText(display,f'shape score={tracker.score:.2f} minimum={minimum:.2f}',
                            (8,204),0,.43,(255,255,255),1)
                cv2.putText(display,'R=new search; S=head ROI; B=belly ROI; drag then ENTER',(8,138),0,.43,(255,255,255),1)
                cv2.putText(display,timing+(' HEAD MOVING' if head_busy else ''),(8,160),0,.43,(255,255,255),1)
                camera = head if name == 'head' else belly
                cv2.putText(display,f'capture={camera.stream_status()}',(8,182),0,.43,(255,255,255),1)
                if name == 'belly':
                    cv2.putText(display,f'seen={bt.frames}/5 stable={bt.stable_frames}/5; aim at center band',
                                (8,72),0,.43,(255,255,255),1)
                cv2.imshow(name,display)
            key = cv2.waitKey(1)&255
            if 65 <= key <= 90: key += 32
            if key in (ord('q'),27): return not actions
            if key in (ord('s'),ord('b')):
                if head_busy:
                    print('Wait for head to settle, then press S/B to select.')
                    continue
                name,frame = ('head',hf) if key == ord('s') else ('belly',bf)
                # Select on a frozen, unannotated frame, not on a later pose.
                selection = cv2.selectROI(name,frame,True,False)
                if selection[2] == 0 or selection[3] == 0: continue
                tracker = PatchTracker()
                try:
                    tracker.select(frame,selection)
                except ValueError as error:
                    print(name,'selection rejected:',error)
                    continue
                if name == 'head': ht = tracker
                else: bt = tracker
                state.reset_follow()
                head.discard_frames(1); belly.discard_frames(1)
                measured_at = time.monotonic(); measured_frames = 0
                read_total = detect_total = 0.0
                print(name,'ball selected:',tracker.color_profile())
                if body:
                    body.reset_observation()
                    body.reset_search()
                continue
            if key == ord('g'):
                gt.reset()
                state = Handover(head_profile)
                servo.begin_vertical(state.forward)
                enabled = True
                ht.notify_pitch_change()
                if body:
                    body.reset_observation()
                    body.reset_search()
                continue
            if key == ord('r'):
                gt.reset()
                ht,bt = PatchTracker(),PatchTracker()
                for name in ('hue','hue_width','min_s','min_v'): setattr(bt,name,foot['color'][name])
                state.reset_follow()
                if body:
                    body.reset_observation()
                    body.reset_search()
                continue

            if not actions or head_busy or time.monotonic() < body_ready_at:
                continue
            ambiguous = 'competing' in bt.reason or (state.phase == 'HEAD' and 'competing' in ht.reason)
            view = settings['head'] if state.phase == 'HEAD' else settings['belly']
            action = body.decide(state.phase,hb,bb,ht.stable_frames,bt.stable_frames,
                                 hf.shape[:2] if state.phase == 'HEAD' else bf.shape[:2],
                                 view['flip'],
                                 head_score=ht.score,belly_score=bt.score,ambiguous=ambiguous,
                                 goal=goal.normalized(hf.shape) if goal is not None else None,
                                 goal_stable=gt.stable_frames,head_flip=settings['head']['flip'])
            if action == 'STOP':
                print('Body stopped:',body.reason)
                return False
            if action == 'WAIT':
                continue
            if action in ('REACQUIRE','LOWER_HEAD','RAISE_HEAD'):
                state.phase = 'HEAD'
                state.reset_follow()
                if action in ('LOWER_HEAD','RAISE_HEAD'):
                    low,high = sorted((state.up_limit,state.down))
                    scan_direction = 1 if action == 'LOWER_HEAD' else -1
                    angle = max(low,min(high,state.angle+3*state.direction*scan_direction))
                    moved = angle != state.angle
                    if moved:
                        state.angle = angle
                        servo.begin_vertical(angle,settle_seconds=settle)
                    body.mark_head_search(moved)
                    print('Search head:',state.angle,body.reason)
                else:
                    print('Visual search: reset location lock, preserve ball color.')
                ht.begin_search(); bt.begin_search()
                continue
            blind = body.search_stage == 'VISUAL' and action == 'UP_LITTLE'
            body.mark_sent(action,blind=blind)
            print('Body action:',action,'blind steps:',body.blind_steps)
            move.robotMove(action)
            head.discard_frames(1); belly.discard_frames(1)
            ht.notify_body_move(); bt.notify_body_move()
            gt.reset()
            state.reset_follow()
            body_ready_at = time.monotonic()+.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward',type=int,default=INITIAL_HEAD_POSITION)
    parser.add_argument('--fps',type=float,default=30)
    parser.add_argument('--settle',type=float,default=.18,help='small tracking move settling time, seconds')
    parser.add_argument('--actions',action='store_true',help='enable goal alignment and walking; no kick action')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.forward,args.fps,args.settle,args.actions) else 1)
