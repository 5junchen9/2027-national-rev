"""双摄形状反馈右脚踢球；预览不打开串口或GPIO，真实运行必须先标定。"""
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import time

import cv2
from robot_config import ROOT
from roboteye import RobotEye, CAMERA_FPS
from kick_shapes import Box, observe

CALIBRATION_PATH = ROOT / 'config/kick_calibration.json'


def camera_settings():
    return {
        'head': {'backend': 'usb', 'device': os.environ.get('ROBOT_KICK_HEAD_DEVICE', '0'),
                 'flip': os.environ.get('ROBOT_KICK_HEAD_FLIP', '0'), 'fps': CAMERA_FPS},
        'belly': {'backend': 'csi', 'device': os.environ.get('ROBOT_KICK_BELLY_DEVICE', '0'),
                  'flip': os.environ.get('ROBOT_KICK_BELLY_FLIP', 'none'), 'fps': CAMERA_FPS}}


def validate_calibration(profile):
    if not isinstance(profile, dict):
        raise ValueError('Calibration must be an object')
    if profile.get('version') != 1:
        raise ValueError('Unsupported calibration version')
    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    for key in ('handover_y', 'goal_aim_x', 'ball_radius'):
        if not number(profile.get(key)) or not 0 < profile[key] < 1:
            raise ValueError('Invalid calibration: '+key)
    if not .05 < profile['handover_y'] < .98 or profile['ball_radius'] >= .4:
        raise ValueError('Handover must precede disappearance; invalid ball radius')
    if type(profile.get('near_y_direction')) is not int or profile['near_y_direction'] not in (-1,1):
        raise ValueError('Missing measured belly near/far direction')
    overlap_ball = profile.get('handover_belly')
    if (not isinstance(overlap_ball,list) or len(overlap_ball) != 4
            or not all(number(n) and 0 < n < 1 for n in overlap_ball)
            or overlap_ball[2]/2 >= profile['ball_radius']*.9):
        raise ValueError('Record an overlapping handover before the ball reaches the kick position')
    roi = profile.get('right_foot_roi')
    if (not isinstance(roi, list) or len(roi) != 4 or not all(number(v) for v in roi)
            or not 0 <= roi[0] < roi[2] <= 1 or not 0 <= roi[1] < roi[3] <= 1
            or not .01 <= roi[2]-roi[0] <= .35 or not .01 <= roi[3]-roi[1] <= .35):
        raise ValueError('Select a small right-foot target area (width/height .01.. .35 of frame)')
    for key in ('head_shape', 'belly_shape'):
        shape = profile.get(key)
        if not isinstance(shape, list) or len(shape) != 2 or not all(type(n) is int and n > 0 for n in shape):
            raise ValueError('Invalid frame dimensions: '+key)
    if profile.get('cameras') != camera_settings():
        raise ValueError('Camera configuration changed; recalibrate with current settings')
    return profile


def load_calibration(path):
    if not path.is_file():
        raise ValueError('Calibration missing. Run --mode dual --preview, then H / two belly clicks / S.')
    return validate_calibration(json.loads(path.read_text(encoding='utf-8')))


class KickPlanner:
    """每次只决定一个已有动作；距离和90度转角不用于猜走几步。"""
    def __init__(self, calibration, confirm_frames=3):
        self.profile = calibration
        self.phase = 'approach'
        self.confirm_frames = confirm_frames
        self.proposal, self.count, self.anchor = None, 0, None
        self.missing = 0
        self.kicked = False
        self.reason = 'waiting for one ball and one goal'

    def reset_confirmation(self):
        self.proposal, self.count, self.anchor = None, 0, None

    def decide(self, head_ball, goal, belly_ball=None):
        if self.kicked:
            self.reason = 'kick already attempted; no retry'
            return 'STOP'
        ball = head_ball if self.phase == 'approach' else belly_ball
        if goal is None or ball is None:
            self.reset_confirmation()
            self.missing += 1
            self.reason = 'target missing or ambiguous; no motion'
            return 'STOP' if self.missing >= 15 else 'WAIT'
        self.missing = 0
        p = self.profile
        error = goal.cx-p['goal_aim_x']
        tolerance = max(.025, min(.08, goal.width*.22))
        if self.phase == 'approach' and head_ball.bottom >= p['handover_y']:
            action = 'OPEN_BELLY'
        elif abs(error) > tolerance:
            action = 'TURN_RIGHT' if error > 0 else 'TURN_LEFT'
            if p['cameras']['head']['flip'] in ('1', '-1'):
                action = 'TURN_LEFT' if action == 'TURN_RIGHT' else 'TURN_RIGHT'
        elif self.phase == 'approach':
            action = 'SIDE_LEFT' if ball.cx < .38 else 'SIDE_RIGHT' if ball.cx > .62 else 'UP_LITTLE'
            if p['cameras']['head']['flip'] in ('1', '-1') and action.startswith('SIDE_'):
                action = 'SIDE_RIGHT' if action == 'SIDE_LEFT' else 'SIDE_LEFT'
        else:
            left, top, right, bottom = p['right_foot_roi']
            radius = ball.width/2
            if ball.cx < left:
                action = 'SIDE_LEFT'
            elif ball.cx > right:
                action = 'SIDE_RIGHT'
            elif ball.cy < top or ball.cy > bottom:
                toward_target = (top+bottom)/2-ball.cy
                action = 'UP_LITTLE' if toward_target*p['near_y_direction'] > 0 else 'BACK'
            elif radius < p['ball_radius']*.75:
                action = 'UP_LITTLE'
            elif radius > p['ball_radius']*1.25:
                action = 'BACK'
            else:
                action = 'RIGHT_BALL'
            if p['cameras']['belly']['flip'] in ('1', '-1') and action.startswith('SIDE_'):
                action = 'SIDE_RIGHT' if action == 'SIDE_LEFT' else 'SIDE_LEFT'
        positions = (ball.cx, ball.cy, goal.cx, goal.cy)
        stable = self.anchor is not None and max(abs(a-b) for a,b in zip(positions,self.anchor)) <= .025
        self.count = self.count+1 if action == self.proposal and stable else 1
        if self.count == 1:
            self.anchor = positions
        self.proposal = action
        self.reason = action+' confirm '+str(self.count)+'/'+str(self.confirm_frames)
        if self.count < self.confirm_frames:
            return 'WAIT'
        self.reset_confirmation()
        if action == 'OPEN_BELLY':
            self.phase = 'near'
        return action

    def mark_sent(self, action):
        # 发送前锁定；即使串口发送中途失败，也不自动补踢。
        if action == 'RIGHT_BALL':
            self.kicked = True


def draw(frame, targets, name, status):
    for kind, color in [('ball', (0,255,255)), ('goal', (0,255,0))]:
        box = targets[kind]
        if box is not None:
            cv2.rectangle(frame, (int(box.x),int(box.y)),
                          (int(box.x+box.width),int(box.bottom)), color, 2)
            cv2.putText(frame, kind, (int(box.x),max(20,int(box.y)-4)), 0, .55, color, 1)
    cv2.putText(frame, name+' balls='+str(targets['ball_count'])+' goals='+str(targets['goal_count']),
                (8,22), 0, .5, (255,255,255), 1)
    cv2.putText(frame, status, (8,44), 0, .45, (255,255,255), 1)


def run(preview=False, calibration_path=None, robot=None):
    path = Path(calibration_path) if calibration_path else CALIBRATION_PATH
    settings = camera_settings()
    profile = None if preview else load_calibration(path)  # 拒绝先连机器人再发现没有标定。
    planner = None if preview else KickPlanner(profile)
    roi, clicks, handover, handover_belly = None, [], None, None
    old = None
    if preview and path.is_file():
        try:
            old = load_calibration(path)
            roi, handover = old['right_foot_roi'], old['handover_y']
            handover_belly = old['handover_belly']
        except (ValueError, json.JSONDecodeError):
            print('Existing calibration incompatible; record H and belly area again. Original retained.')
    shapes = {}
    def mouse(event, x, y, flags, param):
        nonlocal roi
        if event == cv2.EVENT_LBUTTONDOWN and 'belly' in shapes:
            if len(clicks) == 2: clicks.clear()
            height, width = shapes['belly']
            clicks.append((x/width, y/height))
            roi = None
            if len(clicks) == 2:
                roi = [min(clicks[0][0],clicks[1][0]),min(clicks[0][1],clicks[1][1]),
                       max(clicks[0][0],clicks[1][0]),max(clicks[0][1],clicks[1][1])]
    with ExitStack() as stack:
        def close_windows():
            try: cv2.destroyAllWindows()
            except cv2.error: pass
        stack.callback(close_windows)
        head = RobotEye(**settings['head']); stack.callback(head.close)
        belly = None
        def open_belly():
            camera = RobotEye(**settings['belly']); stack.callback(camera.close)
            return camera
        if preview:
            belly = open_belly()
            cv2.namedWindow('belly', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('belly', mouse)
            print('No motion. H=record head handover; belly two clicks=right-foot area; S=save; R=clear; Q=quit.')
        log = None
        if not preview:
            log_dir = ROOT/'logs'; log_dir.mkdir(exist_ok=True)
            log = stack.enter_context((log_dir/('dual-kick-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.jsonl')).open('x',encoding='utf-8'))
            log.write(json.dumps({'event':'start','profile':profile})+'\n'); log.flush()
        deadline = time.monotonic()+120
        actions = 0
        move = robot
        while preview or time.monotonic() < deadline:
            head_ok, hf = head.getImage()
            belly_ok, bf = belly.getImage() if belly is not None else (False,None)
            if not head_ok or (belly is not None and not belly_ok):
                if not preview:
                    print('Camera read failed; stopping without further actions.'); return False
                cv2.waitKey(20)
                continue
            shapes['head'] = list(hf.shape[:2])
            if bf is not None: shapes['belly'] = list(bf.shape[:2])
            if preview and old is not None and any(shapes[name] != old[name+'_shape'] for name in shapes):
                roi, handover, handover_belly, old = None,None,None,None
                print('Frame dimensions changed; record H and target area again.')
            if not preview:
                for name in shapes:
                    if shapes[name] != profile[name+'_shape']:
                        raise ValueError('Resolution changed; recalibrate: '+name)
            ht = observe(hf)
            bt = observe(bf,with_goal=False) if bf is not None else {'ball':None,'goal':None,'ball_count':0,'goal_count':0}
            normalized = lambda box, frame: box.normalized(frame.shape) if box is not None else None
            if preview:
                status = 'H / belly clicks / S / R / Q'
                action = 'WAIT'
            else:
                action = planner.decide(normalized(ht['ball'],hf),normalized(ht['goal'],hf),
                                        normalized(bt['ball'],bf) if bf is not None else None)
                status = planner.phase+' '+planner.reason
            draw(hf,ht,'head',status)
            if preview and handover is not None:
                cv2.line(hf,(0,int(handover*hf.shape[0])),(hf.shape[1]-1,int(handover*hf.shape[0])),(255,0,255),2)
            cv2.imshow('head',hf)
            if bf is not None:
                draw(bf,bt,'belly',status)
                area = roi if preview else profile['right_foot_roi']
                if area:
                    h,w = bf.shape[:2]
                    cv2.rectangle(bf,(int(area[0]*w),int(area[1]*h)),(int(area[2]*w),int(area[3]*h)),(255,0,255),2)
                cv2.imshow('belly',bf)
            key = cv2.waitKey(1)&0xff
            if ord('A') <= key <= ord('Z'): key += 32
            if key in (ord('q'),27): return True if preview else False
            if preview:
                if key == ord('r'):
                    roi, handover, handover_belly = None,None,None
                    clicks.clear()
                if key == ord('h'):
                    if ht['ball'] is not None and bt['ball'] is not None:
                        handover = ht['ball'].bottom/hf.shape[0]
                        hb = bt['ball'].normalized(bf.shape)
                        handover_belly = [hb.cx,hb.cy,hb.width,hb.height]
                        print('Overlapping handover:',handover)
                    else:
                        handover, handover_belly = None,None
                        print('Not recorded: both cameras must see the same single ball at handover.')
                if key == ord('s'):
                    try:
                        if ht['goal'] is None or bt['ball'] is None or roi is None:
                            raise ValueError('Need exactly one head goal, one belly ball and a selected target area')
                        ball = bt['ball'].normalized(bf.shape)
                        if not roi[0] <= ball.cx <= roi[2] or not roi[1] <= ball.cy <= roi[3]:
                            raise ValueError('Place the ball in the tested right-foot area first')
                        if handover_belly is None or abs(ball.cy-handover_belly[1]) < .02:
                            raise ValueError('Record H at a farther overlapping pose, then place ball at kick position')
                        saved = validate_calibration({'version':1,'handover_y':handover,
                            'goal_aim_x':ht['goal'].cx/hf.shape[1],'right_foot_roi':roi,
                            'near_y_direction':1 if ball.cy > handover_belly[1] else -1,
                            'handover_belly':handover_belly,
                            'ball_radius':ball.width/2,'head_shape':shapes['head'],
                            'belly_shape':shapes['belly'],'cameras':settings})
                        if path.exists():
                            shutil.copy2(path,path.with_name(path.stem+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.bak.json'))
                        temporary = path.with_name(path.name+'.tmp')
                        temporary.write_text(json.dumps(saved,indent=2)+'\n',encoding='utf-8')
                        temporary.replace(path)
                        print('Saved:',path)
                    except (ValueError,OSError) as error: print('Not saved:',error)
                continue
            event = {'time':time.monotonic(),'phase':planner.phase,'action':action,'reason':planner.reason,
                     'head':{k:asdict(v) if isinstance(v,Box) else v for k,v in ht.items()},
                     'belly':{k:asdict(v) if isinstance(v,Box) else v for k,v in bt.items()}}
            log.write(json.dumps(event)+'\n'); log.flush()
            if time.monotonic() >= deadline:
                print('Timeout; no new action.'); return False
            if action == 'STOP': print(planner.reason); return False
            if action == 'WAIT': continue
            if action == 'OPEN_BELLY':
                belly = open_belly()
                head.discard_frames()
                planner.reset_confirmation()
                continue
            if move is None:
                from robotmove import RobotMove
                move = RobotMove(None); stack.callback(move.close)
                # 连接会发送站姿，不能继续使用连接前的观测执行动作。
                head.discard_frames()
                if belly: belly.discard_frames()
                planner.reset_confirmation()
                continue
            if actions >= 30:
                print('Action limit; no further movement.'); return False
            planner.mark_sent(action)
            log.write(json.dumps({'event':'sending','action':action})+'\n'); log.flush()
            print('Feedback action:',action)
            move.robotMove(action)
            actions += 1
            if action == 'RIGHT_BALL':
                log.write(json.dumps({'event':'kick_attempt_completed','goal_result':'unverified'})+'\n'); log.flush()
                print('RIGHT_BALL completed once. Goal outcome is not verified.'); return True
            head.discard_frames()
            if belly: belly.discard_frames()
            planner.reset_confirmation()
        print('Timeout; no further movement.'); return False
