"""Interactive head-only calibration; no body serial connection."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
import shutil

import cv2
from robot_config import ROOT
from dual_kick import camera_settings, draw
from kick_shapes import observe
from roboteye import RobotEye


class HeadPosition:
    def __init__(self, start=95):
        if not 85 <= start <= 180:
            raise ValueError('Start command must be within the existing driver range 85..180')
        self.command = start
        self.forward = self.handover = None
        self.down_sign = 1  # Field-confirmed K: increasing command lowers the head.
        self.overlap_confirmation = None
        self.step = 1
        self.active = False

    def bounds(self):
        if self.forward is None or self.down_sign is None:
            return 85, 180
        endpoints = (self.forward-self.down_sign*45, self.forward+self.down_sign*20)
        return max(85,min(endpoints)), min(180,max(endpoints))

    def candidate(self, sign):
        low, high = self.bounds()
        if self.command < low:
            return min(low,self.command+self.step) if sign > 0 else self.command
        if self.command > high:
            return max(high,self.command-self.step) if sign < 0 else self.command
        return max(low,min(high,self.command+sign*self.step))

    def profile(self, settings, shapes):
        if self.forward is None or self.handover is None or self.down_sign is None:
            raise ValueError('Record F, choose J/K down direction, then record H')
        displacement = (self.handover-self.forward)*self.down_sign
        if not 0 < displacement <= 20:
            raise ValueError('H must be below F, within 20 command units')
        return dict(version=1, gpio_bcm=27, forward=self.forward,
                    handover=self.handover, down_sign=self.down_sign,
                    bounds=list(self.bounds()), cameras=settings, shapes=shapes,
                    overlap_confirmation=self.overlap_confirmation,
                    units='servo command units; physical degrees not calibrated')

    def record_handover(self, source):
        if not self.active or self.forward is None:
            raise ValueError('Enable with G and record forward reference with F first')
        if not 0 < (self.command-self.forward)*self.down_sign <= 20:
            raise ValueError('H must be below F within 20 command units')
        self.handover = self.command
        self.overlap_confirmation = source


def run(start=124, forward=124):
    position = HeadPosition(start)
    if not 85 <= forward <= 180:
        raise ValueError('Forward command outside driver range')
    position.forward = forward
    if not position.bounds()[0] <= start <= position.bounds()[1]:
        raise ValueError('Start outside forward-relative bounds')
    pending = None
    settings = camera_settings()
    path = ROOT/'config/head_calibration.json'
    print('HEAD ONLY. No body motion or serial. Initial command is unverified until G.')
    print('G=enable at displayed command; A/D=-/+; 1/2/5=step; F=forward reference')
    print('Default forward=124, K down direction. F can replace forward reference.')
    print('J/K=down direction; H=overlap; V=confirm SAME ball visually if H detection fails; S=save; Q=quit')
    print('45 up / 20 down are COMMAND offsets, not measured physical angles.')
    with ExitStack() as stack:
        def close_windows():
            try: cv2.destroyAllWindows()
            except cv2.error: pass
        stack.callback(close_windows)
        head = RobotEye(**settings['head']); stack.callback(head.close)
        belly = RobotEye(**settings['belly']); stack.callback(belly.close)
        servo = None
        while True:
            okh, hf = head.getImage(); okb, bf = belly.getImage()
            if not okh or not okb:
                raise RuntimeError('Camera read failed; no further head commands')
            ht, bt = observe(hf), observe(bf,with_goal=False)
            status = f'cmd={position.command} step={position.step} limits={position.bounds()} enabled={position.active}'
            draw(hf,ht,'head',status); draw(bf,bt,'belly',status)
            cv2.imshow('head',hf); cv2.imshow('belly',bf)
            key = cv2.waitKey(20)&255
            if 65 <= key <= 90: key += 32
            if key in (ord('q'),27):
                return True  # Do not jump back to an unverified reference on exit.
            if key in (ord('1'),ord('2'),ord('5')): position.step = int(chr(key))
            if key == ord('g') and not position.active:
                from Head import RobotHeadServoOnly
                servo = RobotHeadServoOnly(); stack.callback(servo.cleanup)
                servo.turn_vertical(position.command)
                position.active = True
                head.discard_frames(); belly.discard_frames()
            if key in (ord('a'),ord('d')):
                if not position.active:
                    print('Press G to enable the displayed start command first.'); continue
                candidate = position.candidate(-1 if key == ord('a') else 1)
                if candidate != position.command:
                    servo.turn_vertical(candidate); position.command = candidate
                    pending = None
                    head.discard_frames(); belly.discard_frames()
            if key == ord('f') and position.active:
                position.forward = position.command
                position.handover = position.overlap_confirmation = pending = None
                print('Forward recorded. Confirmed K down direction retained:',position.down_sign)
            if key in (ord('j'),ord('k')) and position.forward is not None:
                direction = -1 if key == ord('j') else 1
                if direction != position.down_sign:
                    position.handover = position.overlap_confirmation = pending = None
                position.down_sign = direction
                print('Down direction:',position.down_sign,'limits:',position.bounds())
                if not position.bounds()[0] <= position.command <= position.bounds()[1]:
                    print('Current position outside new bounds. Return using A/D; H and S are blocked.')
            if key == ord('h'):
                if (position.active and position.forward is not None
                        and 0 < (position.command-position.forward)*position.down_sign <= 20):
                    if ht['ball'] is not None and bt['ball'] is not None:
                        position.record_handover('unique_shape_candidates; identity checked by operator')
                        pending = None
                        print('H recorded:',position.command,'. Verify SAME ball visually, then S.')
                    else:
                        position.handover = position.overlap_confirmation = None
                        pending = position.command
                        print('H detection incomplete: head balls=',ht['ball_count'],'belly balls=',bt['ball_count'])
                        print('If BOTH live views show the SAME ball, press V to confirm manually. Otherwise reposition and H.')
                else:
                    pending = None
                    print('H rejected: enable G; forward required; lower pose must be within 20 command units.')
            if key == ord('v'):
                if pending is not None and pending == position.command:
                    position.record_handover('operator_visual_confirmation; shape detection incomplete')
                    pending = None
                    print('Manual H recorded:',position.command,'. Press S to save head poses.')
                else:
                    print('Press H at current overlap first; moving cancels pending confirmation.')
            if key == ord('s'):
                try:
                    saved = position.profile(settings,{'head':list(hf.shape[:2]),'belly':list(bf.shape[:2])})
                    if path.exists():
                        shutil.copy2(path,path.with_name(path.stem+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.bak.json'))
                    temporary = path.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps(saved,indent=2)+'\n',encoding='utf-8')
                    temporary.replace(path)
                    print('Saved:',path,'Head poses only; automatic kick has NOT been enabled.')
                except (ValueError,OSError) as error: print('Not saved:',error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=int,default=124,help='forward command; G explicitly applies it')
    parser.add_argument('--forward',type=int,default=124,help='forward reference, adjustable with F')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.start,args.forward) else 1)
