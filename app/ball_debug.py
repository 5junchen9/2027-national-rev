"""Manual partial-ball tracking and right-foot reference, no motion."""
from contextlib import ExitStack
from datetime import datetime
import json
import shutil
import time

import cv2
import numpy as np
from robot_config import ROOT
from dual_kick import camera_settings
from roboteye import RobotEye


def fit_visible_arc(contour, shape):
    """丢掉画面截断的直边，只用可见圆弧估计球心和半径。"""
    height,width = shape[:2]
    points = contour.reshape(-1,2).astype(float)
    points = points[(points[:,0] > 3)&(points[:,0] < width-4)
                    &(points[:,1] > 3)&(points[:,1] < height-4)]
    if len(points) < 6:
        return None
    x,y = points[:,0],points[:,1]
    matrix = np.column_stack((2*x,2*y,np.ones(len(points))))
    cx,cy,c = np.linalg.lstsq(matrix,x*x+y*y,rcond=None)[0]
    squared_radius = cx*cx+cy*cy+c
    if squared_radius <= 0:
        return None
    radius = float(np.sqrt(squared_radius))
    if not 6 <= radius <= max(width,height)*.6:
        return None
    error = np.abs(np.hypot(x-cx,y-cy)-radius)
    angles = np.sort(np.arctan2(y-cy,x-cx))
    gaps = np.diff(np.r_[angles,angles[0]+2*np.pi])
    arc = 2*np.pi-max(gaps)
    if arc < np.pi/3 or np.percentile(error,80)/radius > .12:
        return None
    return cx,cy,radius


class PatchTracker:
    """Yellow-green region detection, including frame-clipped balls; no template identity claim."""
    def __init__(self):
        self.box = self.last_box = None
        self.score = 0.0
        self.frames = 0
        self.hue = 40
        self.hue_width = 14
        self.min_s = 55
        self.min_v = 100  # Reject darker wood; a dim ball can be sampled explicitly.
        self.missing = 0
        self.locked = False
        self.edges = []
        self.mask = None
        self.manual = False
        self.reason = 'waiting for yellow-green ball'
        self.seeded = False
        self.stable_frames = 0
        self.pitch_changed = False
        self.color_learned = False
        self.velocity = (0.0,0.0)
        self.seen_at = None
        self.searching = False
        self.partial = False

    def begin_search(self):
        """保留颜色，重新搜位置；候选按形状评分选择。"""
        self.box = self.last_box = None
        self.frames = self.stable_frames = self.missing = 0
        self.score = 0.0
        self.seeded = self.manual = self.pitch_changed = False
        self.velocity = (0.0,0.0)
        self.seen_at = None
        self.searching = True
        self.partial = False
        self.reason = 'visual search: selecting highest shape score'

    def notify_body_move(self):
        # 身体移动同时改变横、纵像素位置，不能沿用仅俯仰的横向锁。
        self.begin_search()

    def notify_pitch_change(self):
        self.pitch_changed = True
        self.box = None
        self.frames = self.stable_frames = 0
        self.velocity = (0.0,0.0)
        self.seen_at = None

    def overlap(self, box):
        if self.last_box is None: return 0.0
        x,y,w,h = box
        lx,ly,lw,lh = self.last_box
        intersection = max(0,min(x+w,lx+lw)-max(x,lx))*max(0,min(y+h,ly+lh)-max(y,ly))
        # A color sample is small; after acquisition use IoU so contained debris
        # cannot score the same as the whole previously tracked ball.
        denominator = lw*lh if self.seeded else w*h+lw*lh-intersection
        return intersection/max(1,denominator)

    def select(self, frame, box):
        x,y,w,h = map(int,box)
        if min(w,h) < 12 or x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            raise ValueError('Select a visible ball patch at least 12x12 pixels')
        if w > frame.shape[1]*.5 or h > frame.shape[0]*.5:
            raise ValueError('Color sample too large: select only visible tennis-ball surface')
        hsv = cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
        pixels = hsv[(hsv[:,:,0] >= 24)&(hsv[:,:,0] <= 85)
                     &(hsv[:,:,1] >= 12)&(hsv[:,:,2] >= 60)]
        if len(pixels) < 100:
            raise ValueError('Sample green/fluorescent-green surface, not white seam, orange floor or background')
        hue = float(np.median(pixels[:,0]))
        if not 24 <= hue <= 85:
            raise ValueError('Sample is outside the green ball color family')
        self.hue = round(hue)
        self.hue_width = 14
        self.min_s = max(35,round(float(np.median(pixels[:,1]))*.35))
        self.min_v = max(60,round(float(np.median(pixels[:,2]))*.45))
        self.last_box = (x,y,w,h)
        self.box = None
        self.frames = 0
        self.missing = 0
        self.locked = False
        self.manual = True
        self.color_learned = True
        self.pitch_changed = False
        self.seeded = True
        self.stable_frames = 0
        self.velocity = (0.0,0.0)
        self.seen_at = None
        self.reason = 'color sampled; finding selected region'

    def color_profile(self):
        return dict(hue=self.hue,hue_width=self.hue_width,min_s=self.min_s,min_v=self.min_v)

    def update(self, frame, _green_retry=False, now=None):
        now = time.monotonic() if now is None else now
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # Search yellow-green and true-green in separate bands. Combining the
        # whole family at once can join a tennis ball to wood/cyan background.
        low,high = max(24,self.hue-self.hue_width),min(85,self.hue+self.hue_width)
        strong = cv2.inRange(hsv,(low,self.min_s,self.min_v),(high,255,255))
        # Pale yellow still has a color hint; plain white is only considered near
        # a previously identified ball, never as a whole-image acquisition rule.
        pale = cv2.inRange(hsv,(low,12,max(160,self.min_v)),(high,255,255))
        green = cv2.bitwise_or(strong,pale)
        # White shirts/walls must not join the ball into one giant contour.
        # Use OpenCV's enclosing circle to bound seam reconstruction locally.
        self.mask = green.copy()
        seeds = cv2.morphologyEx(self.mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        contours,_ = cv2.findContours(seeds,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        white = cv2.inRange(hsv,(0,0,170),(179,65,255)) > 0
        elapsed = (min(.15,max(0,now-self.seen_at))
                   if self.seen_at is not None and self.missing <= 3 else 0.0)
        shift_x,shift_y = (v*elapsed for v in self.velocity)
        for contour in contours:
            area = cv2.contourArea(contour)
            (cx,cy),radius = cv2.minEnclosingCircle(contour)
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull,True)
            if (area < 70 or radius < 4 or not perimeter
                    or area/(np.pi*radius*radius) < .28
                    or 4*np.pi*cv2.contourArea(hull)/(perimeter*perimeter) < .60):
                continue
            # Only visit the circle's bounding region, not every camera pixel
            # once per contour (costly on the Pi with noisy backgrounds).
            cx,cy,radius = round(cx),round(cy),round(radius)
            left,right = max(0,cx-radius),min(frame.shape[1],cx+radius+1)
            top,bottom = max(0,cy-radius),min(frame.shape[0],cy+radius+1)
            support = np.zeros((bottom-top,right-left),np.uint8)
            cv2.circle(support,(cx-left,cy-top),radius,255,-1)
            region = self.mask[top:bottom,left:right]
            region[(support > 0)&white[top:bottom,left:right]] = 255
        if self.last_box is not None and not self.seeded:
            lx,ly,lw,lh = self.last_box
            lx = max(0,min(frame.shape[1]-lw,round(lx+shift_x)))
            ly = max(0,min(frame.shape[0]-lh,round(ly+shift_y)))
            # Pure-white recovery stays within the old footprint. It must not
            # grow into surrounding clothes while green is still present.
            if np.count_nonzero(green[ly:ly+lh,lx:lx+lw]) < 20:
                self.mask[ly:ly+lh,lx:lx+lw] |= (white[ly:ly+lh,lx:lx+lw]*255).astype(np.uint8)
        kernel = np.ones((3,3),np.uint8)
        self.mask = cv2.morphologyEx(self.mask,cv2.MORPH_OPEN,kernel)
        pieces,_ = cv2.findContours(self.mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        span = 50
        if pieces:
            rect = cv2.boundingRect(max(pieces,key=cv2.contourArea))
            span = max(rect[2:])
        close_size = max(5,min(17,int(span*.06)|1))
        self.mask = cv2.morphologyEx(self.mask,cv2.MORPH_CLOSE,np.ones((close_size,close_size),np.uint8))
        contours,_ = cv2.findContours(self.mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        partial_boxes = set()
        outside_lock = False
        for contour in contours:
            area = cv2.contourArea(contour)
            x,y,w,h = cv2.boundingRect(contour)
            margin = max(3,min(12,round(max(w,h)*.03)))
            clipped = (x <= margin or y <= margin or x+w >= frame.shape[1]-margin
                       or y+h >= frame.shape[0]-margin)
            arc = fit_visible_arc(contour,frame.shape) if clipped else None
            if (min(w,h) < 8 or area < 70 or area/(w*h) < .35
                    or w*h > frame.shape[0]*frame.shape[1]*.85):
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            # White seams damage raw perimeter, but do not change the ball's outer shape.
            if hull_area <= 0 or area/hull_area < (.60 if arc is not None else .70):
                continue
            hint_fraction = np.count_nonzero(green[y:y+h,x:x+w])/(w*h)
            if hint_fraction < .05:
                perimeter = cv2.arcLength(hull,True)
                if (not perimeter or 4*np.pi*hull_area/(perimeter*perimeter) < .85
                        or max(w,h)/min(w,h) > 1.35):
                    continue  # A white paper rectangle is not a recovered pale ball.
            # Full candidates must be round-ish; clipped candidates can be only a cap.
            selected = self.manual and self.overlap((x,y,w,h)) >= .3
            # 圆弧不足时仍可走原来的整球外形筛选，不放宽矩形背景。
            if (not clipped or arc is None) and not selected:
                perimeter = cv2.arcLength(hull,True)
                if (area/hull_area < .75 or not perimeter
                        or 4*np.pi*hull_area/(perimeter*perimeter) < .65
                        or max(w,h)/min(w,h) > 1.6):
                    continue
                polygon = cv2.approxPolyDP(hull,.025*perimeter,True)
                if len(polygon) == 4 and cv2.contourArea(polygon)/hull_area > .90:
                    continue  # Solid green rectangles can also pass a loose roundness test.
            if self.last_box:
                # 手动框选仍限定首次目标，避免评分最高的背景抢走选中的球。
                if self.seeded and self.overlap((x,y,w,h)) < .3:
                    continue
                lx,ly,lw,lh = self.last_box
                if not self.seeded and w < lw*.5 and h < lh*.5:
                    continue  # A sudden tiny cap/background fragment is not a reliable handover.
                if not self.seeded:
                    limit = max(25,min(60,max(lw,lh)*.65))
                    limit += min(40,np.hypot(shift_x,shift_y)*.35)
                    dx,dy = x+w/2-lx-lw/2-shift_x,y+h/2-ly-lh/2-shift_y
                    if (abs(dx) > limit if self.pitch_changed else np.hypot(dx,dy) > limit):
                        outside_lock = True
                        continue
                    if not (.25 if arc is not None else .45) <= w/lw <= 2.2: continue
            candidates.append(((x,y,w,h),area/(w*h)))
            if arc is not None:
                partial_boxes.add((x,y,w,h))
        count = len(candidates)
        if candidates:
            # score 是轮廓面积 / 包围框面积，不是识别概率。
            # 先按评分选；同分时选面积较大的候选，避免取决于轮廓顺序。
            best = max(candidates, key=lambda item: (item[1], item[0][2]*item[0][3]))
            candidates = [best]
        if len(candidates) != 1:
            if not candidates and not self.color_learned and not _green_retry:
                original_hue = self.hue
                self.hue = 66
                result = self.update(frame,_green_retry=True,now=now)
                if result is None: self.hue = original_hue
                return result
            self.box = None; self.frames = 0; self.score = 0.0; self.edges = []
            self.partial = False
            self.stable_frames = 0
            self.missing += 1
            self.reason = ('outside target lock: R or select ball' if outside_lock and count == 0
                           else 'no ball shape/color: R or select ball' if count == 0
                           else f'{count} competing color regions: select ball')
            return None
        self.box,self.score = candidates[0]
        self.partial = self.box in partial_boxes
        x,y,w,h = self.box
        stable = False
        if self.last_box is not None and not self.seeded and self.missing == 0:
            lx,ly,lw,lh = self.last_box
            stable = (abs(x+w/2-lx-lw/2) <= frame.shape[1]*.025
                      and abs(y+h/2-ly-lh/2) <= frame.shape[0]*.025
                      and .85 <= w/lw <= 1.18 and .85 <= h/lh <= 1.18)
        self.stable_frames = self.stable_frames+1 if stable else 1
        if self.last_box is not None and not self.seeded and not self.pitch_changed and self.seen_at is not None:
            dt = now-self.seen_at
            if 1/120 <= dt <= .5:
                lx,ly,lw,lh = self.last_box
                measured = ((x+w/2-lx-lw/2)/dt,(y+h/2-ly-lh/2)/dt)
                self.velocity = tuple(float(np.clip(.8*m+.2*v,-frame.shape[1]*3,frame.shape[1]*3))
                                      for m,v in zip(measured,self.velocity))
            else:
                self.velocity = (0.0,0.0)
        else:
            self.velocity = (0.0,0.0)
        self.seen_at = now
        self.last_box = self.box
        self.seeded = False
        self.pitch_changed = False
        self.missing = 0; self.frames += 1
        # Learn from the interior only AFTER a shape/location candidate passed.
        # Later changes are gradual and only occur while geometry is stable.
        inset = max(1,round(min(w,h)*.2))
        interior = hsv[y+inset:y+h-inset,x+inset:x+w-inset]
        if interior.size:
            pixels = interior[(interior[:,:,0] >= 24)&(interior[:,:,0] <= 85)
                              &(interior[:,:,1] >= 12)&(interior[:,:,2] >= 60)]
            if len(pixels) >= 20:
                measured = float(np.median(pixels[:,0]))
                if not self.color_learned:
                    self.hue = round(measured)
                    self.color_learned = True
                elif self.stable_frames >= 3:
                    self.hue = round(.8*self.hue+.2*measured)
        self.reason = 'partial ball arc' if self.partial else 'visible region'
        self.reason += '; confirming' if self.frames < 5 else ' confirmed'
        if self.stable_frames >= 5:
            self.searching = False
        x,y,w,h = self.box
        self.edges = [name for name,condition in [('left',x<=2),('top',y<=2),
                     ('right',x+w>=frame.shape[1]-2),('bottom',y+h>=frame.shape[0]-2)] if condition]
        return self.box


def run():
    settings = camera_settings()
    tracker = PatchTracker()
    print('BELLY ONLY. No servo, GPIO or robot serial. Move BALL by hand slowly.')
    print('COLOR mode: GREEN / GREEN-WHITE / FLUORESCENT-GREEN ball only; partial ball allowed.')
    print('Shape/location candidate first, then interior color sampling. S=select green ball region.')
    print('Local target lock enabled; pale-yellow acquisition / nearby-white recovery. R explicitly unlocks.')
    print('S=freeze image, drag region then ENTER; R=reset; K=save right-foot position; Q=quit.')
    with ExitStack() as stack:
        def close_windows():
            try: cv2.destroyAllWindows()
            except cv2.error: pass
        stack.callback(close_windows)
        camera = RobotEye(**settings['belly']); stack.callback(camera.close)
        while True:
            ok,frame = camera.getImage()
            if not ok: raise RuntimeError('Belly frame failed; stopping')
            box = tracker.update(frame)
            display = frame.copy()
            if box:
                x,y,w,h = box
                cv2.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
            status = ('PARTIAL '+','.join(tracker.edges) if box and tracker.edges else 'TRACKING' if box else 'LOST / AMBIGUOUS')
            status += f' color-area={tracker.score:.0%} (not confidence)'
            cv2.putText(display,status,(8,25),0,.65,(0,255,0) if box else (0,0,255),2)
            cv2.putText(display,f'stable={tracker.stable_frames}/5 '+tracker.reason,(8,48),0,.45,(255,255,255),1)
            cv2.imshow('belly',display)
            cv2.imshow('mask',tracker.mask)
            key = cv2.waitKey(1)&255
            if 65 <= key <= 90: key += 32
            if key in (ord('q'),27): return True
            if key == ord('s'):
                selection = cv2.selectROI('belly',frame,True,False)
                if selection[2] == 0 or selection[3] == 0: continue
                tracker = PatchTracker()
                try:
                    tracker.select(frame,selection)
                    print('Sampled ball color:',tracker.color_profile())
                except ValueError as error: print('Selection rejected:',error)
                camera.discard_frames()
            if key == ord('r'):
                tracker = PatchTracker()
            if key == ord('k'):
                if box is None or tracker.stable_frames < 5:
                    print('Not saved:',tracker.reason,'; stable frames=',tracker.stable_frames,'/5')
                    continue
                # Store visible appearance, not an invented off-screen sphere center.
                stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                path = ROOT/'config/right_foot_reference.json'
                image_path = ROOT/'config'/('right-foot-'+stamp+'.png')
                x,y,w,h = box
                if not cv2.imwrite(str(image_path),frame[y:y+h,x:x+w]):
                    raise RuntimeError('Could not save reference patch')
                saved = dict(version=2,method='yellow_green_visible_region',camera=settings['belly'],shape=list(frame_shape),
                             visible_patch_box=[x/frame.shape[1],y/frame.shape[0],w/frame.shape[1],h/frame.shape[0]],
                             patch_file=image_path.name,color=tracker.color_profile(),color_fill=tracker.score,
                             clipped_edges=tracker.edges,position_definition='bounding box of VISIBLE color region; not full ball center',
                             confirmation='operator placed ball at physically tested right-foot position',
                             automatic_kick_enabled=False)
                if path.exists(): shutil.copy2(path,path.with_name(path.stem+'-'+stamp+'.bak.json'))
                temporary = path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(saved,indent=2)+'\n',encoding='utf-8')
                temporary.replace(path)
                print('Saved:',path,'. Reference only; automatic kick not enabled.')


if __name__ == '__main__':
    raise SystemExit(0 if run() else 1)
