"""每次决定一个已有身体动作；调用者负责读新画面和发送动作。"""
import time

MIN_BALL_SCORE = .55

def enlarged_reference(reference):
    """以原框中心为基准，宽高各增加3%；画面外部分裁掉。"""
    result = dict(reference)
    x, y, width, height = reference['visible_patch_box']
    left = max(0.0, x-width*.015)
    top = max(0.0, y-height*.015)
    right = min(1.0, x+width+width*.015)
    bottom = min(1.0, y+height+height*.015)
    result['visible_patch_box'] = [left, top, right-left, bottom-top]
    return result


class BallDeparture:
    """只比较同一腹部相机、同一头位的动作前后画面；不是进球判定。"""
    def __init__(self, flip='none'):
        self.flip = flip
        self.reset()

    def reset(self):
        self.reference = None
        self.frames = 0
        self.since = None

    def arm(self, box, shape, angle, now=None):
        x,y,w,h = box
        cy = (y+h/2)/shape[0]
        if self.flip in ('0','-1'): cy = 1-cy
        self.reference = (w,h,cy,angle)
        self.started_at = time.monotonic() if now is None else now
        self.frames = 0
        self.since = None

    def observe(self, box, stable_frames, shape, angle, now=None):
        if self.reference is None: return None
        now = time.monotonic() if now is None else now
        width,height,previous_y,previous_angle = self.reference
        if angle != previous_angle or now-self.started_at > 3:
            self.reset()
            return None
        moved_away = False
        if box is not None and stable_frames >= 3:
            x,y,w,h = box
            cy = (y+h/2)/shape[0]
            if self.flip in ('0','-1'): cy = 1-cy
            moved_away = (w <= width*.85 and w*h <= width*height*.65
                          and previous_y-cy >= .06)
        if moved_away:
            if self.since is None: self.since = now
            self.frames += 1
            if self.frames >= 5 and now-self.since >= .3:
                return 'DONE'
            return 'WAIT'
        self.frames = 0
        self.since = None
        # 每次近处前进后先观察至少一秒；丢球不会被判为完成。
        return 'WAIT' if now-self.started_at < 1 else None


class WalkKick:
    """可靠目标驱动动作；丢球只用头部反复上下搜索。"""
    def __init__(self):
        self.blind_steps = 0
        self.actions = 0
        self.pending = None
        self.confirm_frames = 0
        self.started_at = time.monotonic()
        self.reason = 'waiting for a stable ball'
        self.lost_frames = 0
        self.lost_since = None
        self.search_stage = 'TRACK'
        self.search_since = 0.0
        self.search_head_moves = 0
        self.search_head_direction = 1

    def reset_observation(self):
        self.pending = None
        self.confirm_frames = 0

    def reset_search(self):
        self.lost_frames = 0
        self.lost_since = None
        self.search_stage = 'TRACK'
        self.search_head_moves = 0
        self.search_head_direction = 1
        self.blind_steps = 0

    def search_action(self, now, ambiguous=False):
        self.reset_observation()
        self.lost_frames += 1
        if self.lost_since is None:
            self.lost_since = now
        if self.search_stage == 'TRACK':
            self.reason = 'target unreliable; holding body'
            if self.lost_frames < 3 or now-self.lost_since < .3:
                return 'WAIT'
            self.search_stage = 'VISUAL'
            self.search_since = now
            self.reason = 'visual reacquisition; hold and observe'
            return 'REACQUIRE'
        self.reason = 'head search; observe each pose for 1.5 seconds'
        if now-self.search_since < 1.5:
            return 'WAIT'
        if self.search_head_direction > 0:
            self.reason = 'ball lost; lower head by 3 units'
            return 'LOWER_HEAD'
        self.reason = 'ball lost; raise head by 3 units'
        return 'RAISE_HEAD'

    def mark_head_search(self, moved, now=None):
        # 每四次反向；到舵机指令边界也反向，避免一直顶着限位找。
        self.search_head_moves += 1
        if not moved or self.search_head_moves >= 4:
            self.search_head_direction *= -1
            self.search_head_moves = 0
        self.search_since = time.monotonic() if now is None else now

    def decide(self, phase, head_box, belly_box, head_stable, belly_stable,
               shape, flip, now=None, head_score=1.0,
               belly_score=1.0, ambiguous=False):
        now = time.monotonic() if now is None else now
        if self.actions >= 30:
            self.reason = '30 body actions limit'
            return 'STOP'

        # 这是颜色区域形状评分，不是经过标定的识别概率。
        if head_score < MIN_BALL_SCORE: head_box = None
        if belly_score < MIN_BALL_SCORE: belly_box = None
        if ambiguous:
            return self.search_action(now,ambiguous=True)

        if phase == 'HEAD':
            if belly_box is not None:
                self.reset_observation()
                if belly_stable < 5:
                    self.reset_search()
                    self.reason = 'belly sees ball; wait for handover confirmation'
                    return 'WAIT'
                self.reason = 'waiting for belly handover'
                return 'WAIT'
            if head_box is not None:
                if head_stable < 5:
                    self.reset_search()
                    self.reset_observation()
                    self.reason = 'head ball visible but moving; wait for confirmation'
                    return 'WAIT'
                self.reset_search()
                x, y, width, height = head_box
                error_x = (x+width/2)/shape[1]-.5
                action = 'SIDE_LEFT' if error_x < -.08 else 'SIDE_RIGHT' if error_x > .08 else 'UP_LITTLE'
                self.reason = 'head ball: align then approach'
            else:
                return self.search_action(now)
        else:
            if belly_box is None:
                action = self.search_action(now)
                return action
            if belly_stable < 5:
                self.reset_search()
                self.reset_observation()
                self.reason = 'belly ball visible but moving; hold body'
                return 'WAIT'
            self.reset_search()
            # 腹部接管后只根据球的位置横向对齐，再向前行走带球。
            x,y,width,height = belly_box
            ball_error = (x+width/2)/shape[1]-.5
            if abs(ball_error) > .08:
                action = 'SIDE_LEFT' if ball_error < 0 else 'SIDE_RIGHT'
                self.reason = 'belly ball: move sideways to align'
            else:
                action = 'UP_LITTLE'
                self.reason = 'belly ball centered; walk forward'

        if flip in ('1', '-1') and action.startswith('SIDE_'):
            action = 'SIDE_RIGHT' if action == 'SIDE_LEFT' else 'SIDE_LEFT'
        # 按当前实机反馈交换横移动作；前进及转向不变。
        if action == 'SIDE_LEFT':
            action = 'SIDE_RIGHT'
        elif action == 'SIDE_RIGHT':
            action = 'SIDE_LEFT'
        self.confirm_frames = self.confirm_frames+1 if action == self.pending else 1
        self.pending = action
        return action if self.confirm_frames >= 5 else 'WAIT'

    def mark_sent(self, action, blind=False):
        self.actions += 1
        if blind:
            self.blind_steps += 1
            self.search_since = time.monotonic()
        self.reset_observation()
