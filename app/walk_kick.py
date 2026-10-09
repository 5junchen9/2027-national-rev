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
    """腹部前进一步后，连续未检测到球则结束；不判断实际进球。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.angle = None
        self.frames = 0
        self.since = None

    def arm(self, angle, now=None):
        self.angle = angle
        self.started_at = time.monotonic() if now is None else now
        self.frames = 0
        self.since = None

    def observe(self, box, angle, now=None):
        if self.angle is None:
            return None
        now = time.monotonic() if now is None else now
        if angle != self.angle or now-self.started_at > 3:
            self.reset()
            return None
        if box is None:
            if self.since is None:
                self.since = now
            self.frames += 1
            if self.frames >= 5 and now-self.since >= .3:
                return 'DONE'
            return 'WAIT'
        self.frames = 0
        self.since = None
        # 前进后至少观察一秒，球仍可见再交回对齐和前进。
        return 'WAIT' if now-self.started_at < 1 else None


class WalkKick:
    """稳定球驱动左右转和前进；丢球只用头部反复上下搜索。"""
    def __init__(self, fixed_head=False):
        self.fixed_head = fixed_head
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
        if self.fixed_head:
            self.reason = 'ball lost; head fixed at 129, hold body and observe'
            return 'WAIT'
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
                action = 'TURN_LEFT' if error_x < -.08 else 'TURN_RIGHT' if error_x > .08 else 'UP_LITTLE'
                self.reason = 'head ball: turn toward ball, then approach'
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
            # 独立调试的腹部阶段同样转向对球；比赛检测到腹部球后交回前进3步。
            x,y,width,height = belly_box
            ball_error = (x+width/2)/shape[1]-.5
            if abs(ball_error) > .08:
                action = 'TURN_LEFT' if ball_error < 0 else 'TURN_RIGHT'
                self.reason = 'belly ball: turn toward ball'
            else:
                action = 'UP_LITTLE'
                self.reason = 'belly ball centered; walk forward'

        # 沿用GitHub原dual_kick.KickPlanner的左右转及水平镜像映射。
        # 原代码转向看球门，这里按当前要求看球；不恢复球门识别。
        # https://github.com/5junchen9/2027-national-rev/blob/20be9378e1d3285162352732a19b360e2fcce7e6/app/dual_kick.py
        if flip in ('1', '-1') and action in ('TURN_LEFT', 'TURN_RIGHT'):
            action = 'TURN_RIGHT' if action == 'TURN_LEFT' else 'TURN_LEFT'
        self.confirm_frames = self.confirm_frames+1 if action == self.pending else 1
        self.pending = action
        return action if self.confirm_frames >= 5 else 'WAIT'

    def mark_sent(self, action, blind=False):
        self.actions += 1
        if blind:
            self.blind_steps += 1
            self.search_since = time.monotonic()
        self.reset_observation()
