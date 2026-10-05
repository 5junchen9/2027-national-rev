"""每次决定一个已有身体动作；调用者负责读新画面和发送动作。"""
import time

MIN_BALL_SCORE = .55
MIN_PARTIAL_SCORE = .40  # 已通过圆弧筛选的半球使用独立填充阈值。

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


class WalkKick:
    """可靠目标驱动动作；丢球先重找、再低头、最后有限前进。"""
    def __init__(self):
        self.goal_aligned = False
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

    def reset_observation(self):
        self.pending = None
        self.confirm_frames = 0

    def reset_search(self):
        self.lost_frames = 0
        self.lost_since = None
        self.search_stage = 'TRACK'
        self.search_head_moves = 0
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
        self.reason = 'visual search; waiting for stable target'
        if now-self.search_since < 1.0:
            return 'WAIT'
        if self.search_head_moves < 3:
            self.reason = 'visual search failed; lower head by 3 units'
            return 'LOWER_HEAD'
        if ambiguous:
            self.reason = 'multiple targets after head search; no blind walk'
            return 'STOP'
        self.reason = 'head search exhausted; no body search actions'
        return 'STOP'

    def mark_head_search(self, moved, now=None):
        # 已到机械限位时跳过剩余低头尝试，仍先等视觉重新确认。
        self.search_head_moves = self.search_head_moves+1 if moved else 3
        self.search_since = time.monotonic() if now is None else now

    def decide(self, phase, head_box, belly_box, head_stable, belly_stable,
               shape, flip, now=None, head_score=1.0,
               belly_score=1.0, ambiguous=False, goal=None, goal_stable=0, head_flip='0',
               head_partial=False, belly_partial=False):
        now = time.monotonic() if now is None else now
        self.goal_aligned = (goal is not None and goal_stable >= 5
                             and abs(goal.cx-.5) <= max(.025,min(.08,goal.width*.22)))
        if self.actions >= 30 or now-self.started_at >= 120:
            self.reason = '30 actions / 120 seconds limit'
            return 'STOP'

        # 这是颜色区域形状评分，不是经过标定的识别概率。
        if head_score < (MIN_PARTIAL_SCORE if head_partial else MIN_BALL_SCORE): head_box = None
        if belly_score < (MIN_PARTIAL_SCORE if belly_partial else MIN_BALL_SCORE): belly_box = None
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
                    return self.search_action(now)
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
            # 腹部接管后：头部看门，腹部看球；两个相机各自按中心对齐。
            # 这是二维近似，不把不同相机的像素坐标直接相减。
            if goal is None or goal_stable < 5:
                self.reset_observation()
                self.reason = 'goal missing or unstable; hold body'
                return 'WAIT'
            goal_error = goal.cx-.5
            tolerance = max(.025,min(.08,goal.width*.22))
            if abs(goal_error) > tolerance:
                action = 'TURN_RIGHT' if goal_error > 0 else 'TURN_LEFT'
                if head_flip in ('1', '-1'):
                    action = 'TURN_LEFT' if action == 'TURN_RIGHT' else 'TURN_RIGHT'
                self.reason = 'turn toward goal center'
            else:
                x,y,width,height = belly_box
                ball_error = (x+width/2)/shape[1]-.5
                if abs(ball_error) > .08:
                    action = 'SIDE_LEFT' if ball_error < 0 else 'SIDE_RIGHT'
                    self.reason = 'goal centered; move sideways to align ball'
                else:
                    action = 'UP_LITTLE'
                    self.reason = 'goal and ball centered; walk ball toward goal'

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
