"""比赛路标和蓝地判定，不连接硬件。"""
import cv2
import threading
import time


class RouteVision:
    """后台读双摄并解码，行走期间也扫描；线程从不发送身体或头部动作。"""
    def __init__(self, head_eye, belly_eye, decode, confirm_frames):
        self.head_eye, self.belly_eye = head_eye, belly_eye
        self.decode = decode
        self.confirm_frames = confirm_frames
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.sequence = self.delivered = 0
        self.expected = self.camera = None
        self.frames = 0
        self.reached = False
        self.error = None
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def watch(self, expected, camera):
        with self.condition:
            self.expected, self.camera = expected, camera
            self.frames = 0
            self.reached = False

    def saw_target(self):
        with self.condition:
            return self.reached

    def discard(self):
        with self.condition:
            self.delivered = self.sequence

    def capture(self):
        try:
            while not self.stopped.is_set():
                okh, head = self.head_eye.getImage()
                okb, belly = self.belly_eye.getImage()
                if not okh or not okb:
                    raise RuntimeError('比赛双摄读取失败')
                codes = {'head': self.decode(head), 'belly': self.decode(belly)}
                with self.condition:
                    if self.expected is not None:
                        contents = [text for text, _ in codes[self.camera]]
                        self.frames = self.frames+1 if contents.count(self.expected) == 1 else 0
                        if self.frames >= self.confirm_frames:
                            # 记住动作期间扫到的路标，不要求动作结束后仍在画面内。
                            self.reached = True
                    self.latest = (head, belly, codes)
                    self.received_at = time.monotonic()
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def read(self):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > self.delivered or self.error is not None, timeout=2)
            if self.error is not None:
                raise RuntimeError('后台比赛视觉停止') from self.error
            if not ready or time.monotonic()-self.received_at > 1:
                raise RuntimeError('比赛画面过期，不再动作')
            self.delivered = self.sequence
            return self.latest

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('比赛视觉线程未能及时停止')


class QRStepPlanner:
    """当前目标码出现后先停下，连续确认后切换；每步之后重新计数。"""
    def __init__(self, expected, confirm_frames, max_steps, deadline):
        self.expected = expected
        self.confirm_frames = confirm_frames
        self.max_steps = max_steps
        self.deadline = deadline
        self.frames = 0
        self.steps = 0

    def decide(self, contents, now, ready=True):
        if now >= self.deadline:
            return "STOP"
        if not ready:
            self.frames = 0
            return "WAIT"
        matches = contents.count(self.expected)
        if matches == 1:
            self.frames += 1
            return "DONE" if self.frames >= self.confirm_frames else "WAIT"
        self.frames = 0
        if matches > 1:
            return "WAIT"
        return "STOP" if self.steps >= self.max_steps else "UP_LITTLE"

    def mark_step(self):
        self.steps += 1
        self.frames = 0


def blue_ratios(frame, near_roi, far_roi):
    """分别检查近、远地面区域，拒绝单个小蓝方块。ROI是归一化坐标。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (95, 80, 50), (130, 255, 255))
    height, width = mask.shape
    ratios = []
    for left, top, right, bottom in (near_roi, far_roi):
        region = mask[round(top * height):round(bottom * height),
                      round(left * width):round(right * width)]
        ratios.append(cv2.countNonZero(region) / region.size)
    return ratios


class BlueEntry:
    def __init__(self, min_ratio, confirm_frames):
        self.min_ratio = min_ratio
        self.confirm_frames = confirm_frames
        self.frames = 0

    def update(self, ratios):
        self.frames = self.frames + 1 if all(r >= self.min_ratio for r in ratios) else 0
        return self.frames >= self.confirm_frames


def run_course(io, color):
    """比赛的唯一顺序；io负责真实观察/动作，模拟也执行同一份顺序。"""
    settings = io.settings
    io.phase("寻找face")
    io.scan_until(settings["face_qr"], "belly")
    io.phase("人脸前小步左转")
    io.left(1)
    io.phase("姓名和性别识别")
    io.identity()
    io.phase("人脸完成后小步右转回路线")
    io.right(1)
    io.phase("寻找sber")
    io.scan_until(settings["factory_qr"], "belly")
    io.forward(settings["after_sber_steps"])
    io.right(settings["sber_right_actions"])
    io.phase("寻找action1")
    io.scan_until(settings["carry_qr"], "belly")
    io.right(settings["action1_right_actions"])
    io.phase("工厂入口直线接近")
    # 先走现场配置的小步，再由双摄搬运完成剩余接近距离。
    io.forward(settings["factory_entry_forward_steps"])
    io.phase("指定颜色搬运")
    io.carry(color, settings["drop_qr"])
    io.phase("返回赛道")
    io.right(settings["return_right_actions"][color])
    io.forward(settings["after_return_steps"])
    io.phase("足球")
    io.sport()
    io.right(settings["after_sport_right_actions"])
    io.phase("头部寻找dance")
    io.set_head(settings["dance_head_position"])
    io.scan_until(settings["dance_qr"], "head")
    io.phase("腹部确认蓝色舞区")
    io.enter_blue()
    io.phase("音乐和舞蹈")
    io.dance()
    io.phase("流程结束；实际任务得分须现场确认")
    return True
