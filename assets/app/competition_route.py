"""比赛路标和蓝地判定，不连接硬件。"""
import cv2
from concurrent.futures import ThreadPoolExecutor
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
        self.candidate_seen = False
        self.decode_count = 0
        self.missed_decodes = 0
        self.contents = []
        self.watch_number = 0
        self.error = None
        self.decoding = False
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def watch(self, expected, camera, confirm_frames=None):
        with self.condition:
            self.expected, self.camera = expected, camera
            self.watch_number += 1
            self.decoding = expected is not None
            if confirm_frames is not None:
                self.confirm_frames = confirm_frames
            self.frames = 0
            self.reached = False
            self.candidate_seen = False
            self.decode_count = 0
            self.missed_decodes = 0
            self.contents = []

    def qr_progress(self):
        """只统计已完成的独立解码，不把预览刷新当成新解码。"""
        with self.condition:
            return self.decode_count, self.frames, self.missed_decodes, list(self.contents)

    def saw_target(self):
        with self.condition:
            return self.reached

    def saw_candidate(self):
        with self.condition:
            return self.candidate_seen

    def discard(self):
        with self.condition:
            self.delivered = self.sequence

    def waiting_for_decode(self):
        with self.condition:
            return self.decoding

    def capture(self):
        # 只有一个解码任务；忙时持续读新画面，不排队旧帧。
        try:
            with ThreadPoolExecutor(max_workers=1) as worker:
                pending = None
                job = None
                while not self.stopped.is_set():
                    okh, head = self.head_eye.getImage()
                    okb, belly = self.belly_eye.getImage()
                    if not okh or not okb:
                        raise RuntimeError('比赛双摄读取失败')
                    captured_at = time.monotonic()
                    with self.condition:
                        expected, camera = self.expected, self.camera
                        watch_number = self.watch_number
                    codes = {'head': [], 'belly': []}
                    if pending is None and expected is not None:
                        job = (expected, camera, watch_number)
                        image = head if camera == 'head' else belly
                        pending = worker.submit(self.decode,image.copy())
                    if pending is not None and pending.done():
                        decoded = pending.result()
                        pending = None
                        with self.condition:
                            if job == (self.expected,self.camera,self.watch_number):
                                codes[job[1]] = decoded
                                contents = [text for text, _ in decoded]
                                matched = contents.count(job[0]) == 1
                                self.decode_count += 1
                                self.contents = contents
                                self.frames = self.frames+1 if matched else 0
                                self.missed_decodes = 0 if matched else self.missed_decodes+1
                                if matched:
                                    self.candidate_seen = True
                                if self.frames >= self.confirm_frames:
                                    self.reached = True
                    with self.condition:
                        self.decoding = self.expected is not None and (
                            pending is not None or job != (self.expected,self.camera,self.watch_number))
                        self.latest = (head, belly, codes)
                        self.received_at = captured_at
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
    io.phase("face确认后后退一步")
    io.backward(1)
    io.phase("人脸前小步右转")
    io.right(1)
    io.phase("下蹲对准人脸立牌")
    io.squat()
    io.phase("姓名和性别识别")
    io.identity()
    io.phase("识别完成后站起")
    io.stand()
    io.phase("人脸完成后小步左转回路线")
    io.left(1)
    io.phase("寻找left")
    io.scan_until(settings["factory_qr"], "belly")
    io.right(settings["left_right_actions"])
    io.phase("寻找action1")
    io.scan_until(settings["carry_qr"], "belly")
    io.right(settings["action1_right_actions"])
    io.phase("指定颜色搬运")
    carry_result = io.carry(color, settings["drop_qr"])
    if carry_result != "scan_limit":
        io.phase("action2投放完成，开始足球")
        io.phase("足球区内侧边线对齐")
        io.align_football()
        io.sport()
        io.right(settings["after_sport_right_actions"])
    io.phase("头部寻找dance")
    io.set_head(settings["dance_head_position"])
    io.scan_until(settings["dance_qr"], "head", confirm_frames=2)
    io.phase("腹部确认蓝色舞区")
    io.enter_blue()
    io.phase("音乐和舞蹈")
    io.dance()
    io.phase("流程结束；实际任务得分须现场确认")
    return True
