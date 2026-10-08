"""独立比赛流程：检查、模拟、双摄预览及真实比赛。"""
import argparse
import importlib.util
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import time

import cv2
from robot_config import ROOT
from competition_route import QRStepPlanner, RouteVision, BlueEntry, blue_ratios, run_course
from carry_vision import find_blocks, steering
from route_line import correction_action
from competition_qr import read_codes
from dual_kick import camera_settings

CONFIG_FILE = ROOT / "config/competition.json"
BODY_PORT = "/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0"


def load_settings(path):
    settings = json.loads(path.read_text(encoding="utf-8"))
    # 只限制会直接造成不合理运动的参数。
    counts = [settings[key] for key in (
        "left_right_actions", "action1_right_actions", "delivery_search_right_actions", "delivery_right_actions",
        "after_sport_right_actions", "blue_extra_steps", "route_max_steps")]
    if any(type(count) is not int or not 0 <= count <= 30 for count in counts):
        raise ValueError("运动次数须为0至30的整数")
    if type(settings.get("carry_exit_right_actions",7)) is not int or not 1 <= settings.get("carry_exit_right_actions",7) <= 30:
        raise ValueError("搬运超限右转次数须为1至30的整数")
    if settings["delivery_right_actions"] > 10:
        raise ValueError("抱起后预转向最多10次，次数须现场测量")
    if not 0 < settings["total_seconds"] <= 480:
        raise ValueError("比赛总期限须在0至480秒之间")
    for key in ("face_head_position", "dance_head_position", "sport_head_position"):
        if not 85 <= settings[key] <= 180:
            raise ValueError(key + "超出头部指令范围")
    for key in ("qr_confirm_frames", "blue_confirm_frames"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(key + "须为正整数")
    if settings["route_timeout_seconds"] <= 0 or settings["observe_seconds"] < 0.5:
        raise ValueError("观察至少0.5秒，每段期限必须大于零")
    for key in ("blue_roi_near", "blue_roi_far"):
        left, top, right, bottom = settings[key]
        if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
            raise ValueError(key + "须为画面内有效区域")
    if not 0 < settings["blue_min_ratio"] <= 1:
        raise ValueError("蓝色比例须在0至1之间")
    for key in ("face_qr", "factory_qr", "carry_qr", "drop_qr", "dance_qr"):
        if not isinstance(settings[key], str) or not settings[key]:
            raise ValueError(key + "须为二维码完整内容")
    return settings


def legacy_root(settings):
    return (ROOT / settings["legacy_root"]).resolve()


def preflight(settings, use_original_drop=False):
    """仅查文件与标定，不启动摄像头、GPIO、串口或模型进程。"""
    from robotmove import ACTIONS
    from dreammaker_protocol import load_dzz, motion_frame, apply_offsets, DEFAULT_INITIAL_POSITIONS
    problems = []
    try:
        from pyzbar.pyzbar import decode
    except ImportError as error:
        problems.append("二维码解码依赖未就绪：" + str(error) + "；需要现有requirements中的pyzbar和系统libzbar0")
    legacy = legacy_root(settings)
    if not (importlib.util.find_spec("rapidocr_onnxruntime") or importlib.util.find_spec("rapidocr")):
        problems.append("缺少姓名OCR依赖；请在比赛venv安装 rapidocr==2.1.0 和 onnxruntime")
    if importlib.util.find_spec("onnxruntime") is None:
        problems.append("缺少OCR运行库 onnxruntime")
    required = [legacy / item for item in (
        "app/face_main.py", "hardware/face_detector_dnn.py", "hardware/name_ocr.py",
        "assets/models/gender/fairface.onnx",
        "assets/models/face_detector/opencv_face_detector_uint8.pb",
        "assets/models/face_detector/opencv_face_detector.pbtxt")]
    required += [ROOT / item for item in (
        "app/chinese_speech.py", "app/robot_audio.py",
        "assets/models/tts/matcha-icefall-zh-baker/model-steps-3.onnx",
        "assets/models/tts/matcha-icefall-zh-baker/vocos-22khz-univ.onnx",
        "assets/models/tts/matcha-icefall-zh-baker/lexicon.txt",
        "assets/models/tts/matcha-icefall-zh-baker/tokens.txt",
        "assets/models/tts/matcha-icefall-zh-baker/phone.fst",
        "assets/models/tts/matcha-icefall-zh-baker/date.fst",
        "assets/models/tts/matcha-icefall-zh-baker/number.fst")]
    required.append(ROOT / "assets/music/dance.mp3")
    for path in required:
        if not path.is_file():
            problems.append("缺少 " + str(path))
    for filename, _ in ACTIONS.values():
        if filename is None:
            continue
        path = ROOT / "assets/actions" / filename
        try:
            for offsets, milliseconds in load_dzz(path):
                motion_frame(apply_offsets(DEFAULT_INITIAL_POSITIONS, offsets), milliseconds)
        except (OSError, ValueError) as error:
            problems.append(str(error))
    settings_cameras = camera_settings()
    try:
        reference = json.loads((ROOT / "config/carry_dual_reference.json").read_text())
        from carry_vision import reference_for_target
        reference = reference_for_target(reference,settings["drop_qr"])
        if (reference.get("version") not in (2, 3) or reference.get("cameras") != settings_cameras
                or reference.get("target_qr") != settings["drop_qr"]
                or "pickup" not in reference):
            problems.append("双摄搬运标定缺失或视图/目的地不匹配，请先恢复原相机配置并核对H标定")
    except (OSError, ValueError) as error:
        problems.append("搬运标定：" + str(error))
    try:
        from handover_debug import validate
        head = json.loads((ROOT / "config/head_calibration.json").read_text())
        foot = json.loads((ROOT / "config/right_foot_reference.json").read_text())
        head["forward"] = settings["sport_head_position"]
        validate(head, foot, settings_cameras)
    except (OSError, ValueError) as error:
        problems.append("足球标定：" + str(error))
    for problem in problems:
        print("[未就绪]", problem)
    print("[说明] 各段右转和回程次数必须现场测量；blue比例不能证明整机已进入舞区。")
    print("[说明] 足球沿用行走带球，连续确认球远离后退出；不证明进球。")
    return not problems


class GuardedRobot:
    """复用唯一身体连接，发送前检查整场期限。"""
    def __init__(self, robot, deadline):
        self.robot = robot
        self.deadline = deadline

    def robotMove(self, action):
        from robotmove import ACTIONS, ACTION_DIR, STRAIGHT_ACTIONS, WALK_DURATION_SCALE, ACTION_INTERVAL_SECONDS
        from dreammaker_protocol import load_dzz, scaled_duration_ms, MOTION_EXTRA_SECONDS
        filename, repeats = ACTIONS[action]
        scale = WALK_DURATION_SCALE if action in STRAIGHT_ACTIONS else None
        if action == "HOLD_BOX":
            scale = 1.0
        duration = 0.04 if filename is None else repeats * (
            sum(scaled_duration_ms(ms, scale) / 1000 + max(0, MOTION_EXTRA_SECONDS)
                for _, ms in load_dzz(Path(ACTION_DIR) / filename))
            + max(0, ACTION_INTERVAL_SECONDS))
        if time.monotonic() + duration >= self.deadline:
            raise RuntimeError("剩余比赛时间不足以执行下一动作")
        self.robot.robotMove(action)


class CompetitionIO:
    def __init__(self, settings, robot, deadline, use_original_drop=False):
        self.settings = settings
        self.robot = robot
        self.deadline = deadline
        self.use_original_drop = use_original_drop
        self.views = None
        self.stream = None
        self.phase_name = "启动"
        self.identity_models = None

    def prepare_identity(self):
        from competition_identity import load_models
        started = time.monotonic()
        print("[人脸预加载] 开始加载检测、性别和OCR模型。", flush=True)
        self.identity_models = load_models(legacy_root(self.settings))
        print(f"[人脸预加载] 完成，耗时={time.monotonic()-started:.2f}s。", flush=True)

    def close_identity_models(self):
        models, self.identity_models = self.identity_models, None
        if models is not None:
            for model in models:
                model.close()

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise RuntimeError("比赛总期限到达")

    def phase(self, name):
        self.check_time()
        self.phase_name = name
        print("[比赛阶段]", name, flush=True)

    def open_views(self, head_position=129):
        if self.views is not None:
            return
        from roboteye import RobotEye
        from Head import RobotHeadServoOnly
        self.views = ExitStack()
        try:
            cameras = camera_settings()
            self.head_eye = RobotEye(**cameras["head"], latest=True)
            self.views.callback(self.head_eye.close)
            self.belly_eye = RobotEye(**cameras["belly"], latest=True)
            self.views.callback(self.belly_eye.close)
            self.servo = RobotHeadServoOnly(hold=True)
            self.views.callback(self.servo.cleanup)
            self.views.callback(cv2.destroyAllWindows)
            self.servo.turn_vertical(head_position)
            self.views.callback(self.stop_route)
            self.start_route()
            self.view_shapes = None
        except BaseException:
            self.close_views()
            raise

    def close_views(self):
        if self.views is not None:
            views, self.views = self.views, None
            views.close()
            self.stream = None

    def stop_route(self):
        """先停止路线读图线程，再将相机交给下一阶段；保持相机打开。"""
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def start_route(self):
        self.stream = RouteVision(self.head_eye, self.belly_eye, read_codes,
                                  self.settings["qr_confirm_frames"])
        self.flush()

    def resume_route(self, position=129):
        self.servo.turn_vertical(position)
        self.head_eye.discard_frames(1)
        self.belly_eye.discard_frames(1)
        self.start_route()

    def set_head(self, position):
        self.open_views()
        self.servo.turn_vertical(position)
        self.flush()

    def flush(self):
        # 保留后台已确认的路标，只要求主流程下一次读取新画面。
        self.stream.discard()
        self.ready_at = time.monotonic() + self.settings["observe_seconds"]

    def observe(self):
        self.check_time()
        self.open_views()
        head, belly, codes = self.stream.read()
        shapes = (head.shape[:2], belly.shape[:2])
        if self.view_shapes is not None and self.view_shapes != shapes:
            raise RuntimeError("相机尺寸改变，停止身体动作")
        self.view_shapes = shapes
        for name, frame in (("head", head), ("belly", belly)):
            display = frame.copy()
            for content, box in codes[name]:
                cv2.rectangle(display, (int(box.x), int(box.y)),
                              (int(box.x + box.width), int(box.bottom)), (0, 255, 0), 2)
                cv2.putText(display, content, (int(box.x), max(20, int(box.y))), 0, .6, (0, 255, 0), 1)
            cv2.imshow("competition " + name, display)
        if cv2.waitKey(1) & 255 in (ord("q"), 27):
            raise KeyboardInterrupt
        return head, belly, codes

    def move(self, action):
        self.check_time()
        self.robot.robotMove(action)
        if self.views is not None:
            self.flush()

    def forward(self, count):
        for _ in range(count):
            # 固定补偿段也读取当前摄像头，掉线时不继续。
            self.observe_ready()
            self.move("UP_LITTLE")

    def squat(self):
        self.observe_ready()
        self.move("SQUAT")

    def stand(self):
        self.observe_ready()
        self.move("STAND")

    def backward(self, count):
        for _ in range(count):
            # 每次后退前确认双摄仍在更新，动作后继续观察再执行下一步。
            self.observe_ready()
            self.move("BACK")

    def right(self, count):
        for _ in range(count):
            self.observe_ready()
            self.move("TURN_RIGHT")

    def left(self, count):
        for _ in range(count):
            self.observe_ready()
            self.move("TURN_LEFT")

    def observe_ready(self):
        """动作结束后的观察时间内继续刷新预览，再允许下一个固定动作。"""
        while True:
            frames = self.observe()
            if time.monotonic() >= self.ready_at:
                return frames

    def scan_until(self, content, camera, confirm_frames=None):
        self.open_views()
        confirm_frames = self.settings["qr_confirm_frames"] if confirm_frames is None else confirm_frames
        planner = QRStepPlanner(content, confirm_frames,
                                self.settings["route_max_steps"],
                                min(self.deadline, time.monotonic() + self.settings["route_timeout_seconds"]))
        print("寻找完整二维码：", content, "相机：", camera, flush=True)
        candidate_paused = False
        last_decode = 0
        line_checked_at_step = 0
        if self.stream is not None:
            self.stream.watch(content, camera, confirm_frames=confirm_frames)
        try:
            while True:
                _, belly, codes = self.observe()
                ready = time.monotonic() >= self.ready_at
                # 有后台时只使用它的独立解码计数，预览的空结果不代表漏读。
                contents = [text for text, _ in codes[camera]] if self.stream is None else []
                action = planner.decide(contents, time.monotonic(), ready=ready)
                target_seen = self.stream.saw_target() if self.stream is not None else False
                if self.stream is not None:
                    decoded, frames, misses, texts = self.stream.qr_progress()
                    if decoded != last_decode:
                        last_decode = decoded
                        print(f"[扫码] {content} 连续={frames}/{confirm_frames}；"
                              f"本次读到={texts}；连续未匹配={misses}", flush=True)
                    if self.stream.saw_candidate() and not target_seen and misses >= 5:
                        raise RuntimeError("曾读到" + content +
                                           "，随后连续5次解码未匹配；停止前进，请检查完整二维码是否仍在画面内")
                if (time.monotonic() < planner.deadline and ready and self.stream is not None
                        and target_seen):
                    action = "DONE"
                elif (action in ("UP_LITTLE", "STOP") and time.monotonic() < planner.deadline
                      and self.stream is not None
                      and self.stream.waiting_for_decode()):
                    action = "WAIT"
                elif (action in ("UP_LITTLE", "STOP") and time.monotonic() < planner.deadline
                      and self.stream is not None and self.stream.saw_candidate()):
                    # 行走期间首次读到目标后，原地等待完整确认，不继续跨过路标。
                    action = "WAIT"
                    if not candidate_paused:
                        print("已读到目标二维码，暂停前进并等待连续确认：", content, flush=True)
                        candidate_paused = True
                if action == "DONE":
                    print("二维码连续确认：", content, flush=True)
                    return
                if action == "STOP":
                    raise RuntimeError("寻找" + content + "超出步数或期限")
                if action == "UP_LITTLE":
                    # 每前进3小步检查一次；一次最多纠正一个动作，之后恢复扫码。
                    if planner.steps and planner.steps % 3 == 0 and line_checked_at_step != planner.steps:
                        line_checked_at_step = planner.steps
                        if self.correct_line(belly, camera):
                            continue
                    planner.mark_step()
                    self.move(action)
        finally:
            if self.stream is not None:
                self.stream.watch(None, None)

    def correct_line(self, belly, camera):
        # 只在腹部扫码的普通路段使用；头部找dance时不纠偏。
        if camera != "belly":
            return False
        action = correction_action(belly)
        if action is None:
            return False
        _, next_belly, codes = self.observe()
        # 二维码优先；两幅新画面给出相同方向才纠正。
        if self.stream is not None and self.stream.saw_candidate():
            return True  # 本轮不前进，回到扫码确认。
        if correction_action(next_belly) != action:
            return False
        flip = camera_settings()["belly"]["flip"]
        if action.startswith("SIDE_"):
            action = steering(action, flip)
        elif flip in ("1", "-1"):
            action = "TURN_RIGHT" if action == "TURN_LEFT" else "TURN_LEFT"
        print("[路线纠偏]", action, flush=True)
        self.move(action)
        return True

    def identity(self):
        self.stop_route()
        from competition_identity import recognize
        try:
            if not recognize(legacy_root(self.settings), self.settings["face_head_position"],
                             min(60, self.deadline-time.monotonic()),
                             eye=self.head_eye, head=self.servo, models=self.identity_models):
                print("姓名和性别未完成，跳过播报并继续下一项。", flush=True)
        finally:
            self.close_identity_models()
        self.check_time()
        self.resume_route()

    def carry(self, color, target):
        self.stop_route()
        from carry_vision import run
        result = run(color, target, actions=True, robot=self.robot,
                   search_right_actions=self.settings["delivery_search_right_actions"], deadline=self.deadline,
                   right_scan=True, exit_right_actions=self.settings.get("carry_exit_right_actions",7),
                   qr_reader=read_codes, use_original_drop=self.use_original_drop,
                   eyes=(self.head_eye, self.belly_eye), servo=self.servo)
        if not result:
            raise RuntimeError("搬运未完成，停止比赛")
        next_head = "dance_head_position" if result == "scan_limit" else "sport_head_position"
        self.resume_route(self.settings[next_head])
        return result

    def align_football(self):
        from football_align import run
        run(self)

    def sport(self):
        self.stop_route()
        from handover_debug import run
        if not run(forward=self.settings["sport_head_position"],
                   fps=camera_settings()["head"]["fps"], actions=True, robot=self.robot,
                   deadline=self.deadline, eyes=(self.head_eye, self.belly_eye), servo=self.servo):
            raise RuntimeError("足球阶段未完成，停止比赛")
        self.resume_route()

    def enter_blue(self):
        entry = BlueEntry(self.settings["blue_min_ratio"], self.settings["blue_confirm_frames"])
        steps = 0
        deadline = min(self.deadline, time.monotonic() + self.settings["route_timeout_seconds"])
        while time.monotonic() < deadline:
            _, belly, _ = self.observe()
            if time.monotonic() < self.ready_at:
                entry.frames = 0
                continue
            ratios = blue_ratios(belly, self.settings["blue_roi_near"], self.settings["blue_roi_far"])
            if entry.update(ratios):
                self.forward(self.settings["blue_extra_steps"])
                return
            if entry.frames:
                continue  # 有蓝地候选，先停下来确认。
            if steps >= self.settings["route_max_steps"]:
                break
            steps += 1
            self.move("UP_LITTLE")
        raise RuntimeError("舞区未连续确认，停止比赛")

    def dance(self):
        self.close_views()
        import dance_main
        from robot_audio import start
        music = start(dance_main.BACKGROUND_MUSIC)
        try:
            time.sleep(.2)
            if music.poll() is not None:
                raise RuntimeError("音乐未持续播放，不启动舞蹈")
            for action in dance_main.DANCE_ACTIONS:
                if music.poll() is not None:
                    raise RuntimeError("音乐已提前结束，停止后续舞蹈")
                self.move(action)
            if music.poll() is not None:
                raise RuntimeError("音乐在动作完成前结束，舞蹈未确认完成")
            self.check_time()
        finally:
            if music.poll() is None:
                music.terminate()
                try:
                    music.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    music.kill()
                    music.wait(timeout=3)


def simulate(settings, color):
    class SimulatedIO:
        def __init__(self):
            self.settings = settings
        def phase(self, name): print("[模拟阶段]", name)
        def scan_until(self, text, camera, confirm_frames=None):
            count = self.settings["qr_confirm_frames"] if confirm_frames is None else confirm_frames
            print("[模拟观测]", camera, "连续识别", text, count, "次")
        def identity(self): print("[模拟识别] 姓名/性别及播报成功；不检验模型")
        def squat(self): print("[模拟动作] SQUAT 单帧下蹲；保持蹲姿识别人脸")
        def stand(self): print("[模拟动作] STAND 识别后站起，再转向")
        def forward(self, count): print("[模拟动作] UP_LITTLE ×", count)
        def backward(self, count): print("[模拟动作] BACK ×", count, "；距离需现场确认")
        def right(self, count): print("[模拟动作] TURN_RIGHT ×", count, "；角度未验证")
        def left(self, count): print("[模拟动作] TURN_LEFT ×", count, "；转向，不是横移")
        def carry(self, color, target):
            print("[模拟搬运]", color, "HOLD_BOX → 抱物右平移最多10次并扫码", target,
                  "→ 腹部识别目标码就放下；超限原地放下并右转接dance，转角未实测")
            return True
        def align_football(self): print("[模拟对齐] 搜索足球区双层白框与中圈 → 转向 → 横移居中；不验证实际识别")
        def sport(self): print("[模拟足球] 行走带球 → 连续确认球远离 → 停止；进球未验证")
        def set_head(self, position): print("[模拟头部]", position)
        def enter_blue(self): print("[模拟舞区] 近远蓝地连续确认 → 补偿步数", settings["blue_extra_steps"])
        def dance(self): print("[模拟舞蹈] 音乐 + DANCE1/3/4/5")
    print("[模拟动作] STAND一次；不访问硬件")
    return run_course(SimulatedIO(), color)


def preview(settings, color):
    io = CompetitionIO(settings, None, time.monotonic() + settings["total_seconds"])
    try:
        io.open_views(settings["face_head_position"])
        print("双摄预览：仅头部定位，不打开身体串口；Q退出。")
        seen = set()
        while True:
            head, belly, codes = io.observe()
            for camera, items in codes.items():
                for text, _ in items:
                    if (camera, text) not in seen:
                        print("[二维码]", camera, repr(text));seen.add((camera, text))
            ratios = blue_ratios(belly, settings["blue_roi_near"], settings["blue_roi_far"])
            display = belly.copy()
            for key in ("blue_roi_near", "blue_roi_far"):
                left, top, right, bottom = settings[key]
                height, width = display.shape[:2]
                cv2.rectangle(display, (round(left*width), round(top*height)),
                              (round(right*width), round(bottom*height)), (255, 255, 0), 2)
            cv2.putText(display, f"blue near={ratios[0]:.0%} far={ratios[1]:.0%}", (8, 25), 0, .6, (255, 255, 255), 1)
            for box in find_blocks(belly, color):
                cv2.rectangle(display, (int(box.x), int(box.y)),
                              (int(box.x + box.width), int(box.bottom)), (0, 255, 0), 2)
            cv2.imshow("competition belly", display)
    except KeyboardInterrupt:
        print("预览已退出。")
        return True
    finally:
        io.close_views()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--color", choices=("red", "blue", "yellow"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--calibrate-carry", action="store_true", help="双摄搬运H/D标定，不打开身体串口")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--actions", action="store_true")
    parser.add_argument("--use-original-drop", action="store_true",
                        help="兼容旧命令；仍须有效腹部D，不再按二维码下沿直接放下")
    args = parser.parse_args()
    if sum((args.check, args.simulate, args.preview, args.calibrate_carry)) > 1:
        parser.error("check、simulate、preview、calibrate-carry只能选一种")
    if args.actions and (not args.run or args.check or args.simulate or args.preview or args.calibrate_carry):
        parser.error("真实身体动作只用 --run --actions")
    settings = load_settings(args.config)
    if args.check or not (args.simulate or args.preview or args.calibrate_carry or args.run):
        return preflight(settings, use_original_drop=args.use_original_drop)
    color = args.color or ("blue" if args.simulate else input("现场指定颜色 red/blue/yellow：").strip().lower())
    if color not in ("red", "blue", "yellow"):
        parser.error("请选择red、blue、yellow")
    if args.simulate:
        return simulate(settings, color)
    if args.calibrate_carry:
        if not args.run:
            parser.error("摄像头和头部标定需要 --run --calibrate-carry")
        from carry_vision import run
        run(color, settings["drop_qr"], qr_reader=read_codes)
        return True
    if args.preview:
        if not args.run:
            parser.error("摄像头和头部预览需要 --run --preview")
        return preview(settings, color)
    if not args.actions:
        parser.error("整场动作需要 --run --actions；无身体动作请使用 --run --preview")
    if not preflight(settings, use_original_drop=args.use_original_drop):
        return False
    from robotmove import RobotMove
    from robot_audio import configure
    configure()
    from chinese_speech import warm_up
    warm_up()  # 比赛计时和身体连接之前完成首次加载、合成。
    io = CompetitionIO(settings, None, float('inf'), use_original_drop=args.use_original_drop)
    try:
        io.prepare_identity()  # 开跑前加载；进入人脸阶段直接复用。
        deadline = time.monotonic() + settings["total_seconds"]
        io.deadline = deadline
        io.open_views()
        head, belly, _ = io.observe()
        reference = json.loads((ROOT / "config/carry_dual_reference.json").read_text())
        if reference.get("shapes") != dict(head=list(head.shape[:2]), belly=list(belly.shape[:2])):
            raise ValueError("实际双摄尺寸与搬运标定不同，身体未启动")
        head_reference = json.loads((ROOT / "config/head_calibration.json").read_text())
        foot_reference = json.loads((ROOT / "config/right_foot_reference.json").read_text())
        if (head_reference.get("shapes") != dict(head=list(head.shape[:2]), belly=list(belly.shape[:2]))
                or foot_reference.get("shape") != list(belly.shape[:2])):
            raise ValueError("实际双摄尺寸与足球标定不同，身体未启动")
        io.check_time()
        robot = RobotMove(None, port=BODY_PORT)  # 握手后自动STAND一次。
        try:
            io.robot = GuardedRobot(robot, deadline)
            io.flush()
            return run_course(io, color)
        finally:
            robot.close()
    finally:
        try:
            io.close_views()
        finally:
            io.close_identity_models()
