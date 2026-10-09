"""额外的寻线比赛版本，原competition.py保持不变。"""
import argparse
import json
from pathlib import Path
import time
from contextlib import ExitStack
from unittest.mock import Mock

import cv2
from robot_config import ROOT
from competition_main import (CompetitionIO, GuardedRobot, BODY_PORT, CONFIG_FILE,
                              load_settings, preflight)
from carry_vision import find_blocks, BlockTracker
from kick_shapes import overlap
from dual_kick import camera_settings
from line_search_route import (line_mask, route_segments, near_line_sample,
                               LinePlanner, LineWidthReference, run_course, run_from_line,
                               run_from_carry)


class LineCompetitionIO(CompetitionIO):
    line_color = "white"

    def follow_to_carry(self, color):
        # 寻线不限时，也不占用其他比赛阶段的剩余时间。
        started = time.monotonic()
        deadline = self.deadline
        robot_deadline = self.robot.deadline
        self.deadline = float("inf")
        self.robot.deadline = float("inf")
        try:
            return self.follow_line(color)
        finally:
            elapsed = time.monotonic()-started
            self.deadline = deadline+elapsed
            self.robot.deadline = robot_deadline+elapsed

    def follow_line(self, color):
        self.open_views()
        self.stream.watch(None, None)  # 不等待left和action1，也不让二维码触发转向。
        planner = LinePlanner()
        width_reference = LineWidthReference()
        previous_forward_completed = False
        self.set_head(127)
        print("[寻线] 正常前进后随走采10帧线宽，采样不额外停步。", flush=True)
        block_frames = 0
        previous_block = None
        block_tracker = BlockTracker(recover_head=True)
        while True:
            head, belly, _ = self.observe_ready()
            blocks = find_blocks(head, color)
            block = block_tracker.update(blocks,frame=head,color=color)
            if block is not None:
                previous_forward_completed = False
                same_block = previous_block is not None and overlap(previous_block, block) > .3
                block_frames = block_frames + 1 if same_block else 1
                previous_block = block
                display = head.copy()
                cv2.rectangle(display, (int(block.x), int(block.y)),
                              (int(block.x+block.width), int(block.bottom)), (0,255,0), 2)
                cv2.putText(display, f"{color} block {block_frames}/3", (10,30),
                            0, .7, (0,255,0), 2)
                cv2.imshow("competition head", display)
                # 出现候选时停步确认，由搬运模块继续找物和对位。
                if block_frames >= 3:
                    print("指定颜色物块连续确认3帧，转入搬运。",
                          f"颜色={color}，候选框={block}", flush=True)
                    return
                continue
            block_frames = 0
            previous_block = None
            mask = line_mask(belly, self.line_color)
            flip = camera_settings()["belly"]["flip"]
            vertical, diagonal = route_segments(mask, mirrored=flip in ("1", "-1"),
                                                recovering=planner.after_bend)
            vertical = width_reference.filter(mask, vertical)
            diagonal = width_reference.filter(mask, diagonal)
            x, angle = near_line_sample(vertical, belly.shape[0], belly.shape[1],
                                        planner.previous_x)
            action = planner.decide(x, belly.shape[1], right_diagonal=bool(diagonal),
                                    angle=angle, mirrored=flip in ("1", "-1"),
                                    original_visible=bool(vertical))
            # 上次前进已返回，当前新图仍满足直线前进条件，才用于线宽参考。
            # 采样未完成时照常使用原寻线规则，不额外WAIT。
            if previous_forward_completed and action == "UP_LITTLE" and x is not None:
                width_reference.sample(mask, vertical)
            display = belly.copy()
            sample_top = round(belly.shape[0]*.75)
            cv2.line(display, (0,sample_top), (belly.shape[1]-1,sample_top), (0,255,255), 2)
            for segments, paint in ((vertical,(0,255,0)),(diagonal,(0,165,255))):
                for x1,y1,x2,y2 in segments:
                    cv2.line(display,(x1,y1),(x2,y2),paint,2)
            cv2.putText(display, action, (10,30), 0, .7, (0,255,0),2)
            cv2.putText(display, f"vertical={len(vertical)} diagonal={len(diagonal)} angle={angle:.1f}",
                        (10,60),0,.7,(0,255,0),2)
            width_status = "width locked +/-30%" if width_reference.ready else f"width sampling {len(width_reference.frames)}/10"
            cv2.putText(display, width_status, (10,90), 0,.7,(0,255,255),2)
            cv2.imshow("line search", display)
            cv2.imshow("line mask", mask)
            print("[寻线]", action, "全画面竖直段：", len(vertical),
                  "右弯斜段：", len(diagonal), "仅斜线确认：", planner.diagonal_frames,
                  "角度：", round(angle,1), "弯后调正：", planner.after_bend,
                  "线宽：", width_status, flush=True)
            if action == "WAIT":
                previous_forward_completed = False
                continue  # 原地继续读取新图，可用Q或Ctrl+C退出。
            if action == "TURN_RIGHT_2":
                previous_forward_completed = False
                print("[寻线] 原竖直线已离开腹部画面，只剩右弯斜线，执行右转2次。", flush=True)
                for number in range(1,3):
                    # 每次转向前检查双摄，掉线不继续。
                    self.observe_ready()
                    print(f"[弯道] 右转 {number}/2", flush=True)
                    self.move("TURN_RIGHT")
                continue
            self.move(action)
            previous_forward_completed = action == "UP_LITTLE"


def preview_line(settings, line_color):
    io = LineCompetitionIO(settings, None, time.monotonic() + settings["total_seconds"])
    planner = LinePlanner()
    try:
        io.open_views()
        print("仅预览线路，不连接身体串口；Q退出。")
        while True:
            _, belly, _ = io.observe()
            mask = line_mask(belly, line_color)
            flip = camera_settings()["belly"]["flip"]
            vertical, diagonal = route_segments(mask, mirrored=flip in ("1", "-1"),
                                                recovering=planner.after_bend)
            x, angle = near_line_sample(vertical, belly.shape[0], belly.shape[1],
                                        planner.previous_x)
            action = planner.decide(x, belly.shape[1], right_diagonal=bool(diagonal),
                                    angle=angle, mirrored=flip in ("1", "-1"),
                                    original_visible=bool(vertical))
            display = belly.copy()
            sample_top = round(belly.shape[0]*.75)
            cv2.line(display, (0,sample_top), (belly.shape[1]-1,sample_top), (0,255,255), 2)
            for segments,paint in ((vertical,(0,255,0)),(diagonal,(0,165,255))):
                for x1,y1,x2,y2 in segments:
                    cv2.line(display,(x1,y1),(x2,y2),paint,2)
            cv2.putText(display,action,(10,30),0,.7,(0,255,0),2)
            cv2.putText(display,f"vertical={len(vertical)} diagonal={len(diagonal)} angle={angle:.1f}",
                        (10,60),0,.7,(0,255,0),2)
            width_status = "width inactive (preview)"
            cv2.putText(display, width_status, (10,90), 0,.7,(0,255,255),2)
            cv2.imshow("line search", display)
            cv2.imshow("line mask", mask)
    finally:
        io.close_views()
        cv2.destroyAllWindows()


def run_line_test(settings, color, line_color):
    """从当前位置直接执行寻线，确认物块后结束，不进入搬运。"""
    from robotmove import RobotMove
    deadline = float("inf")
    io = LineCompetitionIO(settings, None, deadline)
    io.line_color = line_color
    print("[寻线实走测试] 从当前位置开始；会站立并行走，请做好防倒保护。", flush=True)
    with ExitStack() as stack:
        stack.callback(io.close_views)
        stack.callback(cv2.destroyAllWindows)
        io.open_views()
        io.observe_ready()  # 双摄有新画面后才连接身体；不需要搬运/足球标定。
        io.check_time()
        robot = RobotMove(None, port=BODY_PORT)
        stack.callback(robot.close)
        io.robot = GuardedRobot(robot, deadline)
        io.flush()
        io.phase("只测试寻线，不执行扫码、人脸或搬运")
        io.follow_to_carry(color)
        print("[寻线实走测试] 已确认指定颜色物块，测试结束，不执行抱起或后续动作。", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--color", choices=("red", "blue", "yellow"), default="red")
    parser.add_argument("--line-color", choices=("white", "black"), default="white")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--actions", action="store_true")
    parser.add_argument("--line-only", action="store_true", help="从当前位置直接实走寻线，不执行其他比赛阶段")
    parser.add_argument("--from-line", action="store_true", help="临时测试：跳过人脸，从寻线连续执行到跳舞；可搭配--simulate")
    parser.add_argument("--from-carry", action="store_true", help="临时测试：跳过人脸和寻线，从搬运连续执行到跳舞；可搭配--simulate")
    args = parser.parse_args()
    if sum((args.check, args.simulate, args.preview)) > 1:
        parser.error("check、simulate、preview只能选一种")
    if args.actions and (not args.run or args.check or args.simulate or args.preview):
        parser.error("真实动作只用 --run --actions")
    if args.line_only and (not args.run or not args.actions or args.check or args.simulate or args.preview):
        parser.error("仅寻线实走须使用 --run --actions --line-only")
    if args.from_line and (args.line_only or args.preview):
        parser.error("from-line不能与line-only或preview同时使用")
    if args.from_carry and (args.from_line or args.line_only or args.preview):
        parser.error("from-carry不能与from-line、line-only或preview同时使用")
    settings = load_settings(args.config)
    course = run_from_line if args.from_line else run_course
    if args.from_carry:
        course = run_from_carry
    partial_start = args.from_line or args.from_carry
    if args.line_only:
        return run_line_test(settings, args.color, args.line_color)
    if args.simulate:
        io = Mock()
        io.settings = settings
        course(io, args.color)
        for call in io.method_calls:
            print("[模拟]", call)
        return True
    if args.preview:
        if not args.run:
            parser.error("摄像头预览须指定 --run --preview")
        return preview_line(settings, args.line_color)
    if args.check or not args.run:
        return preflight(settings)
    if not args.actions:
        parser.error("真实比赛须同时指定 --run --actions")
    if not preflight(settings):
        return False
    from robotmove import RobotMove
    from robot_audio import configure
    from chinese_speech import warm_up
    configure()
    if not partial_start:
        warm_up()
    io = LineCompetitionIO(settings, None, float('inf'))
    io.line_color = args.line_color
    with ExitStack() as stack:
        stack.callback(io.close_identity_models)
        if not partial_start:
            io.prepare_identity()
        deadline = time.monotonic() + settings["total_seconds"]
        io.deadline = deadline
        stack.callback(io.close_views)
        stack.callback(cv2.destroyAllWindows)
        io.open_views()
        head, belly, _ = io.observe()
        shapes = dict(head=list(head.shape[:2]), belly=list(belly.shape[:2]))
        carry = json.loads((ROOT / "config/carry_dual_reference.json").read_text())
        head_ref = json.loads((ROOT / "config/head_calibration.json").read_text())
        foot_ref = json.loads((ROOT / "config/right_foot_reference.json").read_text())
        if carry.get("shapes") != shapes or head_ref.get("shapes") != shapes or foot_ref.get("shape") != shapes["belly"]:
            raise ValueError("实际双摄尺寸与标定不同，身体未启动")
        io.check_time()
        robot = RobotMove(None, port=BODY_PORT)
        stack.callback(robot.close)
        io.robot = GuardedRobot(robot, deadline)
        io.flush()
        if args.from_line:
            print("[后半程实走测试] 从当前位置寻线到跳舞；会抱物、转向和行走，请做好防倒保护。", flush=True)
        if args.from_carry:
            print("[搬运后半程实走测试] 从当前位置找物搬运到跳舞；会站立、抱物和行走，请做好防倒保护。", flush=True)
        return course(io, args.color)
