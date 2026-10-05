"""独立踢球调试入口：不接入 main_1.py。"""

import argparse
import os
import sys
import time
from dataclasses import dataclass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(BASE_DIR, "b31", "2")
if not os.path.isdir(MODULE_DIR):
    MODULE_DIR = os.path.join(BASE_DIR, "modules")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)


@dataclass
class KickConfig:
    ball_color: str = "ball"
    head_angle: int = 120
    timeout_seconds: float = 60.0
    align_left: int = 200
    align_right: int = 450
    image_center_x: int = 320
    kick_trigger_y: int = 380
    kick_approach_steps: int = 1
    near_confirm_frames: int = 2
    search_after_misses: int = 5
    min_size: int = 8
    max_size: int = 270


def choose_action(center_x, center_y, config):
    """根据球在画面中的位置选择下一次已有动作。"""
    if center_x < config.align_left:
        return "RIGHT"
    if center_x > config.align_right:
        return "LEFT1"
    if center_y < config.kick_trigger_y:
        return "UP_LITTLE"
    return "LEFT_BALL" if center_x <= config.image_center_x else "RIGHT_BALL"


class KickTask:
    # ponytail: 第一版只复用 HSV 颜色轮廓；颜色不稳定时再增加圆形筛选或专用模型。
    def __init__(self, config, dry_run):
        from Head import RobotHeadServoOnly
        from findbox import FindBox
        from roboteye import RobotEye

        self.config = config
        self.dry_run = dry_run
        self.eye = RobotEye()
        self.head = RobotHeadServoOnly()
        self.finder = FindBox()
        self.move = None
        if not dry_run:
            from robotmove import RobotMove
            self.move = RobotMove(self.eye)

    def _move(self, action):
        if self.dry_run:
            print("[模拟动作]", action)
            time.sleep(0.15)
            return
        self.move.robotMove(action)

    def _find_ball(self, frame):
        found, frame = self.finder.findRect(self.config.ball_color, frame)
        if not found:
            return None, frame
        rect = self.finder.cFindBoxRect
        if not (self.config.min_size < rect.width < self.config.max_size and
                self.config.min_size < rect.height < self.config.max_size):
            return None, frame
        return rect, frame

    def _show(self, frame, message):
        import cv2

        cv2.putText(frame, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2)
        self.eye.showImage(frame)
        return cv2.waitKey(1) & 0xFF

    def run(self):
        mode = "模拟动作" if self.dry_run else "真实动作"
        print("== 独立踢球调试：{} ==".format(mode))
        print("按 q 可停止；球色：{}；补偿步数：{}".format(
            self.config.ball_color, self.config.kick_approach_steps))
        self.head.turn_vertical(self.config.head_angle)

        deadline = time.monotonic() + self.config.timeout_seconds
        misses = 0
        near_frames = 0
        last_action = None

        try:
            while time.monotonic() < deadline:
                ok, frame = self.eye.getImage()
                if not ok:
                    print("摄像头取帧失败")
                    continue

                ball, frame = self._find_ball(frame)
                if ball is None:
                    near_frames = 0
                    misses += 1
                    if self._show(frame, "searching ball") == ord("q"):
                        return False
                    if misses >= self.config.search_after_misses:
                        self._move("LEFT2")
                        misses = 0
                    continue

                misses = 0
                action = choose_action(ball.centerX, ball.centerY, self.config)
                message = "ball=({}, {}) next={}".format(ball.centerX, ball.centerY, action)
                if self._show(frame, message) == ord("q"):
                    return False

                if action in ("LEFT_BALL", "RIGHT_BALL"):
                    near_frames += 1
                    print("球进入最后可见区：{}/{}".format(
                        near_frames, self.config.near_confirm_frames))
                    if near_frames < self.config.near_confirm_frames:
                        continue

                    for _ in range(self.config.kick_approach_steps):
                        self._move("UP_LITTLE")
                    self._move(action)
                    print("== 踢球完成：{} ==".format(action))
                    return True

                near_frames = 0
                if action != last_action:
                    print(message)
                last_action = action
                self._move(action)

            print("== 踢球超时：未找到或未对准球 ==")
            return False
        finally:
            import cv2

            cv2.destroyAllWindows()
            self.head.cleanup()


def _check():
    config = KickConfig()
    assert choose_action(199, 200, config) == "RIGHT"
    assert choose_action(451, 200, config) == "LEFT1"
    assert choose_action(320, 379, config) == "UP_LITTLE"
    assert choose_action(320, 380, config) == "LEFT_BALL"
    assert choose_action(321, 380, config) == "RIGHT_BALL"
    print("踢球决策自检通过")


def parse_args():
    parser = argparse.ArgumentParser(description="独立找球与踢球调试")
    parser.add_argument("--check", action="store_true", help="只运行无硬件决策自检")
    parser.add_argument("--run", action="store_true", help="允许发送真实机器人动作")
    parser.add_argument("--timeout", type=float, default=60.0, help="找球超时秒数")
    parser.add_argument("--approach-steps", type=int, default=1, help="脚边盲区补偿前进次数")
    parser.add_argument("--trigger-y", type=int, default=380, help="最后可见区域的 Y 触发线")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check:
        _check()
        raise SystemExit(0)

    config = KickConfig(
        timeout_seconds=args.timeout,
        kick_approach_steps=max(0, args.approach_steps),
        kick_trigger_y=args.trigger_y,
    )
    success = KickTask(config, dry_run=not args.run).run()
    raise SystemExit(0 if success else 1)
