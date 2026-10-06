import os
import sys
import time

from dreammaker_protocol import DreamMakerController


from robot_config import ROOT
ACTION_DIR = str(ROOT / "assets/actions")
ACTION_INTERVAL_SECONDS = float(os.environ.get("ROBOT_ACTION_INTERVAL", "0.8"))
WALK_DURATION_SCALE = float(os.environ.get("ROBOT_WALK_SCALE", "2.5"))
STRAIGHT_ACTIONS = ("UP", "UP_LITTLE", "BACK", "UP_HOLDBOX")


ACTIONS = {
    "STAND": (None, 1),
    "UP": ("forward.dzz", 1),
    "UP_LITTLE": ("forward.dzz", 1),
    "BACK": ("后退.dzz", 1),
    "LEFT": ("左转.dzz", 1),
    "TURN_LEFT": ("左转.dzz", 1),
    "LEFT2": ("左转.dzz", 1),
    "BIGLEFT": ("左转.dzz", 7),
    "RIGHT1": ("右转.dzz", 7),
    "TURN_RIGHT": ("右转.dzz", 1),
    "BIGRIGHT": ("右转.dzz", 7),
    "LEFT1": ("左平移.dzz", 1),
    "SIDE_LEFT": ("左平移.dzz", 1),
    "RIGHT": ("右平移.dzz", 1),
    "SIDE_RIGHT": ("右平移.dzz", 1),
    "HOLD_BOX": ("抱低处物块-减小前倾测试.dzz", 1),
    "DOWN_BOX": ("松.dzz", 1),
    "UP_HOLDBOX": ("搬运行走.dzz", 1),
    "LEFT_HOLDBOX": ("搬运左转.dzz", 1),
    "RIGHT_HOLDBOX": ("搬运右转.dzz", 1),
    "LEFT_BALL": ("L踢球.dzz", 1),
    "RIGHT_BALL": ("R踢球.dzz", 1),
    "DANCE": ("智能版人形跳舞1.dzz", 1),
    "DANCE1": ("智能版人形跳舞1.dzz", 1),
    "DANCE2": ("智能版人形跳舞2.dzz", 1),
    "DANCE3": ("智能版人形跳舞3.dzz", 1),
    "DANCE4": ("智能版人形跳舞4.dzz", 1),
    "DANCE5": ("智能版人形跳舞5.dzz", 1),
}


class RobotMove:
    def __init__(self, eye, port=None):
        self.__robotEye = eye
        self.__controller = DreamMakerController(port, online_protocol=True)
        try:
            self.__controller.open()
            self.stand()
        except BaseException:
            self.__controller.close()
            raise

    def stand(self):
        print("控制--> STAND，标准初始站姿")
        self.__controller.stand()

    def robotMove(self, ccAction):
        action = ccAction.upper()
        if action not in ACTIONS:
            raise ValueError("未知机器人动作：{}".format(ccAction))
        filename, repeats = ACTIONS[action]
        if action == "STAND":
            self.stand()
            return
        path = os.path.join(ACTION_DIR, filename)
        print("控制--> {}，{}，执行 {} 次".format(action, filename, repeats))
        duration_scale = WALK_DURATION_SCALE if action in STRAIGHT_ACTIONS else None
        if action == "HOLD_BOX":
            # 使用实机测试过的600毫秒过渡，不再叠加通用1.5倍时长。
            duration_scale = 1.0
        for _ in range(repeats):
            self.__controller.play_dzz(path, duration_scale=duration_scale)
            time.sleep(max(0.0, ACTION_INTERVAL_SECONDS))

    def close(self):
        self.__controller.close()


def main():
    """打开一次串口，完成一次握手，然后连续接收英文动作名。"""
    robot = RobotMove(None)
    print("Ready. Enter an action name; enter QUIT to exit.")
    print("Actions: " + " ".join(sorted(ACTIONS)))
    try:
        while True:
            action = input("action> ").strip().upper()
            if action in ("QUIT", "EXIT", "Q"):
                return True
            if not action:
                continue
            try:
                robot.robotMove(action)
            except ValueError as error:
                print(error)
    finally:
        robot.close()


if __name__ == "__main__":
    try:
        success = main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
        success = True
    except Exception as error:
        print("Stopped: {}".format(error), file=sys.stderr)
        success = False
    raise SystemExit(0 if success else 1)
