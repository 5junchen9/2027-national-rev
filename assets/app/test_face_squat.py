"""单独测试下蹲和头部取景；不改变比赛配置。"""
import argparse
from contextlib import ExitStack
import time

from robot_config import ROOT
from dreammaker_protocol import load_dzz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="执行下蹲并打开头部摄像头")
    args = parser.parse_args()
    action = ROOT / "assets/actions/人脸识别下蹲-测试.dzz"
    print("动作：", action)
    print("顺序：站立一次 → 等待2秒 → 仅下蹲第一帧 → 保持姿态调头位。")
    for number, frame in enumerate(load_dzz(action), 1):
        print("帧", number, frame)
    if not args.run:
        print("仅查看数据；加 --run 才连接硬件。")
        return

    import cv2
    from Head import RobotHeadServoOnly
    from roboteye import RobotEye
    from robotmove import RobotMove
    from competition_main import BODY_PORT
    from dual_kick import camera_settings

    with ExitStack() as stack:
        # 先确认摄像头可用，再发送身体动作。
        eye = RobotEye(**camera_settings()["head"], latest=True)
        stack.callback(eye.close)
        head = RobotHeadServoOnly(hold=True, backend="pigpio")
        stack.callback(head.cleanup)
        stack.callback(cv2.destroyAllWindows)
        angle = 130
        head.turn_vertical(angle)
        input("空手并扶住机器人，回车执行站立和下蹲；Ctrl+C取消：")
        robot = RobotMove(None, port=BODY_PORT)
        stack.callback(robot.close)
        time.sleep(2)
        robot.robotMove("SQUAT")
        print("点击画面：A减1，D加1，Q退出；退出保持蹲姿，不自动起身。")

        while True:
            ok, image = eye.getImage()
            if not ok:
                raise RuntimeError("头部摄像头读取失败")
            cv2.putText(image, f"HEAD={angle}  A:-1 D:+1 Q:quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("face squat calibration", image)
            key = cv2.waitKey(20) & 0xff
            if key in (ord("q"), ord("Q"), 27):
                break
            new_angle = angle
            if key in (ord("a"), ord("A")):
                new_angle = max(85, angle - 1)
            elif key in (ord("d"), ord("D")):
                new_angle = min(180, angle + 1)
            if new_angle != angle:
                angle = new_angle
                head.turn_vertical(angle)
                print("头位：", angle, flush=True)
        print("最终头位：", angle)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("测试停止；已发送的动作不能撤回，身体不自动起身。")
