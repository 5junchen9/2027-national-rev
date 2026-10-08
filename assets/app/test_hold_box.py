"""独立抱起测试：先站立，再逐帧执行；默认仅打印，不连接机器人。"""

import argparse
import time

from robot_config import ROOT
from dreammaker_protocol import (
    DEFAULT_INITIAL_POSITIONS,
    DreamMakerController,
    apply_offsets,
    load_dzz,
    motion_frame,
)


ACTION_FILE = ROOT / "assets/actions/抱低处物块-减小前倾测试.dzz"
PORT = "/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="连接机器人并执行动作")
    args = parser.parse_args()
    frames = load_dzz(ACTION_FILE)

    print("测试动作：", ACTION_FILE, flush=True)
    print("髋部9/15号偏移减到-150/+150；这是待实机验证的姿态。", flush=True)
    print("动作顺序：STAND → 等待2秒 → 抱起4帧。", flush=True)
    controller = None
    try:
        if args.run:
            print("请先做好防倒保护。每帧前回车继续，输入q结束。", flush=True)
            if input("回车执行站立，q退出：").strip().lower() == "q":
                return
            controller = DreamMakerController(PORT, online_protocol=True)
            controller.open()

        print("控制--> STAND，标准初始站姿（一次）", flush=True)
        if controller is not None:
            controller.stand()
            time.sleep(2)

        for number, (offsets, duration_ms) in enumerate(frames, 1):
            positions = apply_offsets(DEFAULT_INITIAL_POSITIONS, offsets)
            payload = motion_frame(positions, duration_ms)
            print(f"第{number}帧：过渡{duration_ms}毫秒；24路偏移={offsets}", flush=True)
            if controller is not None:
                if input("确认有保护且姿态可继续，回车执行，q结束：").strip().lower() == "q":
                    return
                controller.serial.write(payload)
                controller.serial.flush()
                time.sleep(duration_ms / 1000.0)

        print("测试流程结束，请人工确认是否站稳、抱住。", flush=True)
    finally:
        if controller is not None:
            # 只关闭串口，不自动松开、站立或关闭舵机力矩。
            controller.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n测试停止；已发送的动作不会被撤回。", flush=True)
