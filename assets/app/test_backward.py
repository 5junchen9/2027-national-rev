"""独立后退调试：默认只打印；--run逐帧，--run --continuous完整执行一次。"""

import argparse
import os
import time

from robot_config import ROOT
from dreammaker_protocol import (
    DEFAULT_INITIAL_POSITIONS,
    DreamMakerController,
    load_dzz,
    motion_frame,
    scaled_duration_ms,
)


ACTION_FILE = ROOT / "assets/actions/后退-准备与收尾测试.dzz"
PORT = "/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="连接身体串口，实际执行")
    parser.add_argument("--continuous", action="store_true", help="逐帧检查后，完整执行一次")
    parser.add_argument("--old", action="store_true", help="对比原后退动作，使用当前行走时间倍率")
    args = parser.parse_args()
    action_file = ROOT / "assets/actions/后退.dzz" if args.old else ACTION_FILE
    frames = load_dzz(action_file)
    if args.old:
        walk_scale = float(os.environ.get("ROBOT_WALK_SCALE", "2.5"))
        frames = [(offsets, scaled_duration_ms(duration_ms, walk_scale))
                  for offsets, duration_ms in frames]

    # 在打开串口前检查所有帧，避免越界被apply_offsets静默截断。
    payloads = []
    for offsets, duration_ms in frames:
        positions = [base + offset for base, offset in zip(DEFAULT_INITIAL_POSITIONS, offsets)]
        payloads.append(motion_frame(positions, duration_ms))

    print("测试动作：", action_file, flush=True)
    if args.old:
        print(f"原后退共{len(frames)}帧，行走时间倍率={walk_scale}；不添加准备和收尾帧。", flush=True)
        print("默认倍率2.5时每帧100ms；仅测试后退，不自动接右转。", flush=True)
    else:
        print("共10帧：准备600ms → 原后退8帧各100ms → 腿部归零收尾400ms。", flush=True)
        print("不使用ROBOT_WALK_SCALE或ROBOT_MOTION_SCALE；文件时间就是发送时间。", flush=True)
        print("这是待验证候选，尚未证明稳定或不偏航。", flush=True)
    for number, (offsets, duration_ms) in enumerate(frames, 1):
        changed = " ".join(f"{channel}:{value:+d}" for channel, value in enumerate(offsets, 1) if value)
        print(f"第{number}帧，{duration_ms}ms；非零通道偏移：{changed}", flush=True)
    if not args.run:
        print("仅检查文件，未连接串口、未执行动作。")
        return

    print("先关闭比赛/其他身体控制程序，机器人空手，后方留空，做好防倒扶持。", flush=True)
    print("逐帧停留会改变步态，不能用逐帧成功证明连贯后退稳定。", flush=True)
    if input("保护到位后输入 y：先缓慢回标准站姿；其他输入退出：").strip().lower() != "y":
        return

    controller = DreamMakerController(PORT, online_protocol=True)
    try:
        controller.open()
        # 默认stand只有40ms；调试中显式用1000ms回站姿，避免突然跳姿态。
        print("回标准站姿，过渡1000ms，随后等待2秒。", flush=True)
        controller.serial.write(motion_frame(DEFAULT_INITIAL_POSITIONS, 1000))
        controller.serial.flush()
        time.sleep(3)
        if args.continuous:
            if input("确认双脚落稳，输入 y 完整后退一次；其他输入退出：").strip().lower() != "y":
                return
        for number, ((offsets, duration_ms), payload) in enumerate(zip(frames, payloads), 1):
            if not args.continuous:
                answer = input(f"第{number}/{len(frames)}帧：有扶持且可继续则回车，其他输入停止：")
                if answer.strip():
                    return
            print(f"发送第{number}帧，过渡{duration_ms}ms", flush=True)
            controller.serial.write(payload)
            controller.serial.flush()
            time.sleep(duration_ms / 1000)
        print("动作发送完毕。观察是否站稳、向哪侧偏、退了多远；不自动接右转。", flush=True)
    finally:
        # 中途退出时不能盲目回站或卸力，先由人工扶稳。
        controller.close()
        print("串口已关闭；不自动复位或卸力，已发动作不能撤回。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n测试停止；已发动作不能撤回，请扶稳机器人。", flush=True)
