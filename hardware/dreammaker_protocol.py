# -*- coding: utf-8 -*-
"""DreamMaker/SIGMA 控制板的最小串口协议实现。"""

import argparse
import os
from robot_config import ROOT
import sys
import time


BAUD_RATE = int(os.environ.get("ROBOT_BAUD", "115200"))
TRACE_HEX = os.environ.get("ROBOT_TRACE_HEX", "").lower() in ("1", "true", "yes")
OPEN_SETTLE_SECONDS = float(os.environ.get("ROBOT_OPEN_DELAY", "0"))
F9_SETTLE_SECONDS = float(os.environ.get("ROBOT_F9_DELAY", "1.0"))
FB_SETTLE_SECONDS = float(os.environ.get("ROBOT_FB_DELAY", "0.5"))
MOTION_DURATION_SCALE = float(os.environ.get("ROBOT_MOTION_SCALE", "1.5"))
MOTION_EXTRA_SECONDS = float(os.environ.get("ROBOT_FRAME_EXTRA_DELAY", "0"))
STAND_DURATION_MS = 40
SERVO_COUNT = 24  # DreamMaker 协议槽位数；本机器人实际安装 19 个舵机。
REQUEST_PAIR_COUNT = 25
FRAME_SIZE = 51
ONLINE_FRAME_SIZE = 51
INITIAL_RESPONSE_SIZE = 52
DEFAULT_PORT = os.environ.get(
    "ROBOT_SERIAL_PORT", "/dev/ttyUSB1" if os.name == "posix" else "COM1"
)
DEFAULT_INITIAL_POSITIONS = (
    512, 518, 641, 511, 510, 361, 486, 518,
    362, 522, 771, 645, 506, 518, 649, 515,
    242, 367, 503, 400, 400, 400, 400, 400,
)
KNOWN_WORKING_FRAME = bytes.fromhex(
    "02 00 01 E8 01 7D 01 FF 01 C2 02 6D 01 E6 02 06 01 4C 02 00 "
    "03 03 02 76 01 E1 02 06 02 66 01 F3 01 00 01 52 01 E2 01 90 "
    "01 90 01 90 01 90 01 90 00 28 FD"
)
INITIAL_RESPONSE_EXAMPLE = bytes.fromhex(
    "FB 02 00 02 06 02 81 01 FF 01 FE 01 69 01 E6 02 06 01 6A 02 0A "
    "03 03 02 85 01 FA 02 06 02 89 02 03 00 F2 01 6F 01 F7 01 90 01 90 "
    "01 90 01 90 01 90 03 20 FE"
)

CMD_OPEN = 0xF9
CMD_INITIAL_REQUEST = 0xFA
CMD_INITIAL_RESPONSE = 0xFB
CMD_LIVE_MOVE = 0xFD
CMD_SERVO_DISABLE = 0xD0
CMD_SERVO_ENABLE = 0xD1
CMD_SERVO_READ = 0xD2
CMD_SERVO_RESPONSE = 0xD3


def session_frame(command):
    """打开串口会话：50 个 0x00 加命令字。"""
    return bytes(50) + bytes((command,))


def request_frame(command):
    """联机调试命令：25 组 00 FF 加命令字，共 51 字节。"""
    return bytes((0x00, 0xFF)) * REQUEST_PAIR_COUNT + bytes((command,))


def motion_frame(positions, duration_ms, command=CMD_LIVE_MOVE):
    """24 路绝对位置和动作时长按大端序打包为 51 字节。"""
    if len(positions) != SERVO_COUNT:
        raise ValueError("需要 {} 路舵机位置".format(SERVO_COUNT))
    if not 0 <= duration_ms <= 3000:
        raise ValueError("动作时长必须为 0 至 3000 毫秒")
    values = list(positions) + [duration_ms]
    if any(not 1 <= value <= 1023 for value in positions):
        raise ValueError("舵机绝对位置必须为 1 至 1023")
    return b"".join(int(value).to_bytes(2, "big") for value in values) + bytes((command,))


def scaled_duration_ms(duration_ms, scale=None):
    """按整机稳定系数放慢舵机过渡时间。"""
    scale = MOTION_DURATION_SCALE if scale is None else scale
    if scale <= 0:
        raise ValueError("ROBOT_MOTION_SCALE 必须大于 0")
    return max(0, min(3000, round(duration_ms * scale)))


def apply_offsets(initial_positions, offsets):
    if len(initial_positions) != SERVO_COUNT or len(offsets) != SERVO_COUNT:
        raise ValueError("初始位置和偏移量都必须包含 {} 路数据".format(SERVO_COUNT))
    return [max(1, min(1023, base + offset))
            for base, offset in zip(initial_positions, offsets)]


def parse_initial_positions(text):
    values = [int(value) for value in text.replace(",", " ").split()]
    if len(values) == 1:
        values *= SERVO_COUNT
    if len(values) != SERVO_COUNT:
        raise ValueError("初始位置必须是 1 个整数或 {} 个整数".format(SERVO_COUNT))
    if any(not 1 <= value <= 1023 for value in values):
        raise ValueError("初始位置必须为 1 至 1023")
    return values


def parse_position_response(data, marker):
    marker_index = data.find(bytes((marker,)))
    end = marker_index + 1 + SERVO_COUNT * 2
    if marker_index < 0 or len(data) < end:
        raise ValueError("未找到完整的 0x{:02X} 位置响应".format(marker))
    payload = data[marker_index + 1:end]
    return [int.from_bytes(payload[index:index + 2], "big")
            for index in range(0, len(payload), 2)]


def load_dzz(path):
    frames = []
    with open(path, "r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            values = [int(value) for value in line.split()]
            # 搬运左转第 1 帧的旧文件在时间前多写了一个 0；保留源文件并兼容读取。
            if len(values) == SERVO_COUNT + 2 and values[-2] == 0:
                values = values[:SERVO_COUNT] + values[-1:]
            if len(values) != SERVO_COUNT + 1:
                raise ValueError("{} 第 {} 行应有 25 个整数".format(path, line_number))
            frames.append((values[:SERVO_COUNT], values[-1]))
    if not frames:
        raise ValueError("动作文件没有有效帧：{}".format(path))
    return frames


class DreamMakerController:
    def __init__(self, port=None, timeout=5.0, serial_port=None, online_protocol=False):
        self.port = port or DEFAULT_PORT
        self.timeout = timeout
        self.serial = serial_port
        self._owns_serial = serial_port is None
        self.online_protocol = online_protocol
        self._initial_positions = None

    def _write(self, payload, label):
        if TRACE_HEX:
            print("TX {} [{}]: {}".format(
                label, len(payload), payload.hex(" ").upper()))
        written = self.serial.write(payload)
        if written is not None and written != len(payload):
            raise IOError("{} only wrote {}/{} bytes".format(
                label, written, len(payload)))
        self.serial.flush()

    def _wait_for_online_response(self):
        deadline = time.monotonic() + self.timeout
        received = bytearray()
        while time.monotonic() < deadline:
            chunk = self.serial.read(getattr(self.serial, "in_waiting", 0) or 1)
            if not chunk:
                continue
            received.extend(chunk)
            search_from = 0
            while True:
                start = received.find(bytes((CMD_INITIAL_RESPONSE,)), search_from)
                if start < 0 or len(received) < start + INITIAL_RESPONSE_SIZE:
                    break
                frame = bytes(received[start:start + INITIAL_RESPONSE_SIZE])
                if frame[-1] == 0xFE:
                    if TRACE_HEX:
                        print("RX FB [{}]: {}".format(
                            INITIAL_RESPONSE_SIZE, frame.hex(" ").upper()))
                    self._initial_positions = parse_position_response(
                        frame, CMD_INITIAL_RESPONSE
                    )
                    return self._initial_positions
                search_from = start + 1
        received_hex = bytes(received[:100]).hex(" ").upper() or "<empty>"
        raise TimeoutError(
            "Expected complete FB initial response on {} at {} baud; RX={}".format(
                self.port, BAUD_RATE, received_hex
            )
        )

    def open(self):
        try:
            return self._open()
        except BaseException:
            self.close()
            raise

    def _open(self):
        if self.serial is None:
            import serial
            self.serial = serial.Serial(port=None, baudrate=BAUD_RATE,
                                        timeout=self.timeout,
                                        write_timeout=self.timeout)
            self.serial.port = self.port
        # 对齐 DreamMaker/.NET SerialPort 默认控制线状态。
        if hasattr(self.serial, "dtr"):
            self.serial.dtr = False
        if hasattr(self.serial, "rts"):
            self.serial.rts = False
        if hasattr(self.serial, "is_open") and not self.serial.is_open:
            self.serial.open()
        if self.online_protocol:
            time.sleep(max(0.0, OPEN_SETTLE_SECONDS))
            self._write(session_frame(CMD_OPEN), "F9")
            time.sleep(max(0.0, F9_SETTLE_SECONDS))
            self._write(request_frame(CMD_INITIAL_REQUEST), "FA")
            self._wait_for_online_response()
            time.sleep(max(0.0, FB_SETTLE_SECONDS))
        return self

    def close(self):
        if self.serial is None:
            return
        # 不发送 F8：实机要求保持联机状态，程序只释放本地串口句柄。
        if self._owns_serial:
            self.serial.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def _read_positions(self, request_command, response_command):
        last_received = bytearray()
        for _ in range(2):
            self.serial.reset_input_buffer()
            self._write(request_frame(request_command), "{:02X}".format(request_command))
            deadline = time.monotonic() + self.timeout
            received = bytearray()
            while time.monotonic() < deadline:
                chunk = self.serial.read(getattr(self.serial, "in_waiting", 0) or 1)
                if chunk:
                    received.extend(chunk)
                    try:
                        return parse_position_response(received, response_command)
                    except ValueError:
                        pass
            last_received = received
        received_hex = bytes(last_received[:80]).hex(" ").upper() or "<empty>"
        raise TimeoutError(
            "No complete 0x{:02X} response on {} at {} baud; RX={}".format(
                response_command, self.port, BAUD_RATE, received_hex))

    def initial_positions(self):
        if self._initial_positions is not None:
            return list(self._initial_positions)
        return self._read_positions(CMD_INITIAL_REQUEST, CMD_INITIAL_RESPONSE)

    def read_positions(self):
        return self._read_positions(CMD_SERVO_READ, CMD_SERVO_RESPONSE)

    def set_servo_enabled(self, enabled):
        command = CMD_SERVO_ENABLE if enabled else CMD_SERVO_DISABLE
        self._write(request_frame(command), "{:02X}".format(command))

    def send_action(self, code):
        if len(code) != 1 or ord(code) > 0x7F:
            raise ValueError("动作码必须是一个 ASCII 字符")
        self._write(code.encode("ascii"), "ASCII action")

    def play_dzz(self, path, initial_positions=None, duration_scale=None):
        initial = (self._initial_positions or DEFAULT_INITIAL_POSITIONS
                   if initial_positions is None else initial_positions)
        for frame_number, (offsets, duration_ms) in enumerate(load_dzz(path), 1):
            positions = apply_offsets(initial, offsets)
            duration_ms = scaled_duration_ms(duration_ms, duration_scale)
            self._write(motion_frame(positions, duration_ms),
                        "FD frame {}".format(frame_number))
            time.sleep(duration_ms / 1000.0 + max(0.0, MOTION_EXTRA_SECONDS))

    def send_known_working_frame(self):
        self._write(KNOWN_WORKING_FRAME, "KNOWN FD")

    def stand(self):
        """发送动作表中的24路标准站立绝对位置。"""
        self._write(motion_frame(DEFAULT_INITIAL_POSITIONS, STAND_DURATION_MS),
                    "STAND FD")
        self._initial_positions = list(DEFAULT_INITIAL_POSITIONS)
        time.sleep(STAND_DURATION_MS / 1000.0 + max(0.0, MOTION_EXTRA_SECONDS))


def self_check():
    assert len(session_frame(CMD_OPEN)) == FRAME_SIZE
    assert session_frame(CMD_OPEN)[-1] == 0xF9
    assert len(request_frame(CMD_INITIAL_REQUEST)) == ONLINE_FRAME_SIZE
    assert request_frame(CMD_INITIAL_REQUEST)[:4] == b"\x00\xFF\x00\xFF"
    assert request_frame(CMD_INITIAL_REQUEST)[-1] == 0xFA
    expected = b"\x02\x00" * SERVO_COUNT + b"\x03\x20\xFD"
    assert motion_frame([512] * SERVO_COUNT, 800) == expected
    assert len(KNOWN_WORKING_FRAME) == FRAME_SIZE
    assert KNOWN_WORKING_FRAME[-1] == CMD_LIVE_MOVE
    assert len(motion_frame(DEFAULT_INITIAL_POSITIONS, STAND_DURATION_MS)) == FRAME_SIZE
    assert scaled_duration_ms(50) == min(3000, round(50 * MOTION_DURATION_SCALE))
    assert scaled_duration_ms(50, 2.0) == 100
    assert len(INITIAL_RESPONSE_EXAMPLE) == INITIAL_RESPONSE_SIZE
    assert INITIAL_RESPONSE_EXAMPLE[0] == CMD_INITIAL_RESPONSE
    assert INITIAL_RESPONSE_EXAMPLE[-1] == 0xFE
    assert parse_position_response(
        INITIAL_RESPONSE_EXAMPLE, CMD_INITIAL_RESPONSE
    ) == list(DEFAULT_INITIAL_POSITIONS)
    response = b"noise\xFB" + b"\x02\x00" * SERVO_COUNT
    assert parse_position_response(response, CMD_INITIAL_RESPONSE) == [512] * SERVO_COUNT
    assert apply_offsets([512] * SERVO_COUNT, [-999] + [0] * 22 + [999]) == \
        [1] + [512] * 22 + [1023]
    assert parse_initial_positions("512") == [512] * SERVO_COUNT
    assert parse_initial_positions(" ".join(map(str, range(500, 524)))) == list(range(500, 524))
    print("DreamMaker protocol check passed: {} 8N1, 51-byte frames".format(
        BAUD_RATE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=5.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="只做协议自检，不打开串口")
    subparsers.add_parser("probe", help="读取控制板的 24 路初始位置")
    action = subparsers.add_parser("action", help="发送一个已下载动作的单字符编号")
    action.add_argument("code")
    action.add_argument("--run", action="store_true", help="确认允许机器人动作")
    action.add_argument("--wait", type=float, default=2.0,
                        help="发送后保持会话的秒数，默认 2 秒")
    play = subparsers.add_parser("play", help="直接播放 DreamMaker .dzz 动作文件")
    play.add_argument("path")
    play.add_argument(
        "--initial",
        help="跳过位置回读：填 1 个基准值，或用逗号分隔填写 24 路初始位置",
    )
    play.add_argument("--run", action="store_true", help="确认允许机器人动作")
    known = subparsers.add_parser("known", help="发送实机成功过的固定 FD 测试帧")
    known.add_argument("--run", action="store_true", help="确认允许机器人动作")
    known.add_argument("--wait", type=float, default=5.0,
                       help="发送后等待秒数，默认 5 秒")
    args = parser.parse_args()

    if args.command == "check":
        self_check()
        return True
    if args.command in ("action", "play", "known") and not args.run:
        parser.error("会驱动机器人运动，请加 --run")

    print("Serial: {} -> {} @ {} 8N1".format(
        args.port, os.path.realpath(args.port), BAUD_RATE))
    online_protocol = args.command in ("probe", "play", "known")
    with DreamMakerController(args.port, args.timeout,
                              online_protocol=online_protocol) as controller:
        if args.command == "probe":
            print("Initial positions: " + " ".join(map(str, controller.initial_positions())))
        elif args.command == "action":
            controller.send_action(args.code)
            print("Action byte sent: {!r}; waiting {:.1f}s".format(args.code, args.wait))
            time.sleep(max(0.0, args.wait))
        elif args.command == "known":
            controller.send_known_working_frame()
            print("Known working FD frame sent; waiting {:.1f}s".format(args.wait))
            time.sleep(max(0.0, args.wait))
        else:
            initial = parse_initial_positions(args.initial) if args.initial else None
            print("Using {} initial positions".format(
                "supplied" if args.initial else "FB-confirmed"))
            controller.play_dzz(args.path, initial)
    return True


if __name__ == "__main__":
    try:
        success = main()
    except TimeoutError as error:
        print("Handshake failed: {}".format(error), file=sys.stderr)
        print(
            "RX=<empty> means the controller-to-Pi return path is silent. "
            "Check that the selected USB serial port is the robot controller.",
            file=sys.stderr,
        )
        success = False
    raise SystemExit(0 if success else 1)
