"""离线语音口令控制：在本目录执行 ``python3 voice_control.py``。"""

import json
import queue

COMMANDS = {
    "前进": "UP_LITTLE",
    "往前走": "UP_LITTLE",
    "后退": "BACK",
    "往后走": "BACK",
    "左转": "LEFT",
    "右转": "RIGHT1",
    "跳舞": "DANCE",
    "左脚踢球": "LEFT_BALL",
    "右脚踢球": "RIGHT_BALL",
    "抱箱子": "HOLD_BOX",
    "放箱子": "DOWN_BOX",
}


def command_for(text):
    """将一条识别文本转换为现有 robotmove.py 的动作名。"""
    return next((action for phrase, action in COMMANDS.items() if phrase in text), None)


def main():
    try:
        import pyaudio
        from vosk import KaldiRecognizer, Model
    except ImportError as exc:
        raise SystemExit("请先按 README-voice.md 安装 vosk 和 PyAudio") from exc

    # 延迟导入：让 --check 能在没有串口的电脑上运行。
    from roboteye import RobotEye
    from robotmove import RobotMove

    sample_rate = 16000
    audio_queue = queue.Queue()

    def on_audio(in_data, frame_count, time_info, status):
        if status:
            print("麦克风状态:", status)
        audio_queue.put(in_data)
        return None, pyaudio.paContinue

    model = Model("models/vosk-model-small-cn-0.22")
    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetGrammar(json.dumps([*COMMANDS, "[unk]"], ensure_ascii=False))
    microphone = pyaudio.PyAudio()
    stream = microphone.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=4000,
        stream_callback=on_audio,
    )
    robot = RobotMove(RobotEye())
    stream.start_stream()
    print("语音控制已启动：前进、后退、左转、右转、跳舞、左脚踢球、右脚踢球、抱箱子、放箱子")

    try:
        while True:
            if not recognizer.AcceptWaveform(audio_queue.get()):
                continue
            text = json.loads(recognizer.Result()).get("text", "")
            action = command_for(text)
            if action:
                print(f"识别到：{text}，执行：{action}")
                robot.robotMove(action)
            elif text:
                print(f"未支持的口令：{text}")
    except KeyboardInterrupt:
        print("语音控制已退出")
    finally:
        stream.stop_stream()
        stream.close()
        microphone.terminate()


def _check():
    assert command_for("请前进") == "UP_LITTLE"
    assert command_for("现在右转") == "RIGHT1"
    assert command_for("停止") is None


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        _check()
        print("指令映射检查通过")
    else:
        main()
