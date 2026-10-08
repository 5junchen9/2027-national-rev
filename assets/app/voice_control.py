"""保留 Vosk 麦克风口令功能；无摄像头依赖，按需加载模型。"""
import argparse
from contextlib import ExitStack
import json
import os
import queue
from robot_config import ROOT

COMMANDS = {
    "前进": "UP_LITTLE", "往前走": "UP_LITTLE",
    "后退": "BACK", "往后走": "BACK", "左转": "LEFT", "右转": "RIGHT1",
    "跳舞": "DANCE", "左脚踢球": "LEFT_BALL", "右脚踢球": "RIGHT_BALL",
    "抱箱子": "HOLD_BOX", "放箱子": "DOWN_BOX",
}
MODEL_PATH = ROOT / "assets/models/asr/vosk-model-small-cn-0.22"


def command_for(text):
    text = "".join(text.split())
    return next((action for phrase, action in COMMANDS.items() if phrase in text), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    import pyaudio

    with ExitStack() as cleanup:
        microphone = pyaudio.PyAudio()
        cleanup.callback(microphone.terminate)
        if args.list_devices:
            for index in range(microphone.get_device_count()):
                info = microphone.get_device_info_by_index(index)
                if info["maxInputChannels"]:
                    print(index, info["name"], "input channels:", info["maxInputChannels"])
            return True
        from vosk import KaldiRecognizer, Model
        from robotmove import RobotMove

        recognizer = KaldiRecognizer(Model(str(MODEL_PATH)), 16000)
        # 小中文模型词表不含“左脚踢球”等整句；使用原语言模型，避免
        # 设置整句 grammar 后这些口令被 Vosk 直接丢弃。
        audio_queue = queue.Queue(maxsize=8)

        def on_audio(in_data, _frames, _time_info, status):
            if status:
                print("Microphone status:", status)
            try:
                audio_queue.put_nowait(in_data)
            except queue.Full:
                pass  # 动作期间丢弃过期输入，避免语音缓存无限增长。
            return None, pyaudio.paContinue

        device = os.environ.get("ROBOT_MIC_DEVICE", "")
        stream = microphone.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True,
            input_device_index=int(device) if device else None,
            frames_per_buffer=4000, stream_callback=on_audio)
        cleanup.callback(stream.close)
        cleanup.callback(stream.stop_stream)
        robot = RobotMove(None)
        cleanup.callback(robot.close)
        stream.start_stream()
        print("Microphone control ready. Ctrl+C to stop.")
        while True:
            if not recognizer.AcceptWaveform(audio_queue.get()):
                continue
            text = json.loads(recognizer.Result()).get("text", "")
            action = command_for(text)
            if action:
                print("Recognized:", text, "->", action)
                if action == "DANCE":
                    import dance_main
                    dance_main.run(robot=robot)
                else:
                    robot.robotMove(action)
                while True:
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
