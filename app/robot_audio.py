"""所有提示语和背景音乐共用 WM8960 播放路径。"""
import os
from pathlib import Path
import subprocess
from robot_config import ROOT

CARD = os.environ.get("ROBOT_AUDIO_CARD", "wm8960soundcard")
VOLUME = int(os.environ.get("ROBOT_SPEAKER_VOLUME", "120"))
if not 0 <= VOLUME <= 127:
    raise ValueError("ROBOT_SPEAKER_VOLUME must be between 0 and 127")
SETTINGS = (
    ("Playback Volume", "255"),
    ("Speaker Playback Volume", str(VOLUME)),
    ("Headphone Playback Volume", str(VOLUME)),
    ("PCM Playback -6dB Switch", "off"),
    ("Left Output Mixer PCM Playback Switch", "on"),
    ("Right Output Mixer PCM Playback Switch", "on"),
)


def playback_command(path):
    if os.name == "posix" and Path(path).suffix.lower() == ".wav":
        return ["aplay", "-q", "-D", f"plughw:CARD={CARD},DEV=0", str(path)]
    command = ["mplayer", "-nolirc", "-really-quiet"]
    if os.name == "posix":
        # MPlayer 的 ALSA 子参数要求把 ':' 换成 '='，',' 换成 '.'。
        command += ["-ao", f"alsa:device=plughw={CARD}.0"]
    return command + [str(path)]


def play(path):
    subprocess.run(playback_command(path), check=True, timeout=180)


def start(path):
    return subprocess.Popen(playback_command(path))


def configure():
    for name, value in SETTINGS:
        subprocess.run(["amixer", "-c", CARD, "cset", f"name={name}", value],
                       check=True, timeout=10)
    return True
