# -*- coding: utf-8 -*-
"""离线中文女声播报：优先播放录音，动态文字使用 Matcha Baker。"""

import argparse
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import tempfile
import time
import wave


from robot_config import ROOT
MODEL_DIR = ROOT / "assets/models/tts/matcha-icefall-zh-baker"


@lru_cache(maxsize=1)
def load_tts():
    # 总控进程复用模型，避免每次播报都重新读取一百多 MB 的文件。
    files = ("model-steps-3.onnx", "vocos-22khz-univ.onnx", "lexicon.txt",
             "tokens.txt", "phone.fst", "date.fst", "number.fst")
    missing = [name for name in files if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError("中文女声模型缺少文件：{}；请复制完整目录 {}".format(
            ", ".join(missing), MODEL_DIR))
    try:
        import sherpa_onnx
    except ImportError as error:
        raise RuntimeError("缺少女声合成依赖；请在 venv 中执行 "
                           "python3 -m pip install sherpa-onnx==1.13.8") from error
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(MODEL_DIR / files[0]),
                vocoder=str(MODEL_DIR / files[1]),
                lexicon=str(MODEL_DIR / "lexicon.txt"),
                tokens=str(MODEL_DIR / "tokens.txt")),
            num_threads=2, provider="cpu"),
        rule_fsts=",".join(str(MODEL_DIR / name) for name in files[4:]),
        max_num_sentences=1)
    if not config.validate():
        raise RuntimeError("中文女声模型配置无效，请检查模型目录")
    return sherpa_onnx.OfflineTts(config)


@lru_cache(maxsize=1)
def warm_up():
    """提前加载并做一次无声合成；同一进程只执行一次，不播放提示音。"""
    started = time.monotonic()
    tts = load_tts()
    loaded = time.monotonic()
    tts.generate("你好", sid=0, speed=1.0)
    print(f"[语音预热] 模型加载={loaded-started:.2f}s，首次合成={time.monotonic()-loaded:.2f}s", flush=True)


def synthesize(text, output):
    text = text.strip()
    if not text or len(text) > 1000 or "\0" in text:
        raise ValueError("播报文字须为 1 至 1000 个字符，不能包含空字符")
    if Path(output).exists():
        raise ValueError("输出文件已存在，请换一个文件名，避免覆盖原录音")
    speed = int(os.environ.get("ROBOT_TTS_SPEED", "150"))
    if not 80 <= speed <= 300:
        raise ValueError("ROBOT_TTS_SPEED 须在 80 至 300 之间")
    import numpy as np

    audio = load_tts().generate(text, sid=0, speed=speed / 150)
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError("中文合成器生成了空音频或无效音频")
    peak = float(np.max(np.abs(samples)))
    if peak == 0:
        raise RuntimeError("中文合成器生成了静音")
    # 统一峰值上限，避免增大音量时 PCM 溢出产生破音。
    pcm = (samples * (0.9 * 32767 / peak)).astype("<i2")
    with wave.open(str(output), "wb") as output_audio:
        output_audio.setparams((1, 2, audio.sample_rate, 0, "NONE", "not compressed"))
        output_audio.writeframes(pcm.tobytes())
    return "matcha-zh-baker female"


def play_audio(path):
    from robot_audio import play
    play(path)


def speak_chinese(text, audio_path=None):
    """返回播放是否成功；已有录音优先，缺少或播放失败才合成文字。"""
    try:
        if audio_path and Path(audio_path).is_file():
            try:
                play_audio(audio_path)
                return True
            except (OSError, subprocess.SubprocessError) as error:
                print("[语音] 录音播放失败，尝试普通话合成：{}".format(error))
        with tempfile.TemporaryDirectory(prefix="robot-chinese-") as directory:
            output = Path(directory) / "speech.wav"
            started = time.monotonic()
            voice = synthesize(text, output)
            print(f"[语音] 合成耗时={time.monotonic()-started:.2f}s，开始播放", flush=True)
            print("[语音] 普通话音库：{}".format(voice))
            play_audio(output)
        return True
    except (OSError, ValueError, RuntimeError, EOFError, wave.Error,
            subprocess.SubprocessError) as error:
        print("[语音] 中文播报失败：{}".format(error))
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="你好，我是小童，中文播报测试。")
    parser.add_argument("--audio", help="优先播放的中文录音")
    parser.add_argument("--output", type=Path, help="只生成 WAV，便于检查，不播放声音")
    parser.add_argument("--check", action="store_true", help="无硬件自检，不播放或控制机器人")
    args = parser.parse_args()
    if args.check:
        import unittest
        from test_chinese_speech import ChineseSpeechTests

        result = unittest.TextTestRunner(verbosity=1).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ChineseSpeechTests)
        )
        return result.wasSuccessful()
    if args.output:
        try:
            voice = synthesize(args.text, args.output)
            print("已生成中文音频：{}，音库：{}".format(args.output, voice))
            return True
        except (OSError, ValueError, RuntimeError, EOFError, wave.Error,
                subprocess.SubprocessError) as error:
            print("中文合成失败：{}".format(error))
            return False
    return speak_chinese(args.text, args.audio)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
