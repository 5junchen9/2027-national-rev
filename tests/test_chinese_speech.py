"""中文女声、WAV 和录音优先检查，不连接摄像头或串口。"""

from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import wave
import numpy as np

import chinese_speech as speech


class ChineseSpeechTests(unittest.TestCase):
    def test_face_volume_override_sets_speakers_and_enables_audio_route(self):
        import robot_audio
        with patch.object(robot_audio.subprocess, "run") as run:
            robot_audio.configure(volume=127)
        controls = {call.args[0][-2]: call.args[0][-1] for call in run.call_args_list}
        self.assertEqual(controls['name=Speaker Playback Volume'], '127')
        self.assertEqual(controls['name=Headphone Playback Volume'], '127')
        self.assertEqual(controls['name=PCM Playback -6dB Switch'], 'off')
        self.assertEqual(controls['name=Left Output Mixer PCM Playback Switch'], 'on')
        self.assertEqual(controls['name=Right Output Mixer PCM Playback Switch'], 'on')

    def test_chinese_and_valid_pcm(self):
        text = "张三同学，欢迎你；你好，我是小童。"
        audio = types.SimpleNamespace(samples=[0.0, -2.0, 1.0], sample_rate=22050)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(speech, "load_tts") as load, \
                patch.dict("os.environ", {"ROBOT_TTS_SPEED": "150"}):
            load.return_value.generate.return_value = audio
            output = Path(directory) / "speech.wav"
            self.assertEqual(speech.synthesize(text, output), "matcha-zh-baker female")
            load.return_value.generate.assert_called_once_with(text, sid=0, speed=1.0)
            with wave.open(str(output), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                                 (1, 2, 22050))
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            self.assertLessEqual(int(np.max(np.abs(pcm))), 29491)
            self.assertLess(pcm[1], 0)

    def test_invalid_audio_is_not_played(self):
        for samples in ([], [0.0], [float("nan")], [float("inf")]):
            with patch.object(speech, "load_tts") as load, \
                    patch.object(speech, "play_audio") as play, patch("builtins.print"):
                load.return_value.generate.return_value = types.SimpleNamespace(
                    samples=samples, sample_rate=22050)
                self.assertFalse(speech.speak_chinese("你好"))
                play.assert_not_called()

    def test_missing_model_has_no_mechanical_fallback(self):
        speech.load_tts.cache_clear()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(speech, "MODEL_DIR", Path(directory)), \
                patch.object(speech, "play_audio") as play, patch("builtins.print"):
            self.assertFalse(speech.speak_chinese("你好"))
            play.assert_not_called()
        speech.load_tts.cache_clear()

    def test_recording_and_failed_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "中文.wav"
            audio.write_bytes(b"recording")
            with patch.object(speech, "play_audio") as play, \
                    patch.object(speech, "synthesize") as synthesize:
                self.assertTrue(speech.speak_chinese("你好", audio))
                play.assert_called_once_with(audio)
                synthesize.assert_not_called()
            with patch.object(speech, "play_audio", side_effect=[OSError("failed"), None]), \
                    patch.object(speech, "synthesize", return_value="matcha-zh-baker female") as synthesize:
                self.assertTrue(speech.speak_chinese("你好", audio))
                synthesize.assert_called_once()

    def test_failed_engine_and_invalid_text(self):
        with patch.object(speech, "load_tts", side_effect=RuntimeError("missing")), \
                patch("builtins.print"):
            self.assertFalse(speech.speak_chinese("你好"))
        for text in ("", " \n", "字" * 1001, "你好\0"):
            with self.assertRaises(ValueError):
                speech.synthesize(text, "unused.wav")

    def test_existing_audio_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "original.wav"
            output.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "已存在"):
                speech.synthesize("你好", output)
            self.assertEqual(output.read_bytes(), b"original")

    def test_wm8960_wav_routing(self):
        with patch.object(speech.os, "name", "posix"), \
                patch.object(Path, "exists", return_value=True), \
                patch.object(speech.subprocess, "run") as run:
            speech.play_audio("sample.wav")
        self.assertEqual(run.call_args.args[0], [
            "aplay", "-q", "-D", "plughw:CARD=wm8960soundcard,DEV=0", "sample.wav"])

    def test_face_reports_playback_result(self):
        # 此检查只涉及播报返回值，无需加载摄像头依赖。
        with patch.dict("sys.modules", {"cv2": types.ModuleType("cv2")}):
            import face_main

        plugin = object.__new__(face_main.FaceRecognitionPlugin)
        with patch.object(face_main, "speak_chinese", return_value=False) as speak:
            self.assertFalse(plugin._welcome("haowenwan"))
        self.assertEqual(speak.call_args.args[0], "郝文菀同学，欢迎你")


if __name__ == "__main__":
    unittest.main()
