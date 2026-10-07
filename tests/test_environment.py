"""迁移路径、音频路由及异常释放检查，不打开实机。"""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import robot_config
import robot_audio
import dreammaker_protocol as protocol
import qr_main


class EnvironmentTests(unittest.TestCase):
    def test_configuration_is_data_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.env"
            path.write_text('ROBOT_TTS_SPEED=140\nVOICE_SERIAL_PORT="" # auto\n', encoding="utf-8")
            self.assertEqual(robot_config.read_config(path),
                             {"ROBOT_TTS_SPEED": "140", "VOICE_SERIAL_PORT": ""})
            for content in ('HOME=/tmp\n', 'ROBOT_TTS_SPEED=140\nROBOT_TTS_SPEED=150\n',
                            'ROBOT_TTS_SPEED=150; echo broken\n'):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    robot_config.read_config(path)

    def test_mp3_and_wav_route_to_named_card(self):
        with patch.object(robot_audio.os, "name", "posix"):
            wav = robot_audio.playback_command("sample.wav")
            mp3 = robot_audio.playback_command("sample.mp3")
        self.assertEqual(wav, ["aplay", "-q", "-D", "plughw:CARD=wm8960soundcard,DEV=0", "sample.wav"])
        self.assertIn("alsa:device=plughw=wm8960soundcard.0", mp3)
        self.assertIn("-nolirc", mp3)

    def test_handshake_failure_releases_owned_serial(self):
        serial = Mock()
        controller = protocol.DreamMakerController(serial_port=serial, online_protocol=True)
        controller._owns_serial = True
        with patch.object(controller, "_open", side_effect=TimeoutError("missing FB")):
            with self.assertRaises(TimeoutError):
                controller.open()
        serial.close.assert_called_once()

    def test_qr_timeout_releases_camera(self):
        eye = Mock()
        with patch.dict("sys.modules", {
            "roboteye": types.SimpleNamespace(RobotEye=lambda: eye),
            "robotbarcode": types.SimpleNamespace(RobotBarcode=Mock())}), \
                patch.object(qr_main.time, "monotonic", side_effect=[0, 61]):
            with self.assertRaises(RuntimeError):
                qr_main.scan_route()
        eye.close.assert_called_once()

    def test_face_entry_propagates_current_identity_failure(self):
        import face_main
        with patch.object(face_main, "recognize", side_effect=RuntimeError("bad model")):
            with self.assertRaisesRegex(RuntimeError, "bad model"):
                face_main.run(quick=True)

    def test_microphone_commands_accept_asr_word_spacing(self):
        from voice_control import command_for
        self.assertEqual(command_for("请 往 前 走"), "UP_LITTLE")
        self.assertEqual(command_for("左 脚 踢 球"), "LEFT_BALL")
        self.assertIsNone(command_for("停止"))

    def test_installation_does_not_start_robot_or_enable_service(self):
        script = (robot_config.ROOT / "setup.sh").read_text(encoding="utf-8")
        self.assertNotIn("systemctl", script)
        self.assertNotIn("/etc/systemd", script)
        self.assertNotIn("main.py voice", script)
        self.assertNotIn("/boot/", script)


if __name__ == "__main__":
    unittest.main()
