"""无串口、无摄像头、无音频设备的部署检查。"""
import ast
from pathlib import Path
import unittest
from robot_config import ROOT


def check():
    for directory in ("app", "hardware", "tests"):
        for path in (ROOT / directory).glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    import voice_main
    import emotion_main
    import carry_main
    import kick_main
    import serial_monitor
    import dreammaker_protocol as protocol
    from robotmove import ACTIONS
    from chinese_speech import MODEL_DIR

    voice_main.self_check()
    emotion_main.self_check()
    carry_main._check()
    kick_main._check()
    serial_monitor.self_check()
    # 使用当前已现场验证的 115200、25 对 FA、24 槽 FD 参数。
    assert protocol.SERVO_COUNT == 24
    assert protocol.request_frame(protocol.CMD_INITIAL_REQUEST) == b"\x00\xff" * 25 + b"\xfa"
    assert len(protocol.session_frame(protocol.CMD_OPEN)) == 51
    frames = 0
    for path in (ROOT / "assets/actions").glob("*.dzz"):
        for offsets, duration in protocol.load_dzz(path):
            positions = protocol.apply_offsets(protocol.DEFAULT_INITIAL_POSITIONS, offsets)
            frame = protocol.motion_frame(positions, protocol.scaled_duration_ms(duration))
            assert len(frame) == 51 and frame[-1] == protocol.CMD_LIVE_MOVE
            frames += 1
    assert frames > 0
    assert all(filename is None or (ROOT / "assets/actions" / filename).is_file()
               for filename, _ in ACTIONS.values())
    for name in ("model-steps-3.onnx", "vocos-22khz-univ.onnx", "lexicon.txt",
                 "tokens.txt", "phone.fst", "date.fst", "number.fst"):
        assert (MODEL_DIR / name).is_file(), f"Missing TTS asset: {name}"
    from competition_main import CONFIG_FILE, legacy_root, load_settings
    identity_root = legacy_root(load_settings(CONFIG_FILE))
    for asset in ("assets/models/gender/fairface.onnx",
                  "assets/models/face_detector/opencv_face_detector_uint8.pb",
                  "assets/models/face_detector/opencv_face_detector.pbtxt",
                  "hardware/name_ocr.py"):
        assert (identity_root / asset).is_file(), f"Missing identity asset: {asset}"
    assert (ROOT / "assets/music/dance.mp3").is_file()
    assert (ROOT / "assets/models/asr/vosk-model-small-cn-0.22/am/final.mdl").is_file()
    from voice_control import command_for
    assert command_for("往 前 走") == "UP_LITTLE"
    assert command_for("停止") is None
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(f"Validated {frames} motion frames; no hardware was opened.")
    return result.wasSuccessful()
