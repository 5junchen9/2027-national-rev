"""用模拟机器人执行现有任务调度，识别结果用样例代替，不访问硬件。"""
import types
from unittest.mock import patch
from robot_config import ROOT


def run():
    import voice_main as voice
    import carry_main
    import qr_main
    import dance_main
    import emotion_main
    import dual_kick
    actions = []
    robot = types.SimpleNamespace(robotMove=lambda action: actions.append(action))
    head = types.SimpleNamespace(level=lambda: None, look_down=lambda: None,
                                 cleanup=lambda: None)
    head_module = types.SimpleNamespace(RobotHeadServoOnly=lambda: head)
    classifier = types.SimpleNamespace(recognize_once=lambda: "happy", close=lambda: None)
    music = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None,
                                  wait=lambda timeout: None)

    def face(_robot):
        print("[SIMULATED FACE] haowenwan; vision accuracy is not tested")
        return True

    def qr(move):
        payload = "F2,L90,F1,R90"
        if not qr_main.run(source=payload, dry_run=True):
            return False
        qr_main.execute(None, qr_main.parse_route(payload), robot=move)
        return True

    def dual(preview=False, robot=None, **kwargs):
        print("[SIMULATED DUAL KICK] aligned sample only; camera geometry and goal accuracy are not tested")
        robot.robotMove("RIGHT_BALL")
        return True

    tasks = dict(voice.TASKS)
    tasks["1"] = ("simulated face", face, None)
    tasks["2"] = ("sample QR route", qr, None)
    with patch.dict("sys.modules", {"Head": head_module}), \
            patch.object(voice, "TASKS", tasks), \
            patch.object(voice, "play", side_effect=lambda path: print("[SIMULATED AUDIO]", path)), \
            patch.object(dance_main, "start", return_value=music), \
            patch.object(emotion_main, "EmotionRecognitionPlugin", return_value=classifier), \
            patch.object(dual_kick, "run", side_effect=dual), \
            patch.object(emotion_main, "play", return_value=None):
        for key in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "12", "13", "14", "15"):
            assert voice.run_task(key, robot), f"Simulation failed: {key}"
        with patch.object(classifier, "recognize_once", return_value="sad"):
            assert voice.run_task("13", robot)
    assert "HOLD_BOX" in actions and "DOWN_BOX" in actions
    import os
    expected_kick = "RIGHT_BALL" if os.environ.get("ROBOT_KICK_MODE", "fixed") == "dual" else "LEFT_BALL"
    assert expected_kick in actions and "DANCE2" not in actions
    print("Simulated actions:", " ".join(actions))
    print("Simulation passed: routing/action order only; hardware and vision are not emulated.")
    return True
