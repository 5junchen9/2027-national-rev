"""颜色顺序、旧接口兼容以及跨丢脸间隔的错误确认回归。"""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import robot_config
import emotion_main
import vision_check


class VisionChecks(unittest.TestCase):
    def test_classifier_rgb_conversion_and_old_interface(self):
        runner = Mock()
        runner.get_features_from_image_auto_studio_settings.return_value = ([1], None)
        runner.classify.return_value = {"result": {"classification": {"happy": .8, "sad": .2}}}
        modules = {"edge_impulse_linux": types.ModuleType("edge_impulse_linux"),
                   "edge_impulse_linux.image": types.SimpleNamespace(ImageImpulseRunner=Mock(return_value=runner))}
        path = robot_config.ROOT / "hardware/face_eim.py"
        spec = importlib.util.spec_from_file_location("isolated_face_eim", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", modules):
            spec.loader.exec_module(module)
            classifier = module.FaceClassifier()
        self.assertEqual(classifier.classify(np.array([[[10, 20, 30]]], dtype=np.uint8)), ("happy", .8))
        actual = runner.get_features_from_image_auto_studio_settings.call_args.args[0]
        np.testing.assert_array_equal(actual, [[[30, 20, 10]]])
        runner.classify.return_value = {"result": {"classification": {}}}
        with self.assertRaises(RuntimeError):
            classifier.classify_scores(actual)
        classifier.close()
        runner.stop.assert_called_once()

    def test_lost_face_breaks_confirmation(self):
        plugin = emotion_main.EmotionRecognitionPlugin.__new__(emotion_main.EmotionRecognitionPlugin)
        plugin.cv2 = cv2
        plugin.eye = Mock()
        plugin.eye.getImage.return_value = (True, np.zeros((160, 160, 3), dtype=np.uint8))
        plugin.face_detector = Mock()
        plugin.face_detector.detectMultiScale.side_effect = [
            [(20, 20, 80, 80)], [(20, 20, 80, 80)], [], [(20, 20, 80, 80)]]
        plugin.classifier = Mock()
        plugin.classifier.classify.return_value = ("happy", .9)
        with patch.object(emotion_main.time, "monotonic", side_effect=[0, 1, 2, 3, 4, 7]):
            self.assertEqual(plugin.recognize_once(), "unknown")

    def test_single_image_evaluation_keeps_scores_and_task_rejection(self):
        classifier = Mock()
        classifier.classify_scores.return_value = {"happy": .51, "sad": .49}
        with patch.object(cv2, "imdecode", return_value=np.zeros((16, 16, 3), dtype=np.uint8)), \
                patch.object(np, "fromfile", return_value=np.array([0], dtype=np.uint8)):
            row = vision_check.evaluate(classifier, Path("sample.png"), "emotion")
        self.assertEqual(row["task_label"], "unknown")
        self.assertAlmostEqual(row["second_score_gap"], .02)
        self.assertEqual(row["scores"], {"happy": .51, "sad": .49})


if __name__ == "__main__":
    unittest.main()
