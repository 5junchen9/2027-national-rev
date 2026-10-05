"""Edge Impulse 人脸分类模型的最小适配层。"""

from pathlib import Path

import cv2
from edge_impulse_linux.image import ImageImpulseRunner


MODEL_PATH = Path(__file__).parent / "models" / "face" / "face.eim"


class FaceClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.runner = ImageImpulseRunner(str(model_path))
        self.info = self.runner.init()

    def classify(self, bgr_image):
        """返回模型预测的最高标签及其置信度。"""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        features, _ = self.runner.get_features_from_image_auto_studio_settings(rgb_image)
        result = self.runner.classify(features)["result"]["classification"]
        label, score = max(result.items(), key=lambda item: item[1])
        return label, score

    def close(self):
        self.runner.stop()
