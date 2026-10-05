"""Edge Impulse 人脸分类模型的最小适配层。"""

from pathlib import Path

import cv2
from edge_impulse_linux.image import ImageImpulseRunner


from robot_config import ROOT
MODEL_PATH = ROOT / "assets/models/face/face.eim"


class FaceClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.runner = ImageImpulseRunner(str(model_path))
        try:
            self.info = self.runner.init()
        except BaseException:
            self.runner.stop()
            raise

    def classify_scores(self, bgr_image):
        """保留全部类别分数，便于用现场样本检查混淆与拒识阈值。"""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        features, _ = self.runner.get_features_from_image_auto_studio_settings(rgb_image)
        result = self.runner.classify(features)["result"]["classification"]
        if not result:
            raise RuntimeError("Model returned no classification scores")
        return {str(label): float(score) for label, score in result.items()}

    def classify(self, bgr_image):
        """保留原任务使用的最高标签和分数接口。"""
        return max(self.classify_scores(bgr_image).items(), key=lambda item: item[1])

    def close(self):
        self.runner.stop()
