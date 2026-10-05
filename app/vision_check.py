"""检查EIM元信息或已裁剪样本；不打开相机、串口、GPIO，不执行动作。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from robot_config import ROOT

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def task_label(model, label, score):
    if model == "face":
        from face_main import FACE_NAMES, FACE_CONFIDENCE
        return label if label in FACE_NAMES and score >= FACE_CONFIDENCE else "unknown"
    from emotion_main import decide_emotion
    return decide_emotion(label, score)


def evaluate(classifier, path, model):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read image: " + str(path))
    started = time.monotonic()
    scores = classifier.classify_scores(image)
    elapsed_ms = (time.monotonic() - started) * 1000
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    label, score = ranked[0]
    return {"file": str(path), "top_label": label, "top_score": score,
            "second_score_gap": score - ranked[1][1] if len(ranked) > 1 else None,
            "task_label": task_label(model, label, score), "scores": scores,
            "classification_ms": round(elapsed_ms, 2),
            "input_shape": list(image.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("face", "emotion"), required=True)
    samples = parser.add_mutually_exclusive_group()
    samples.add_argument("--image", type=Path, help="already cropped face image")
    samples.add_argument("--dataset", type=Path, help="label folders containing cropped test images")
    parser.add_argument("--output", type=Path, help="new JSON file; existing files are never overwritten")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output exists; choose a new filename")
    files = []
    if args.image:
        if not args.image.is_file():
            parser.error("Image file does not exist")
        files = [(None, args.image)]
    if args.dataset:
        if not args.dataset.is_dir():
            parser.error("Dataset directory does not exist")
        files = [(path.parent.name, path) for path in sorted(args.dataset.glob("*/*"))
                 if path.is_file() and path.suffix.lower() in SUFFIXES]
        if not files:
            parser.error("Dataset has no images in label subfolders")

    from face_eim import FaceClassifier
    model_path = ROOT / "assets/models" / args.model / (args.model + ".eim")
    classifier = FaceClassifier(model_path=model_path)
    try:
        report = {"model": args.model, "model_path": str(model_path),
                  "model_info": classifier.info,
                  "scope": "cropped single-image classification; no detector or temporal confirmation",
                  "samples": []}
        confusion = Counter()
        for expected, path in files:
            row = evaluate(classifier, path, args.model)
            row["expected"] = expected
            report["samples"].append(row)
            if expected is not None:
                confusion[(expected, row["task_label"])] += 1
        if args.dataset:
            correct = sum(count for (expected, predicted), count in confusion.items()
                          if expected == predicted)
            report["task_accuracy"] = correct / len(files)
            report["confusion"] = [{"expected": expected, "predicted": predicted, "count": count}
                                   for (expected, predicted), count in sorted(confusion.items())]
        payload = json.dumps(report, ensure_ascii=True, indent=2)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload + "\n")
            print("Saved:", args.output)
        else:
            print(payload)
        return True
    finally:
        classifier.close()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
