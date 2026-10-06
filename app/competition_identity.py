"""同学版本的人脸检测、FairFace性别与姓名OCR；使用比赛现有摄像头和头部驱动。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import time

import cv2
import robot_config  # 当前robot_env的驱动与声音配置；不改同学原文件。


def load_source(root, relative_path, module_name):
    """按完整路径读取同学模块，避免误导入robot_env的旧face_main。"""
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_models(root):
    source = load_source(root, "hardware/face_detector_dnn.py", "colleague_detector")
    ocr_source = load_source(root, "hardware/name_ocr.py", "colleague_ocr")
    models = root / "assets/models"
    detector = source.FaceDetectorDNN(
        proto=str(models / "face_detector/opencv_face_detector.pbtxt"),
        model=str(models / "face_detector/opencv_face_detector_uint8.pb"))
    gender = source.GenderClassifier(model=str(models / "gender/fairface.onnx"))
    ocr = ocr_source.NameOcr()
    engine = ocr.engine

    def read_text(region):
        output = engine(region)
        # RapidOCR 2.x无文字时返回txts=None；同学原解析器接受None。
        if hasattr(output, "txts") and output.txts is None:
            return None
        return output

    ocr.engine = read_text
    return detector, gender, ocr


def choose_face(faces):
    # 同学检测器按面积排序；从整个画面选最大的脸，不限制左右位置。
    return faces[0] if faces else None


def infer_identity(image, detector, gender, ocr, source):
    """一帧完整识别；只由后台线程调用，模型不会被并发使用。"""
    started = time.monotonic()
    face = choose_face(detector.detect(image))
    detected = time.monotonic()
    if face is None:
        return None, None, 0.0, None, (detected - started, 0.0, 0.0)
    x, y, width, height = (int(value) for value in face[:4])
    left, top, size = source.square_face_box(image.shape, (x, y, width, height))
    crop = image[top:top + size, left:left + size]
    gender_label, gender_score = gender.classify(crop)
    classified = time.monotonic()
    name, name_score = ocr.read_name(image, (x, y, width, height))
    finished = time.monotonic()
    timings = (detected - started, classified - detected, finished - classified)
    return name, gender_label, gender_score, name_score, timings


def recognize(root, position, timeout):
    from Head import RobotHeadServoOnly
    from roboteye import RobotEye
    from robot_audio import configure

    # 每次启动设置扬声器最大硬件音量、打开播放通路；不依赖.bashrc。
    configure(volume=127)
    print("[声音] 扬声器音量已设为127，开始识别姓名和性别。")

    # 只借用同学的裁剪、性别名称和播报；不调用其旧头部驱动或五人模型。
    source = load_source(root, "app/face_main.py", "colleague_face")
    confirm_frames = 3  # 不沿用robot_env旧五人识别的两帧设置。
    with ExitStack() as stack:
        detector, gender, ocr = load_models(root)
        for resource in (detector, gender, ocr):
            stack.callback(resource.close)
        eye = RobotEye(latest=True)
        stack.callback(eye.close)
        head = RobotHeadServoOnly(hold=True)
        stack.callback(head.cleanup)
        stack.callback(cv2.destroyAllWindows)
        head.turn_vertical(position)
        eye.discard_frames(1)
        # 先退出线程再释放摄像头和模型，避免仍在推理时关闭资源。
        worker = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        pending = None
        candidate, count = None, 0
        started = time.monotonic()
        deadline = started + timeout
        preview_started, preview_frames, preview_fps = started, 0, 0.0
        inference_text = "Inference: waiting"
        while time.monotonic() < deadline:
            ok, image = eye.getImage()
            if not ok:
                raise RuntimeError("人脸相机读取失败")
            # 用户确认姓名文字左右反了；在原安装方向翻转之上再水平翻转。
            # 显示、检测和OCR使用同一幅纠正后的图，不改其它阶段的标定。
            image = cv2.flip(image, 1)
            if pending is not None and pending.done():
                # 每个后台结果只消费一次，预览刷新的次数不参与连续确认。
                name, gender_label, gender_score, name_score, timings = pending.result()
                pending = None
                identity = (name, gender_label) if name and gender_label in source.GENDER_CN else None
                if identity is None:
                    candidate, count = None, 0
                elif identity == candidate:
                    count += 1
                else:
                    candidate, count = identity, 1
                detect_seconds, gender_seconds, ocr_seconds = timings
                inference_text = (f"D {detect_seconds:.2f}s / G {gender_seconds:.2f}s / "
                                  f"OCR {ocr_seconds:.2f}s")
                print(f"姓名={name} {name_score or 0:.0%} 性别={source.GENDER_CN.get(gender_label, '?')} "
                      f"{gender_score:.0%} 连续={count}/{confirm_frames}；"
                      f"检测={detect_seconds:.2f}s 性别={gender_seconds:.2f}s OCR={ocr_seconds:.2f}s")
                # 超时后返回的结果不能触发播报。
                if time.monotonic() >= deadline:
                    break
                if count >= confirm_frames:
                    text = f"{name}，{source.GENDER_CN[gender_label]}"
                    print("[语音]", text)
                    return source.speak_chinese(text)
            if pending is None:
                # 不排队：模型忙时继续预览；空闲时只提交当前最新帧。
                # 拷贝在绘字前完成，OCR不会读到预览状态文字。
                pending = worker.submit(infer_identity, image.copy(), detector, gender, ocr, source)
            preview_frames += 1
            now = time.monotonic()
            if now - preview_started >= 1.0:
                preview_fps = preview_frames / (now - preview_started)
                preview_started, preview_frames = now, 0
            cv2.putText(image, f"Preview {preview_fps:.1f} FPS | confirmed {count}/{confirm_frames}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            cv2.putText(image, inference_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            cv2.imshow("competition face", image)
            if cv2.waitKey(1) & 255 in (ord("q"), 27):
                return False
        print("姓名/性别未连续确认，停止本阶段。")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--head-position", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    raise SystemExit(0 if recognize(args.legacy_root.resolve(), args.head_position,
                                    args.timeout) else 1)
