import datetime
import os
from pathlib import Path
from robot_config import ROOT
import time
import threading

import cv2


CAMERA_DEVICE = os.environ.get("ROBOT_CAMERA_DEVICE", "0")
CAMERA_FPS = float(os.environ.get("ROBOT_CAMERA_FPS", "15"))
CAMERA_BACKEND = os.environ.get("ROBOT_CAMERA_BACKEND", "usb").lower()
CAMERA_FLIP = os.environ.get("ROBOT_CAMERA_FLIP", "none" if CAMERA_BACKEND == "csi" else "0")
if CAMERA_BACKEND not in ("csi", "usb") or CAMERA_FLIP not in ("none", "-1", "0", "1"):
    raise ValueError("Camera backend must be csi/usb; flip must be none/-1/0/1")


def camera_source():
    return int(CAMERA_DEVICE) if CAMERA_DEVICE.isdigit() else CAMERA_DEVICE


def usb_camera_source(device):
    """保留配置中的设备身份，只把实际打开方式转换为已存在节点的编号。"""
    if device.isdigit():
        return int(device)
    path = Path(device)
    if path.is_absolute() and path.exists():
        target = path.resolve()
        number = target.name.removeprefix('video')
        if target.parent == Path('/dev') and target.name.startswith('video') and number.isdigit():
            print(f'USB camera source: {device} -> {target} (index={number})', flush=True)
            return int(number)
    return device  # 不猜其他节点；错误路径仍按原路径报错。


class _CsiCamera:
    """为现有采集逻辑适配 Picamera2；USB仍使用OpenCV。"""
    def __init__(self, index, fps=CAMERA_FPS):
        if not isinstance(index, int):
            raise ValueError("CSI ROBOT_CAMERA_DEVICE must be a camera number, for example 0")
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError("Install python3-picamera2 with apt and use a "
                               "--system-site-packages venv") from error
        cameras = Picamera2.global_camera_info()
        if not 0 <= index < len(cameras):
            raise ValueError(f"Picamera2未发现腹部相机编号{index}，请先枚举相机，不要猜编号")
        # camera_num使用返回列表的位置；Num可能是排序前的内部编号。
        info = cameras[index]
        if 'usb' in info['Id'].lower() or 'uvc' in info['Model'].lower():
            raise ValueError(f"腹部csi编号{index}实际指向USB相机：{info['Id']}；"
                             "不能与头部重复打开，请枚举并选择真实CSI相机")
        self.camera = Picamera2(camera_num=index)
        self.running = False
        try:
            # libcamera RGB888 对应内存中的 BGR，直接兼容 OpenCV。
            config = self.camera.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameRate": fps}, buffer_count=4, queue=False)
            self.camera.configure(config)
            self.camera.start()
            self.running = True
        except BaseException:
            self.camera.close()
            raise

    def read(self):
        if not self.running:
            return False, None
        frame = self.camera.capture_array("main")
        return frame is not None and bool(frame.size), frame

    def isOpened(self):
        return self.running

    def grab(self):
        return self.read()[0]

    def release(self):
        if self.running:
            self.running = False
            try:
                self.camera.stop()
            finally:
                self.camera.close()


class _LatestCamera:
    """One reader owns the capture; consumers receive fresh, independently owned frames."""
    def __init__(self, camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.frame = None
        self.sequence = self.delivered = 0
        self.received_at = 0.0
        self.capture_fps = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        started, count = time.monotonic(), 0
        try:
            while not self.stopped.is_set():
                ok, frame = self.camera.read()
                if not ok or frame is None or not frame.size:
                    raise RuntimeError('Camera stream failed; stopping instead of replaying old frames')
                # Some camera backends reuse their arrays on the next read.
                frame = frame.copy()
                now = time.monotonic()
                with self.condition:
                    self.frame, self.received_at = frame, now
                    self.sequence += 1
                    count += 1
                    if now-started >= 1.0:
                        self.capture_fps = count/(now-started)
                        started, count = now, 0
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            self.camera.release()

    def read(self):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > self.delivered or self.error or self.stopped.is_set(), timeout=1.0)
            if self.error:
                raise RuntimeError('Camera capture thread failed') from self.error
            if not ready or self.stopped.is_set() or time.monotonic()-self.received_at > .5:
                raise RuntimeError('No fresh camera frame within timeout')
            self.delivered = self.sequence
            return True, self.frame.copy()

    def grab(self):
        # Continuously draining the driver already discards action-time frames.
        with self.condition:
            self.delivered = self.sequence
        return True

    def isOpened(self):
        return not self.stopped.is_set()

    def release(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError('Camera read did not stop within 2s')


class RobotEye:
    def __init__(self, device=None, backend=None, flip=None, fps=None, latest=False):
        # 每个实例独立配置，双摄不再通过改全局变量切换原任务的相机。
        self.device = CAMERA_DEVICE if device is None else str(device)
        self.backend = CAMERA_BACKEND if backend is None else backend
        self.flip = CAMERA_FLIP if flip is None else str(flip)
        self.fps = CAMERA_FPS if fps is None else float(fps)
        if self.backend not in ("csi", "usb") or self.flip not in ("none", "-1", "0", "1"):
            raise ValueError("Camera backend must be csi/usb; flip must be none/-1/0/1")
        if not 0 < self.fps <= 120:
            raise ValueError("Camera fps must be positive and <=120")
        self.__camera = None
        self.__read_failures = 0
        self._open_camera()
        if latest:
            self.__camera = _LatestCamera(self.__camera)

    def _open_camera(self):
        if self.__camera is not None:
            self.__camera.release()
            self.__camera = None
        if self.backend == 'usb':
            source = usb_camera_source(self.device)
        else:
            source = int(self.device) if self.device.isdigit() else self.device
        backend = cv2.CAP_V4L2 if os.name == "posix" else cv2.CAP_ANY

        for _ in range(2):
            if self.backend == "csi":
                camera = _CsiCamera(source, self.fps)
            else:
                camera = cv2.VideoCapture(source, backend)
                if not camera.isOpened() and os.name != "posix" and backend != cv2.CAP_ANY:
                    camera.release()
                    camera = cv2.VideoCapture(source)
            if camera.isOpened() and self.backend == "usb":
                # Restore the legacy camera's native format unless explicitly selected.
                fourcc = os.environ.get('ROBOT_CAMERA_FOURCC', 'native').upper()
                if fourcc not in ('NATIVE', 'YUYV', 'MJPG'):
                    camera.release()
                    raise ValueError('ROBOT_CAMERA_FOURCC must be native/YUYV/MJPG')
                if fourcc != 'NATIVE':
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, self.fps)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 4)
            self.__camera = camera
            try:
                if self._warm_up():
                    self.__read_failures = 0
                    if self.backend == 'usb':
                        actual = camera.get(cv2.CAP_PROP_FOURCC)
                        if isinstance(actual, (int, float)):
                            mode = ''.join(chr((int(actual) >> (8*i)) & 255) for i in range(4))
                            print(f'USB camera: {mode} {camera.get(cv2.CAP_PROP_FRAME_WIDTH):g}x'
                                  f'{camera.get(cv2.CAP_PROP_FRAME_HEIGHT):g} '
                                  f'negotiated={camera.get(cv2.CAP_PROP_FPS):g}fps requested={self.fps:g}fps')
                    return
            except BaseException:
                camera.release()
                self.__camera = None
                raise
            camera.release()
            self.__camera = None
            time.sleep(0.2)
        self.__camera = None
        raise RuntimeError("摄像头初始化失败：{} {}".format(self.backend, self.device))

    def _warm_up(self):
        """至少预热一秒并取得连续有效帧，避开启动绿屏和乱条。"""
        if not self.__camera.isOpened():
            return False
        started = time.monotonic()
        deadline = started + 2.5
        valid_frames = 0
        failed_frames = 0
        while time.monotonic() < deadline:
            ok, frame = self.__camera.read()
            if ok and frame is not None and frame.size:
                failed_frames = 0
                valid_frames += 1
                if valid_frames >= 5 and time.monotonic() - started >= 1.0:
                    return True
            else:
                valid_frames = 0
                failed_frames += 1
                if failed_frames >= 3:
                    return False  # 不对已掉线设备持续读到预热超时。
            time.sleep(0.02)
        return False

    def __getImage(self):
        if self.__camera is not None and self.__camera.isOpened():
            self.__camera.read()

    def timeSleep(self, ccSecond):
        deadline = datetime.datetime.now() + datetime.timedelta(seconds=ccSecond)
        while datetime.datetime.now() <= deadline:
            self.__getImage()

    def _read(self):
        if self.__camera is None or not self.__camera.isOpened():
            self._open_camera()
        ok, frame = self.__camera.read()
        if ok and frame is not None and frame.size:
            self.__read_failures = 0
            return True, frame

        self.__read_failures += 1
        if self.__read_failures >= 3:
            self._open_camera()
            ok, frame = self.__camera.read()
            if ok and frame is not None and frame.size:
                return True, frame
        return False, None

    def getImage(self):
        ok, frame = self._read()
        if not ok:
            return False, None
        return True, frame if self.flip == "none" else cv2.flip(frame, int(self.flip))

    def discard_frames(self, count=6):
        """丢弃机器人动作期间产生的旧帧。"""
        for _ in range(max(0, count)):
            if self.__camera is None or not self.__camera.isOpened():
                return
            self.__camera.grab()

    def stream_status(self):
        camera = self.__camera
        if isinstance(camera, _LatestCamera):
            return f'{camera.capture_fps:.1f}fps age={1000*(time.monotonic()-camera.received_at):.0f}ms'
        return 'synchronous'

    def getImage1(self):
        ok, frame = self._read()
        return (True, ok, cv2.flip(frame, -1)) if ok else (False, False, None)

    def showImage(self, ccImg):
        cv2.imshow("camera", ccImg)
        return cv2.waitKey(1)

    def close(self):
        if self.__camera is not None:
            self.__camera.release()
            self.__camera = None


if __name__ == "__main__":
    eye = RobotEye()
    try:
        while True:
            ret, image = eye.getImage()
            if ret:
                if eye.showImage(image) & 0xFF in (ord("q"), 27):
                    break
    finally:
        eye.close()
        cv2.destroyAllWindows()
