"""CSI采集、颜色格式、资源释放及USB兼容检查，无硬件。"""
import types
import queue
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
import roboteye


class CameraTests(unittest.TestCase):
    def test_latest_stream_discards_backlog_and_copies_frames(self):
        incoming=queue.Queue()
        camera=Mock()
        camera.read.side_effect=lambda: incoming.get(timeout=2)
        stream=roboteye._LatestCamera(camera)
        try:
            shared=np.ones((2,3,3),np.uint8)
            for value in (1,2,3): incoming.put((True,shared*value))
            with stream.condition:
                self.assertTrue(stream.condition.wait_for(lambda:stream.sequence == 3,timeout=1))
            ok,frame=stream.read()
            self.assertTrue(ok)
            self.assertTrue((frame == 3).all())
            frame[:]=99
            self.assertTrue((stream.frame == 3).all())
            incoming.put((True,shared*4))
            self.assertTrue((stream.read()[1] == 4).all())
        finally:
            stream.stopped.set()
            incoming.put((False,None))
            stream.release()
        camera.release.assert_called_once()
        self.assertFalse(stream.thread.is_alive())

    def test_capture_failure_is_reported_instead_of_replaying_a_frame(self):
        camera=Mock()
        camera.read.return_value=(False,None)
        stream=roboteye._LatestCamera(camera)
        try:
            with self.assertRaisesRegex(RuntimeError,'capture thread failed'):stream.read()
        finally: stream.release()
        camera.release.assert_called_once()

    def test_stale_frame_is_rejected(self):
        incoming=queue.Queue()
        camera=Mock()
        camera.read.side_effect=lambda: incoming.get(timeout=2)
        stream=roboteye._LatestCamera(camera)
        try:
            incoming.put((True,np.ones((2,3,3),np.uint8)))
            with stream.condition:
                self.assertTrue(stream.condition.wait_for(lambda:stream.sequence == 1,timeout=1))
                stream.received_at=time.monotonic()-1
            with self.assertRaisesRegex(RuntimeError,'fresh camera frame'): stream.read()
        finally:
            stream.stopped.set()
            incoming.put((False,None))
            stream.release()
        camera.release.assert_called_once()

    def test_csi_bgr_frame_and_close(self):
        camera = Mock()
        frame = np.array([[[12, 34, 56]]], dtype=np.uint8)
        camera.capture_array.return_value = frame
        with patch.dict("sys.modules", {"picamera2": types.SimpleNamespace(Picamera2=lambda **kwargs: camera)}):
            capture = roboteye._CsiCamera(0)
            ok, actual = capture.read()
            self.assertTrue(ok)
            np.testing.assert_array_equal(actual, frame)
            self.assertEqual(camera.create_video_configuration.call_args.kwargs["main"]["format"], "RGB888")
            self.assertFalse(camera.create_video_configuration.call_args.kwargs["queue"])
            capture.release()
            capture.release()
        camera.stop.assert_called_once()
        camera.close.assert_called_once()
        self.assertFalse(capture.isOpened())

    def test_failed_configuration_closes_camera(self):
        camera = Mock()
        camera.configure.side_effect = RuntimeError("unsupported mode")
        with patch.dict("sys.modules", {"picamera2": types.SimpleNamespace(Picamera2=lambda **kwargs: camera)}):
            with self.assertRaises(RuntimeError):
                roboteye._CsiCamera(0)
        camera.close.assert_called_once()

    def test_csi_does_not_open_v4l2_and_preserves_frame(self):
        camera = Mock()
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        camera.read.return_value = (True, frame)
        with patch.object(roboteye, "CAMERA_BACKEND", "csi"), \
                patch.object(roboteye, "CAMERA_FLIP", "none"), \
                patch.object(roboteye, "_CsiCamera", return_value=camera), \
                patch.object(roboteye.cv2, "VideoCapture") as usb, \
                patch.object(roboteye.RobotEye, "_warm_up", return_value=True):
            eye = roboteye.RobotEye()
            ok, actual = eye.getImage()
            self.assertTrue(ok)
            np.testing.assert_array_equal(actual, frame)
            eye.discard_frames(2)
            eye.close()
            usb.assert_not_called()
        self.assertEqual(camera.grab.call_count, 2)
        camera.release.assert_called_once()

    def test_usb_backend_and_configured_flip(self):
        camera = Mock()
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        camera.read.return_value = (True, frame)
        with patch.object(roboteye, "CAMERA_BACKEND", "usb"), \
                patch.object(roboteye, "CAMERA_FLIP", "0"), \
                patch.object(roboteye.cv2, "VideoCapture", return_value=camera) as usb, \
                patch.object(roboteye, "_CsiCamera") as csi, \
                patch.object(roboteye.RobotEye, "_warm_up", return_value=True):
            eye = roboteye.RobotEye()
            np.testing.assert_array_equal(eye.getImage()[1], frame[::-1])
            eye.close()
            usb.assert_called_once()
            csi.assert_not_called()
            self.assertFalse(any(call.args[0] == roboteye.cv2.CAP_PROP_FOURCC
                                 for call in camera.set.call_args_list))
            camera.set.assert_any_call(roboteye.cv2.CAP_PROP_BUFFERSIZE,4)

    def test_warmup_exception_releases_camera(self):
        camera = Mock()
        with patch.object(roboteye, "CAMERA_BACKEND", "csi"), \
                patch.object(roboteye, "_CsiCamera", return_value=camera), \
                patch.object(roboteye.RobotEye, "_warm_up", side_effect=RuntimeError("capture failed")):
            with self.assertRaises(RuntimeError):
                roboteye.RobotEye()
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
