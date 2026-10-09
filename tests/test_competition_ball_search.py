"""投放后找球：验证转一步看一次、找到稳定球就停止和有限搜索。"""
from itertools import count
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from competition_main import CompetitionIO, CONFIG_FILE, load_settings


class BallSearchTests(unittest.TestCase):
    def make_io(self):
        io = CompetitionIO(load_settings(CONFIG_FILE), Mock(), float('inf'))
        io.set_head = Mock()
        io.move = Mock()
        frame = np.zeros((480,640,3), np.uint8)
        io.observe_ready = Mock(return_value=(frame,frame,{}))
        return io

    def run_search(self, io, tracker, flip='none'):
        with patch('ball_debug.PatchTracker',return_value=tracker), \
                patch('competition_main.camera_settings',return_value={'head':{'flip':flip}}), \
                patch('competition_main.cv2.imshow'), \
                patch('competition_main.time.monotonic',side_effect=count()):
            io.find_ball()

    def test_missing_ball_turns_once_then_centered_ball_stops_turning(self):
        io = self.make_io()
        tracker = Mock(score=1,stable_frames=3)
        tracker.update.side_effect = [None]*3+[(300,200,40,40)]*3
        self.run_search(io,tracker)
        io.set_head.assert_called_once_with(129)
        io.move.assert_called_once_with('TURN_RIGHT')
        tracker.notify_body_move.assert_called_once()
        self.assertEqual(io.observe_ready.call_count,6)

    def test_center_band_including_boundaries_hands_over_without_turn(self):
        for x in (172,200,300,400,428):
            io = self.make_io()
            tracker = Mock(score=1,stable_frames=3)
            tracker.update.return_value = (x,200,40,40)
            self.run_search(io,tracker)
            io.move.assert_not_called()
            self.assertEqual(io.observe_ready.call_count,3)

    def test_ball_on_either_side_only_turns_right_until_centered(self):
        for x in (2,171,429,590):
            for flip in ('none','1','-1'):
                io = self.make_io()
                tracker = Mock(score=1, stable_frames=3)
                tracker.update.side_effect = [(x,200,40,40)]*3+[(300,200,40,40)]*3
                self.run_search(io, tracker, flip)
                io.move.assert_called_once_with('TURN_RIGHT')
                tracker.notify_body_move.assert_called_once()
                self.assertEqual(io.observe_ready.call_count,6)

    def test_center_confirmation_resets_when_ball_leaves_center_band(self):
        io = self.make_io()
        tracker = Mock(score=1,stable_frames=3)
        tracker.update.side_effect = [(300,200,40,40)]*2+[(590,200,40,40)]+[(300,200,40,40)]*3
        self.run_search(io,tracker)
        io.move.assert_not_called()
        self.assertEqual(io.observe_ready.call_count,6)

    def test_dropout_resets_confirmation_without_immediate_turn(self):
        io = self.make_io()
        tracker = Mock(score=1,stable_frames=3)
        tracker.update.side_effect = [(300,110,35,32)]*2+[None]+[(300,110,35,32)]*3
        self.run_search(io,tracker)
        io.move.assert_not_called()
        self.assertEqual(io.observe_ready.call_count,6)

    def test_unstable_ball_does_not_trigger_handover(self):
        io = self.make_io()
        tracker = Mock(score=1,stable_frames=2)
        tracker.update.return_value = (300,200,40,40)
        # 不稳定时只能等待，不能进入足球，也不能发转向。
        with self.assertRaisesRegex(RuntimeError,'超时'):
            self.run_search(io,tracker)
        io.move.assert_not_called()

    def test_no_ball_stops_at_turn_limit(self):
        io = self.make_io()
        tracker = Mock(score=1,stable_frames=3)
        tracker.update.return_value = None
        # 固定时间用于单独验证动作上限。
        with patch('ball_debug.PatchTracker',return_value=tracker), \
                patch('competition_main.cv2.imshow'), \
                patch('competition_main.time.monotonic',return_value=0):
            with self.assertRaisesRegex(RuntimeError,'转向30次'):
                io.find_ball()
        self.assertEqual(io.move.call_count,30)

    def test_ball_always_outside_center_stops_after_thirty_right_turns(self):
        io = self.make_io()
        tracker = Mock(score=1,stable_frames=3)
        tracker.update.return_value = (2,200,40,40)
        with patch('ball_debug.PatchTracker',return_value=tracker), \
                patch('competition_main.cv2.imshow'), \
                patch('competition_main.time.monotonic',return_value=0):
            with self.assertRaisesRegex(RuntimeError,'转向30次'):
                io.find_ball()
        self.assertEqual(io.move.call_count,30)
        self.assertTrue(all(call.args == ('TURN_RIGHT',) for call in io.move.call_args_list))

    def test_real_detector_accepts_centered_green_ball_without_turn(self):
        io = self.make_io()
        frame = np.zeros((480,640,3), np.uint8)
        cv2.circle(frame,(320,240),30,(0,255,100),-1)
        io.observe_ready.return_value = (frame,frame,{})
        with patch('competition_main.cv2.imshow'), \
                patch('competition_main.time.monotonic',side_effect=count()):
            io.find_ball()
        io.move.assert_not_called()

    def test_real_detector_recovers_wrong_color_after_search_turn(self):
        from ball_debug import PatchTracker
        io = self.make_io()
        hsv = np.full((480,640,3),(110,150,150),np.uint8)
        cv2.circle(hsv,(320,240),32,(35,25,225),-1)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        io.observe_ready.return_value = (frame,frame,{})
        tracker = PatchTracker()
        tracker.hue = 66
        tracker.color_learned = True
        self.run_search(io,tracker)
        io.move.assert_called_once_with('TURN_RIGHT')
        self.assertGreaterEqual(tracker.stable_frames,3)

    def test_camera_failure_prevents_turn(self):
        io = self.make_io()
        io.observe_ready.side_effect = RuntimeError('camera failed')
        with self.assertRaisesRegex(RuntimeError,'camera failed'):
            io.find_ball()
        io.move.assert_not_called()

    def test_dance_search_stops_on_target_and_scans_before_forward(self):
        io = self.make_io()
        io.stream = Mock()
        io.right = Mock()
        io.scan_until = Mock()
        with patch('competition_main.read_codes', side_effect=[[], [('other',None)], [('dance',None)]]):
            io.find_dance()
        self.assertEqual(io.right.call_count,2)
        io.scan_until.assert_called_once_with('dance','belly',confirm_frames=1)

    def test_dance_search_stops_after_nine_right_turns(self):
        io = self.make_io()
        io.stream = Mock()
        io.right = Mock()
        io.scan_until = Mock()
        with patch('competition_main.read_codes', return_value=[]):
            with self.assertRaisesRegex(RuntimeError,'右转9次'):
                io.find_dance()
        self.assertEqual(io.right.call_count,9)
        io.scan_until.assert_not_called()

    def test_belly_success_runs_exactly_eight_forward_steps(self):
        io = self.make_io()
        io.find_ball = Mock()
        io.stop_route = Mock()
        io.resume_route = Mock()
        io.forward = Mock()
        io.head_eye, io.belly_eye, io.servo = Mock(), Mock(), Mock()
        with patch('handover_debug.run',return_value=True) as sport:
            io.sport()
        self.assertTrue(sport.call_args.kwargs['finish_on_belly'])
        self.assertEqual(sport.call_args.kwargs['forward'],129)
        io.forward.assert_called_once_with(8)

    def test_search_failure_prevents_football_start(self):
        io = self.make_io()
        io.find_ball = Mock(side_effect=RuntimeError('ball missing'))
        io.stop_route = Mock()
        with patch('handover_debug.run') as football:
            with self.assertRaisesRegex(RuntimeError,'ball missing'):
                io.sport()
        football.assert_not_called()


if __name__ == '__main__':
    unittest.main()
