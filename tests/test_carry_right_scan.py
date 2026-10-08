import unittest
import threading
import time
from unittest.mock import Mock, patch

import numpy as np

import carry_vision
from competition_route import run_course
from line_search_route import run_course as run_line_course
from dreammaker_protocol import load_dzz, motion_frame, apply_offsets, DEFAULT_INITIAL_POSITIONS
from robot_config import ROOT


class RightScanTests(unittest.TestCase):
    def scan(self, found_after=None, camera_failure=False, quit_now=False):
        robot = Mock()
        frame = np.zeros((480, 640, 3), np.uint8)
        stream = Mock()
        stream.read.return_value = (frame, frame, {})
        if camera_failure:
            stream.read.side_effect = RuntimeError('camera failed')
        stream.waiting_for_decode.return_value = False
        def found():
            actions = [call.args[0] for call in robot.robotMove.call_args_list]
            return found_after is not None and actions.count('SIDE_RIGHT_HOLDBOX') >= found_after
        stream.saw_target.side_effect = found
        servo = Mock();servo.is_moving.return_value = False
        reference = dict(shapes=dict(head=[480,640],belly=[480,640]))
        with patch('competition_route.RouteVision',return_value=stream) as factory, \
             patch('carry_vision.cv2.imshow'), \
             patch('carry_vision.cv2.waitKey',return_value=ord('q') if quit_now else -1), \
             patch('carry_vision.time.monotonic',side_effect=iter(range(10000))):
            try:
                result = carry_vision.scan_while_moving_right(
                    robot,Mock(),Mock(),servo,'action2',reference,Mock(),999,7)
            finally:
                stream.close.assert_called_once()
                stream.watch.assert_called_once_with('action2','belly')
                self.assertEqual(factory.call_args.args[-1],1)
        return result, [call.args[0] for call in robot.robotMove.call_args_list]

    def test_target_after_three_steps_releases_immediately_without_drop_reference(self):
        result, actions = self.scan(found_after=3)
        self.assertIs(result, True)
        self.assertEqual(actions,['SIDE_RIGHT_HOLDBOX']*3+['DOWN_BOX'])

    def test_tenth_step_target_is_checked_before_fallback(self):
        result, actions = self.scan(found_after=10)
        self.assertIs(result, True)
        self.assertEqual(actions,['SIDE_RIGHT_HOLDBOX']*10+['DOWN_BOX'])

    def test_limit_releases_once_before_turning_without_eleventh_step(self):
        result, actions = self.scan()
        self.assertEqual(result,carry_vision.SCAN_LIMIT)
        self.assertEqual(actions,['SIDE_RIGHT_HOLDBOX']*10+['DOWN_BOX']+['TURN_RIGHT']*7)

    def test_quit_stops_without_release_or_turn(self):
        result, actions = self.scan(quit_now=True)
        self.assertIs(result,False)
        self.assertEqual(actions,[])

    def test_camera_failure_does_not_trigger_release_or_turn(self):
        with self.assertRaisesRegex(RuntimeError,'camera failed'):
            self.scan(camera_failure=True)

    def test_real_background_decoder_runs_while_body_action_is_blocked(self):
        frame=np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        def image():
            time.sleep(.002)
            return True,frame
        head.getImage.side_effect=image;belly.getImage.side_effect=image
        servo.is_moving.return_value=False
        moving=threading.Event();decoded_during_move=threading.Event()
        def decode(frame):
            if moving.is_set() and not decoded_during_move.is_set():
                decoded_during_move.set()
                return [('action2',None)]
            return []
        def move(action):
            if action == 'DOWN_BOX':
                return
            self.assertEqual(action,'SIDE_RIGHT_HOLDBOX')
            moving.set()
            self.assertTrue(decoded_during_move.wait(1),'身体动作期间没有后台解码')
            # 仅第一次返回目标；验证一次成功解码就能触发放下。
            time.sleep(.03)
        robot.robotMove.side_effect=move
        reference=dict(drop_camera='belly',drop=[.3,.4,.2,.2],
                       shapes=dict(head=[480,640],belly=[480,640]))
        with patch('carry_vision.cv2.imshow'),patch('carry_vision.cv2.waitKey',return_value=-1), \
             patch('carry_vision.DELIVERY_OBSERVE_SECONDS',.02):
            result=carry_vision.scan_while_moving_right(robot,head,belly,servo,'action2',
                    reference,decode,time.monotonic()+3,7)
        self.assertIs(result,True)
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['SIDE_RIGHT_HOLDBOX','DOWN_BOX'])

    def test_both_courses_skip_sport_after_limit_and_keep_dance_sequence(self):
        import json
        settings=json.loads((ROOT/'config/competition.json').read_text())
        for course in (run_course,run_line_course):
            io=Mock();io.settings=settings;io.carry.return_value=carry_vision.SCAN_LIMIT
            self.assertTrue(course(io,'yellow'))
            io.sport.assert_not_called()
            io.scan_until.assert_any_call('dance','head',confirm_frames=2)
            io.enter_blue.assert_called_once();io.dance.assert_called_once()
            calls=[call[0] for call in io.method_calls]
            self.assertLess(calls.index('enter_blue'),calls.index('dance'))
            # 超限的右转已由搬运完成，路线不再追加足球后的7次转向。
            self.assertEqual(io.right.call_count,3 if course is run_course else 1)

    def test_new_side_step_keeps_grip_and_right_leg_trajectory(self):
        held=load_dzz(ROOT/'assets/actions/抱低处物块-减小前倾测试.dzz')[-1][0]
        source=load_dzz(ROOT/'assets/actions/右平移.dzz')
        new=load_dzz(ROOT/'assets/actions/搬运右平移-测试.dzz')
        self.assertEqual(len(new),len(source))
        for (offsets,ms),(original,original_ms) in zip(new,source):
            self.assertEqual(offsets[:7],held[:7])
            self.assertEqual(offsets[7:],original[7:])
            self.assertEqual(ms,original_ms)
            motion_frame(apply_offsets(DEFAULT_INITIAL_POSITIONS,offsets),ms)


if __name__ == '__main__':
    unittest.main()
