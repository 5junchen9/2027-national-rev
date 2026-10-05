"""形状混淆、双摄交接、失目标禁止动作和单次右踢回归。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np

import dual_kick
import roboteye
from kick_shapes import Box, observe, find_goals


def calibration():
    return {'version':1,'handover_y':.85,'goal_aim_x':.5,
            'near_y_direction':1,'handover_belly':[.5,.4,.06,.08],
            'right_foot_roi':[.6,.65,.7,.8],'ball_radius':.05,
            'head_shape':[480,640],'belly_shape':[480,640],
            'cameras':dual_kick.camera_settings()}


class DualKickTests(unittest.TestCase):
    def test_shapes_separate_rectangle_from_ball_and_merge_edges(self):
        frame = np.full((480,640,3),255,np.uint8)
        cv2.rectangle(frame,(160,50),(440,230),(0,0,0),7)
        cv2.circle(frame,(310,350),24,(0,0,0),-1)
        result = observe(frame)
        self.assertEqual((result['ball_count'],result['goal_count']),(1,1))
        self.assertAlmostEqual(result['ball'].cx,310,delta=3)
        self.assertAlmostEqual(result['goal'].cx,300,delta=8)

    def test_open_goal_and_multiple_balls(self):
        frame = np.full((480,640,3),255,np.uint8)
        cv2.line(frame,(150,230),(150,60),(0,0,0),7)
        cv2.line(frame,(150,60),(440,60),(0,0,0),7)
        cv2.line(frame,(440,60),(440,230),(0,0,0),7)
        cv2.circle(frame,(250,350),20,(0,0,0),-1)
        cv2.circle(frame,(390,350),20,(0,0,0),-1)
        result = observe(frame)
        self.assertEqual(result['goal_count'],1)
        self.assertEqual(result['ball_count'],2)
        self.assertIsNone(result['ball'])

    def test_handover_then_belly_alignment_then_one_right_kick(self):
        planner = dual_kick.KickPlanner(calibration(),confirm_frames=2)
        goal = Box(.3,.1,.4,.3)
        near_head = Box(.45,.77,.1,.1)
        self.assertEqual(planner.decide(near_head,goal),'WAIT')
        self.assertEqual(planner.decide(near_head,goal),'OPEN_BELLY')
        off_foot = Box(.3,.67,.1,.1)
        self.assertEqual(planner.decide(None,goal,off_foot),'WAIT')
        self.assertEqual(planner.decide(None,goal,off_foot),'SIDE_LEFT')
        at_foot = Box(.6,.67,.1,.1)
        self.assertEqual(planner.decide(None,goal,at_foot),'WAIT')
        self.assertEqual(planner.decide(None,goal,at_foot),'RIGHT_BALL')
        planner.mark_sent('RIGHT_BALL')
        self.assertEqual(planner.decide(None,goal,at_foot),'STOP')

    def test_wrong_heading_far_size_lost_goal_and_target_jump_do_not_kick(self):
        planner = dual_kick.KickPlanner(calibration(),confirm_frames=2)
        planner.phase = 'near'
        goal = Box(.3,.1,.4,.3)
        ball = Box(.6,.67,.1,.1)
        wrong_goal = Box(.55,.1,.4,.3)
        self.assertEqual(planner.decide(None,wrong_goal,ball),'WAIT')
        self.assertEqual(planner.decide(None,wrong_goal,ball),'TURN_RIGHT')
        self.assertEqual(planner.decide(None,None,ball),'WAIT')
        self.assertEqual(planner.decide(None,goal,ball),'WAIT')
        self.assertEqual(planner.decide(None,goal,Box(.64,.67,.1,.1)),'WAIT')
        far = Box(.625,.695,.03,.03)
        self.assertEqual(planner.decide(None,goal,far),'WAIT')
        self.assertEqual(planner.decide(None,goal,far),'UP_LITTLE')
        for _ in range(15): result = planner.decide(None,None,ball)
        self.assertEqual(result,'STOP')

    def test_uncalibrated_run_does_not_open_camera_or_robot(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(dual_kick,'RobotEye') as camera:
            with self.assertRaisesRegex(ValueError,'Calibration missing'):
                dual_kick.run(calibration_path=Path(directory)/'missing.json')
            camera.assert_not_called()

    def test_preview_closes_head_when_belly_initialization_fails(self):
        head = Mock()
        with patch.object(dual_kick,'RobotEye',side_effect=[head,RuntimeError('CSI failed')]), \
                patch.object(dual_kick.cv2,'destroyAllWindows'):
            with self.assertRaisesRegex(RuntimeError,'CSI failed'):
                dual_kick.run(preview=True)
        head.close.assert_called_once()

    def test_profiles_reject_missing_nonfinite_and_wrong_camera_settings(self):
        dual_kick.validate_calibration(calibration())
        for key, value in [('handover_y',None),('goal_aim_x',float('nan')),
                           ('right_foot_roi',[0,0,1,1]),('cameras',{})]:
            profile = deepcopy(calibration()); profile[key] = value
            with self.assertRaises(ValueError): dual_kick.validate_calibration(profile)

    def test_explicit_dual_camera_configs_do_not_change_original_defaults(self):
        camera = Mock(); frame = np.arange(18,dtype=np.uint8).reshape(2,3,3)
        camera.read.return_value = (True,frame)
        with patch.object(roboteye,'_CsiCamera',return_value=camera) as csi, \
                patch.object(roboteye.cv2,'VideoCapture') as usb, \
                patch.object(roboteye.RobotEye,'_warm_up',return_value=True):
            eye = roboteye.RobotEye(device=0,backend='csi',flip='none',fps=10)
            np.testing.assert_array_equal(eye.getImage()[1],frame)
            eye.close()
            csi.assert_called_once_with(0,10)
            usb.assert_not_called()
            self.assertEqual(roboteye.CAMERA_BACKEND,'usb')

    def test_measured_vertical_direction_handles_upside_down_belly(self):
        profile = calibration(); profile['near_y_direction'] = -1
        planner = dual_kick.KickPlanner(profile,confirm_frames=1); planner.phase = 'near'
        goal = Box(.3,.1,.4,.3)
        self.assertEqual(planner.decide(None,goal,Box(.6,.8,.1,.1)),'UP_LITTLE')
        self.assertEqual(planner.decide(None,goal,Box(.6,.5,.1,.1)),'BACK')

    def test_runner_handover_one_kick_and_cleanup_even_on_serial_failure(self):
        for fail in (False,True):
            head, belly, move = Mock(), Mock(), Mock()
            hf = np.full((480,640,3),255,np.uint8)
            cv2.rectangle(hf,(180,50),(460,230),(0,0,0),7)
            cv2.circle(hf,(320,390),24,(0,0,0),-1)
            bf = np.full((480,640,3),255,np.uint8)
            cv2.circle(bf,(416,348),32,(0,0,0),-1)
            head.getImage.side_effect = lambda: (True,hf.copy())
            belly.getImage.side_effect = lambda: (True,bf.copy())
            if fail: move.robotMove.side_effect = OSError('serial failed during kick')
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(dual_kick,'ROOT',Path(directory)), \
                    patch.object(dual_kick,'load_calibration',return_value=calibration()), \
                    patch.object(dual_kick,'RobotEye',side_effect=[head,belly]), \
                    patch.dict('sys.modules',{'robotmove':types.SimpleNamespace(RobotMove=lambda eye:move)}), \
                    patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',return_value=-1), \
                    patch.object(cv2,'destroyAllWindows'):
                if fail:
                    with self.assertRaisesRegex(OSError,'serial failed'):
                        dual_kick.run()
                else: self.assertTrue(dual_kick.run())
                logs = list((Path(directory)/'logs').glob('*.jsonl'))
                self.assertEqual(len(logs),1)
                self.assertIn('RIGHT_BALL',logs[0].read_text())
            move.robotMove.assert_called_once_with('RIGHT_BALL')
            move.close.assert_called_once()
            head.close.assert_called_once()
            belly.close.assert_called_once()

    def test_preview_does_not_create_robot_or_call_body_actions(self):
        head, belly, body_factory = Mock(),Mock(),Mock()
        hf = np.full((480,640,3),255,np.uint8)
        head.getImage.return_value = (True,hf.copy())
        belly.getImage.return_value = (True,hf.copy())
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(dual_kick,'RobotEye',side_effect=[head,belly]), \
                patch.dict('sys.modules',{'robotmove':types.SimpleNamespace(RobotMove=body_factory)}), \
                patch.object(cv2,'namedWindow'),patch.object(cv2,'setMouseCallback'), \
                patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',return_value=ord('q')), \
                patch.object(cv2,'destroyAllWindows'):
            self.assertTrue(dual_kick.run(preview=True,calibration_path=Path(directory)/'new.json'))
        body_factory.assert_not_called()
        head.close.assert_called_once(); belly.close.assert_called_once()


if __name__ == '__main__': unittest.main()
