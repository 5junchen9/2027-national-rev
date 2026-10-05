import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import carry_vision
from carry_vision import find_blocks, read_qr_codes, CarryPlanner, StableTarget
from kick_shapes import Box
from dual_kick import camera_settings


class CarryVisionTests(unittest.TestCase):
    def test_selected_color_finds_one_block_among_equal_shapes(self):
        frame = np.zeros((240,400,3),np.uint8)
        for left,color in [(30,(0,0,255)),(150,(255,0,0)),(270,(0,255,0))]:
            cv2.rectangle(frame,(left,70),(left+60,130),color,-1)
        for color,cx in [('red',60),('blue',180),('green',300)]:
            boxes = find_blocks(frame,color)
            self.assertEqual(len(boxes),1)
            self.assertAlmostEqual(boxes[0].cx,cx,delta=2)

    def test_round_ball_and_large_map_region_are_not_blocks(self):
        frame = np.zeros((240,400,3),np.uint8)
        cv2.circle(frame,(200,100),35,(255,0,0),-1)
        self.assertEqual(find_blocks(frame,'blue'),[])
        frame[:] = (255,0,0)
        self.assertEqual(find_blocks(frame,'blue'),[])

    def test_multiple_same_color_blocks_clear_confirmation(self):
        tracker=StableTarget();box=Box(10,10,30,30)
        for _ in range(5): tracker.update([box])
        self.assertIsNone(tracker.update([box,Box(90,10,30,30)]))
        self.assertEqual(tracker.frames,0)

    def test_qr_decoder_keeps_exact_content_and_positions(self):
        detector=Mock()
        corners=np.array([[[10,10],[50,10],[50,50],[10,50]],
                          [[100,10],[140,10],[140,50],[100,50]]],np.float32)
        detector.detectAndDecodeMulti.return_value=(True,['DROP','OTHER'],corners,[])
        codes=read_qr_codes(np.zeros((100,200,3),np.uint8),detector)
        self.assertEqual([text for text,_ in codes],['DROP','OTHER'])
        self.assertAlmostEqual(codes[0][1].cx,30,delta=1)

    def reference(self):
        return dict(pickup=[.4,.5,.1,.1],drop=[.4,.3,.2,.2])

    def confirmed_action(self, planner, box):
        for _ in range(5): action=planner.decide(box,5)
        return action

    def test_pickup_deliver_release_once(self):
        planner=CarryPlanner(self.reference(),'0')
        pickup=Box(.4,.5,.1,.1)
        action=self.confirmed_action(planner,pickup)
        self.assertEqual(action,'HOLD_BOX');planner.mark_sent(action)
        self.assertEqual(planner.phase,'DELIVER')
        self.assertEqual(planner.decide(None,0),'WAIT')
        action=self.confirmed_action(planner,Box(.4,.3,.2,.2))
        self.assertEqual(action,'DOWN_BOX');planner.mark_sent(action)
        self.assertEqual(planner.decide(None,0),'DONE')

    def test_deliver_uses_holding_walk_and_turn_actions(self):
        planner=CarryPlanner(self.reference(),'0');planner.phase='DELIVER'
        self.assertEqual(self.confirmed_action(planner,Box(.425,.2,.15,.15)),'UP_HOLDBOX')
        self.assertEqual(self.confirmed_action(planner,Box(.7,.3,.2,.2)),'RIGHT_HOLDBOX')
        planner.flip='1'
        self.assertEqual(self.confirmed_action(planner,Box(.7,.3,.2,.2)),'LEFT_HOLDBOX')

    def test_lost_target_overshoot_and_action_limit_do_not_drop(self):
        planner=CarryPlanner(self.reference(),'0');planner.phase='DELIVER'
        self.assertEqual(planner.decide(None,0),'WAIT')
        self.assertEqual(planner.decide(Box(.3,.3,.4,.4),5),'STOP')
        planner.actions=40
        self.assertEqual(planner.decide(Box(.4,.3,.2,.2),5),'STOP')

    def test_real_entry_selects_destination_and_closes_resources(self):
        settings=camera_settings()['head']
        frame=np.zeros((480,640,3),np.uint8)
        pickup=Box(256,240,64,48);drop=Box(256,144,128,96)
        eye,servo,robot=Mock(),Mock(),Mock()
        eye.getImage.return_value=(True,frame)
        robot_factory=Mock(return_value=robot)
        reference=dict(version=1,camera=settings,head_position=129,target_qr='DROP',
                       shape=[480,640],**self.reference())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry_reference.json';path.write_text(json.dumps(reference))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',return_value=eye), \
                 patch.object(carry_vision,'find_blocks',return_value=[pickup]), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('OTHER',Box(10,10,30,30)),('DROP',drop)]), \
                 patch.dict('sys.modules',{
                     'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                     'robotmove':types.SimpleNamespace(RobotMove=robot_factory)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(1000))):
                self.assertTrue(carry_vision.run('red','DROP',actions=True))
            self.assertEqual(json.loads(path.read_text()),reference)
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],['HOLD_BOX','DOWN_BOX'])
        servo.turn_vertical.assert_called_once_with(129)
        robot.close.assert_called_once();servo.cleanup.assert_called_once();eye.close.assert_called_once()

    def test_missing_reference_does_not_open_camera_or_serial(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(carry_vision,'REFERENCE_FILE',Path(folder)/'missing.json'), \
                 patch.object(carry_vision,'RobotEye') as eye:
                with self.assertRaises(ValueError): carry_vision.run('red','DROP',actions=True)
                eye.assert_not_called()

    def test_preview_never_opens_body_serial(self):
        frame=np.zeros((480,640,3),np.uint8)
        eye,servo=Mock(),Mock();eye.getImage.return_value=(True,frame)
        factory=Mock()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(carry_vision,'REFERENCE_FILE',Path(folder)/'missing.json'), \
                 patch.object(carry_vision,'RobotEye',return_value=eye), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[]), \
                 patch.dict('sys.modules',{
                     'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                     'robotmove':types.SimpleNamespace(RobotMove=factory)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=ord('q')):
                self.assertFalse(carry_vision.run('red','DROP',actions=False))
        factory.assert_not_called()
        eye.close.assert_called_once();servo.cleanup.assert_called_once()
