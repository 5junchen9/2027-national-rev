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
        settings=camera_settings()
        frame=np.zeros((480,640,3),np.uint8)
        pickup=Box(256,240,64,48);drop=Box(256,144,128,96)
        eye,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        belly.getImage.return_value=(True,frame)
        servo.is_moving.return_value=False
        head_frame=frame.copy()
        eye.getImage.return_value=(True,head_frame)
        approached=False
        def show_belly_after_approach(action):
            nonlocal approached
            if action == 'UP_LITTLE': approached=True
        robot.robotMove.side_effect=show_belly_after_approach
        robot_factory=Mock(return_value=robot)
        reference=dict(version=2,cameras=settings,head_position=129,target_qr='DROP',
                       shapes=dict(head=[480,640],belly=[480,640]),**self.reference())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry_reference.json';path.write_text(json.dumps(reference))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[eye,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=lambda image,color: [Box(288,150,64,48)] if image is head_frame else ([pickup] if approached else [])), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('OTHER',Box(10,10,30,30)),('DROP',drop)]), \
                 patch.dict('sys.modules',{
                     'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                     'robotmove':types.SimpleNamespace(RobotMove=robot_factory)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(1000))):
                self.assertTrue(carry_vision.run('red','DROP',actions=True))
            self.assertEqual(json.loads(path.read_text()),reference)
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],['UP_LITTLE','HOLD_BOX','DOWN_BOX'])
        servo.turn_vertical.assert_called_once_with(129)
        robot.close.assert_called_once();servo.cleanup.assert_called_once();eye.close.assert_called_once();belly.close.assert_called_once()

    def test_missing_reference_does_not_open_camera_or_serial(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(carry_vision,'REFERENCE_FILE',Path(folder)/'missing.json'), \
                 patch.object(carry_vision,'RobotEye') as eye:
                with self.assertRaises(ValueError): carry_vision.run('red','DROP',actions=True)
                eye.assert_not_called()

    def test_preview_never_opens_body_serial(self):
        frame=np.zeros((480,640,3),np.uint8)
        eye,belly,servo=Mock(),Mock(),Mock();eye.getImage.return_value=belly.getImage.return_value=(True,frame)
        servo.is_moving.return_value=False
        factory=Mock()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(carry_vision,'REFERENCE_FILE',Path(folder)/'missing.json'), \
                 patch.object(carry_vision,'RobotEye',side_effect=[eye,belly]), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[]), \
                 patch.dict('sys.modules',{
                     'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                     'robotmove':types.SimpleNamespace(RobotMove=factory)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=ord('q')):
                self.assertFalse(carry_vision.run('red','DROP',actions=False))
        factory.assert_not_called()
        eye.close.assert_called_once();belly.close.assert_called_once();servo.cleanup.assert_called_once()

    def test_preview_saves_belly_pickup_and_head_qr_in_separate_dual_reference(self):
        hf=np.zeros((240,320,3),np.uint8)
        bf=np.zeros((480,640,3),np.uint8)
        head,belly,servo=Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry_dual_reference.json'
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=lambda image,color: [Box(256,240,64,48)] if image is bf else []), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(128,72,64,48))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=[-1]*6+[ord('h'),ord('d'),ord('q')]):
                self.assertFalse(carry_vision.run('blue','DROP'))
            reference=json.loads(path.read_text())
        self.assertEqual(reference['version'],2)
        self.assertEqual(reference['pickup'],[.4,.5,.1,.1])
        self.assertEqual(reference['drop'],[.4,.3,.2,.2])
        self.assertEqual(reference['shapes'],dict(head=[240,320],belly=[480,640]))
        head.close.assert_called_once();belly.close.assert_called_once()

    def test_pickup_search_moves_only_head_and_stops_when_block_appears(self):
        settings=camera_settings();frame=np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=belly.getImage.return_value=(True,frame)
        servo.is_moving.return_value=False
        def blocks(image,color):
            # 已发一个搜索小步后出现未稳定方块，立即停止新的头部搜索。
            return [Box(288,150,64,48)] if servo.begin_vertical.call_count else []
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry_dual_reference.json'
            path.write_text(json.dumps(dict(version=2,cameras=settings,head_position=129,target_qr='DROP',
                                           shapes=dict(head=[480,640],belly=[480,640]),**self.reference())))
            keys=iter([-1]*25+[ord('q')])
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                                          'robotmove':types.SimpleNamespace(RobotMove=lambda *args,**kwargs:robot)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=lambda _:next(keys)), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(1000))):
                self.assertFalse(carry_vision.run('blue','DROP',actions=True))
        servo.begin_vertical.assert_called_once_with(132,settle_seconds=.18)
        # 两个相机已看见，但尚未到腹部标定抱取位置，不发抱取指令。
        self.assertTrue(all(call.args[0] != 'HOLD_BOX' for call in robot.robotMove.call_args_list))
        self.assertTrue(all(call.args[0] not in ('TURN_LEFT','TURN_RIGHT') for call in robot.robotMove.call_args_list))

    def test_old_single_camera_reference_is_not_used_for_dual_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry_dual_reference.json'
            path.write_text(json.dumps(dict(version=1,pickup=[.4,.5,.1,.1],drop=[.4,.3,.2,.2])))
            with patch.object(carry_vision,'REFERENCE_FILE',path),patch.object(carry_vision,'RobotEye') as eye:
                with self.assertRaises(ValueError):carry_vision.run('blue','DROP',actions=True)
                eye.assert_not_called()
