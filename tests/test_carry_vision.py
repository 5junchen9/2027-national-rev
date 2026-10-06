import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import carry_vision
from carry_vision import find_blocks, read_qr_codes, CarryPlanner, StableTarget, BlockTracker, remove_nested_boxes, merge_box_parts, block_quality
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

    def test_clipped_cyan_box_and_large_near_box_are_detected(self):
        frame = np.zeros((240,400,3),np.uint8)
        cv2.rectangle(frame,(100,80),(280,280),(255,255,0),-1)
        boxes = find_blocks(frame,'blue')
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0].bottom,240)
        self.assertEqual(find_blocks(frame,'red'),[])

    def test_notched_open_box_is_detected(self):
        frame = np.zeros((240,400,3),np.uint8)
        points = np.array([[100,40],[240,40],[240,190],[100,190],
                           [100,90],[155,90],[155,65],[100,65]])
        cv2.fillPoly(frame,[points],(255,255,0))
        self.assertEqual(len(find_blocks(frame,'blue')),1)

    def test_pickup_uses_position_instead_of_box_size(self):
        planner = CarryPlanner(self.reference(),'none')
        # 宽度不同但中心与下沿相同，仍在抱取位置。
        self.assertEqual(self.confirmed_action(planner,Box(.3,.3,.3,.3)),'HOLD_BOX')
        self.assertEqual(planner.decide(Box(.4,.65,.1,.1),5),'STOP')

    def test_default_drop_ignores_old_head129_reference(self):
        reference = self.reference();reference.pop('drop_head_position')
        planner = CarryPlanner(reference,'none');planner.phase = 'DELIVER'
        self.assertEqual(self.confirmed_action(planner,Box(.4,.3,.2,.2)),'UP_HOLDBOX')
        self.assertEqual(self.confirmed_action(planner,Box(.4,.7,.2,.15)),'DOWN_BOX')
        self.assertEqual(planner.decide(None,0),'WAIT')

    def test_near_large_box_accepts_top_clipping_and_visible_bottom(self):
        frame = np.zeros((480,640,3),np.uint8)
        cv2.rectangle(frame,(52,0),(639,470),(255,255,0),-1)
        boxes = find_blocks(frame,'blue',near=True)
        self.assertEqual(len(boxes),1)
        self.assertEqual(boxes[0].bottom,471)
        self.assertEqual(find_blocks(frame,'blue'),[])

    def test_head_floor_at_side_rejected_and_lid_body_merged(self):
        frame = np.zeros((480,640,3),np.uint8)
        cv2.rectangle(frame,(0,160),(220,320),(255,0,0),-1)
        cv2.rectangle(frame,(260,195),(348,221),(255,0,0),-1)
        cv2.rectangle(frame,(255,228),(352,307),(255,0,0),-1)
        boxes = find_blocks(frame,'blue')
        self.assertEqual(len(boxes),1)
        self.assertAlmostEqual(boxes[0].cx,304,delta=3)
        self.assertLess(boxes[0].y,200)
        self.assertGreater(boxes[0].bottom,300)

    def test_side_by_side_blocks_remain_ambiguous(self):
        frame = np.zeros((240,400,3),np.uint8)
        cv2.rectangle(frame,(60,70),(120,160),(255,0,0),-1)
        cv2.rectangle(frame,(140,70),(200,160),(255,0,0),-1)
        boxes = find_blocks(frame,'blue')
        self.assertEqual(len(boxes),2)
        self.assertIsNone(BlockTracker().update(boxes))

    def test_locked_block_ignores_new_distractor_but_not_two_near_matches(self):
        tracker = BlockTracker();box = Box(100,100,60,60)
        for _ in range(5): tracker.update([box])
        moved = Box(103,105,62,62)
        self.assertEqual(tracker.update([moved,Box(300,100,60,60)]),moved)
        self.assertGreaterEqual(tracker.frames,5)
        self.assertIsNone(tracker.update([moved,Box(125,105,62,62)]))
        self.assertEqual(tracker.frames,0)

    def test_missing_block_never_returns_old_box_and_eventually_unlocks(self):
        tracker = BlockTracker();tracker.update([Box(100,100,60,60)],now=0)
        for now in (.1,.2,.3,.4,.5):
            self.assertIsNone(tracker.update([],now=now))
        self.assertIsNone(tracker.box)
        tracker.update([Box(200,200,60,60)])
        tracker.after_body_move()
        self.assertEqual(tracker.frames,0)
        self.assertIsNone(tracker.update([]))

    def test_preview_saves_partial_pickup_with_clipped_edges(self):
        frame = np.zeros((480,640,3),np.uint8)
        head,belly,servo = Mock(),Mock(),Mock()
        head.getImage.return_value = belly.getImage.return_value = (True,frame)
        servo.is_moving.return_value = False
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json'
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',return_value=[Box(100,0,400,480)]), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=[-1]*6+[ord('h'),ord('q')]):
                carry_vision.run('blue','DROP')
            reference = json.loads(path.read_text())
            self.assertEqual(reference['pickup_mode'],'visible_region')
            self.assertEqual(reference['pickup_clipped'],[False,True,False,True])

    def test_approach_and_grasp_confirm_two_observations_without_double_wait(self):
        planner = CarryPlanner(self.reference(),'none')
        box = Box(.4,.2,.2,.2)
        self.assertEqual([planner.approach(box,frames) for frames in (1,2)],['WAIT','UP_LITTLE'])
        planner.reset_confirmation()
        pickup = Box(.4,.5,.1,.1)
        self.assertEqual([planner.decide(pickup,frames) for frames in (1,2)],['WAIT','HOLD_BOX'])
        planner.reset_confirmation()
        self.assertEqual(planner.decide(pickup,1),'WAIT')
        self.assertEqual(planner.decide(None,0),'WAIT')
        self.assertEqual(planner.decide(pickup,1),'WAIT')
        self.assertEqual(planner.decide(pickup,2),'HOLD_BOX')
        planner.reset_confirmation()
        self.assertEqual([planner.decide(box,frames) for frames in (1,2)],['WAIT','UP_LITTLE'])
        planner.phase = 'DELIVER';planner.reset_confirmation()
        drop = Box(.4,.3,.2,.2)
        self.assertEqual([planner.decide(drop,frames) for frames in range(1,6)],
                         ['WAIT','WAIT','WAIT','WAIT','DOWN_BOX'])

    def test_tiny_belly_border_patch_is_not_a_near_block(self):
        frame = np.zeros((480,640,3),np.uint8)
        cv2.rectangle(frame,(250,455),(330,479),(255,0,0),-1)
        self.assertEqual(find_blocks(frame,'blue',near=True),[])

    def test_near_transfer_limit_and_clipped_belly_takes_over_with_old_reference(self):
        for clipped_belly in (False,True):
            with self.subTest(clipped_belly=clipped_belly), tempfile.TemporaryDirectory() as folder:
                hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
                head,belly,servo,robot = Mock(),Mock(),Mock(),Mock()
                head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
                servo.is_moving.return_value=False
                def blocks(frame,color,near=False):
                    if near:
                        return [Box(200,350,200,130)] if clipped_belly else []
                    if not clipped_belly and robot.robotMove.call_count: return []
                    return [Box(200,320,240,160)]
                path=Path(folder)/'carry.json'
                path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),head_position=121,
                                                target_qr='DROP',shapes=dict(head=[480,640],belly=[480,640]),
                                                **self.reference())))
                with patch.object(carry_vision,'REFERENCE_FILE',path), \
                     patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                     patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                     patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(256,144,128,96))]), \
                     patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                     patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                     patch.object(cv2,'waitKey',side_effect=[-1]*120+[ord('q')]), \
                     patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.2 for i in range(3000))):
                    self.assertEqual(carry_vision.run('blue','DROP',actions=True,robot=robot),clipped_belly)
                sent = [call.args[0] for call in robot.robotMove.call_args_list]
                self.assertEqual(sent, ['HOLD_BOX','DOWN_BOX'] if clipped_belly else ['UP_LITTLE']*4)
                self.assertEqual([call.args[0] for call in servo.begin_vertical.call_args_list],
                                 [120] if clipped_belly else [])

    def test_transfer_crosses_missing_head_view_then_belly_grasps_and_delivers(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        def blocks(frame,color,near=False):
            forwards = sum(call.args[0] == 'UP_LITTLE' for call in robot.robotMove.call_args_list)
            if not near:
                return [Box(200,320,240,160)] if forwards == 0 else []
            return [Box(256,240,64,48)] if forwards >= 2 else []
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),head_position=121,
                                           target_qr='DROP',shapes=dict(head=[480,640],belly=[480,640]),
                                           **self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(256,144,128,96))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(3000))):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['UP_LITTLE','UP_LITTLE','HOLD_BOX','DOWN_BOX'])
        self.assertEqual([call.args[0] for call in servo.begin_vertical.call_args_list],[120])

    def test_visible_head_target_can_approach_five_steps_before_belly_takes_over(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        def blocks(frame,color,near=False):
            forwards = sum(call.args[0] == 'UP_LITTLE' for call in robot.robotMove.call_args_list)
            if not near:
                return [Box(200,320,240,160)] if forwards < 5 else []
            return [Box(256,240,64,48)] if forwards >= 5 else []
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),head_position=121,
                                           target_qr='DROP',shapes=dict(head=[480,640],belly=[480,640]),
                                           **self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(256,144,128,96))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(3000))):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['UP_LITTLE']*5+['HOLD_BOX','DOWN_BOX'])
        self.assertEqual([call.args[0] for call in servo.begin_vertical.call_args_list],[120])

    def test_small_low_head_target_does_not_trigger_blind_transfer_limit(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        def blocks(frame,color,near=False):
            forwards = sum(call.args[0] == 'UP_LITTLE' for call in robot.robotMove.call_args_list)
            if not near:
                return [Box(280,400,80,80)] if forwards < 5 else []
            return [Box(256,240,64,48)] if forwards >= 5 else []
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),head_position=121,
                                           target_qr='DROP',shapes=dict(head=[480,640],belly=[480,640]),
                                           **self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(256,144,128,96))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(3000))):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['UP_LITTLE']*5+['HOLD_BOX','DOWN_BOX'])
        self.assertEqual([call.args[0] for call in servo.begin_vertical.call_args_list],[120])

    def test_partial_reference_matches_only_current_local_region(self):
        reference = dict(pickup=[.3,.6,.3,.4],pickup_mode='visible_region',
                         pickup_clipped=[False,False,False,True])
        planner = CarryPlanner(reference,'none')
        self.assertEqual(self.confirmed_action(planner,Box(.3,.6,.3,.4)),'HOLD_BOX')
        self.assertEqual(planner.decide(None,0),'WAIT')
        # 裁切方式变化不再单独阻止抱取；位置和尺寸仍需在宽松范围内。
        self.assertEqual(self.confirmed_action(planner,Box(.3,.58,.3,.4)),'HOLD_BOX')
        self.assertEqual(self.confirmed_action(planner,Box(.345,.5,.21,.28)),'UP_LITTLE')

    def test_clipped_old_reference_does_not_require_same_size_or_high_shape_score(self):
        for box in [Box(.2,.6,.6,.4),Box(.3,.7,.4,.3),Box(.4,.9,.2,.1)]:
            with self.subTest(box=box):
                planner = CarryPlanner(self.reference(),'none')
                self.assertEqual(planner.decide(box,1),'WAIT')
                self.assertEqual(planner.decide(box,2),'HOLD_BOX')

    def test_copied_field_reference_and_screenshot_region_no_longer_wait(self):
        reference = dict(pickup=[.278125,.1604166667,.446875,.55625])
        planner = CarryPlanner(reference,'none')
        box = Box(218/640,366/480,286/640,114/480)
        self.assertEqual(planner.decide(box,1),'WAIT')
        self.assertEqual(planner.decide(box,2),'HOLD_BOX')
        # 更小的盒子到同一个可见底边位置，仍按位置而不是参考宽高判断。
        planner.reset_confirmation()
        small = Box(.4515625,.6366666667,.1,.08)
        self.assertEqual(planner.decide(small,1),'WAIT')
        self.assertEqual(planner.decide(small,2),'HOLD_BOX')

    def test_relaxed_reference_accepts_position_drift_but_stops_past_range(self):
        planner = CarryPlanner(self.reference(),'none')
        self.assertEqual(self.confirmed_action(planner,Box(.43,.43,.1,.1)),'HOLD_BOX')
        self.assertEqual(planner.decide(Box(.4,.7,.1,.1),2),'STOP')

    def test_dropout_retains_anchor_half_second_without_returning_it(self):
        tracker = BlockTracker();box = Box(100,100,60,60)
        tracker.update([box],now=10)
        self.assertIsNone(tracker.update([],now=10.2))
        self.assertEqual(tracker.box,box)
        self.assertEqual(tracker.update([Box(102,102,60,60)],now=10.3),Box(102,102,60,60))
        self.assertEqual(tracker.frames,1)
        self.assertIsNone(tracker.update([],now=10.81))
        self.assertIsNone(tracker.box)

    def test_partial_belly_handover_stays_locked_through_dropout(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        seen = 0
        def blocks(frame,color,near=False):
            nonlocal seen
            if not near: return [Box(288,150,64,48)]
            seen += 1
            if 4 <= seen <= 6: return []
            return [Box(192,288,192,192)]
        reference=dict(version=3,cameras=camera_settings(),head_position=121,target_qr='DROP',
                       shapes=dict(head=[480,640],belly=[480,640]),pickup=[.3,.6,.3,.4],
                       pickup_mode='visible_region',pickup_clipped=[False,False,False,True],
                       drop=[.4,.3,.2,.2],drop_head_position=120)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry.json';path.write_text(json.dumps(reference))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(256,144,128,96))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.02 for i in range(3000))):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
            self.assertEqual(json.loads(path.read_text()),reference)
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],['HOLD_BOX','DOWN_BOX'])
        servo.begin_vertical.assert_called_once_with(120)

    def test_head_move_keeps_identity_and_recovers_without_overlap(self):
        tracker=BlockTracker();box=Box(200,300,100,100)
        for now in (0,.1,.2): tracker.update([box],now=now)
        tracker.after_head_move()
        self.assertEqual(tracker.identity_box,box)
        self.assertEqual(tracker.frames,0)
        shifted=Box(205,140,100,100)
        distractor=Box(100,140,30,100)
        self.assertEqual(tracker.update([distractor,shifted],now=.3),shifted)
        self.assertEqual(tracker.frames,1)

    def test_body_move_keeps_identity_through_missed_frames(self):
        tracker=BlockTracker();box=Box(100,100,60,60)
        tracker.update([box],now=0);tracker.after_body_move()
        self.assertIsNone(tracker.update([],now=.7))
        self.assertIsNone(tracker.box)
        self.assertEqual(tracker.identity_box,box)
        shifted=Box(170,160,65,65)
        self.assertEqual(tracker.update([shifted,Box(350,100,60,60)],now=.8),shifted)
        self.assertEqual(tracker.frames,1)

    def test_recovery_uses_color_appearance_and_rejects_tied_candidates(self):
        tracker=BlockTracker();box=Box(100,100,60,60)
        frame=np.zeros((300,400,3),np.uint8)
        cv2.rectangle(frame,(100,100),(159,159),(255,0,0),-1)
        tracker.update([box],now=0,frame=frame);tracker.after_head_move()
        shifted=Box(105,200,60,60);wrong=Box(170,200,60,60)
        frame[:]=0
        cv2.rectangle(frame,(105,200),(164,259),(255,0,0),-1)
        cv2.rectangle(frame,(170,200),(229,259),(0,0,255),-1)
        self.assertEqual(tracker.update([wrong,shifted],now=.1,frame=frame),shifted)
        tracker.after_head_move()
        cv2.rectangle(frame,(125,200),(184,259),(255,0,0),-1)
        self.assertIsNone(tracker.update([shifted,Box(125,200,60,60)],now=.2,frame=frame))
        self.assertEqual(tracker.frames,0)

    def test_near_transfer_ignores_head_distractor_and_waits_for_delivery_servo(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        busy_frames=0
        def begin(angle,**kwargs):
            nonlocal busy_frames
            busy_frames=2
        def is_moving():
            nonlocal busy_frames
            busy = busy_frames > 0
            if busy: busy_frames -= 1
            return busy
        servo.begin_vertical.side_effect=begin
        servo.is_moving.side_effect=is_moving
        approached=False
        def moved(action):
            nonlocal approached
            if action == 'UP_LITTLE': approached=True
        robot.robotMove.side_effect=moved
        def blocks(frame,color,near=False):
            if near: return [Box(256,240,64,48)] if approached else []
            if not approached: return [Box(200,320,240,160)]
            return [Box(100,80,35,95),Box(220,170,200,130)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),head_position=129,
                                           target_qr='DROP',shapes=dict(head=[480,640],belly=[480,640]),
                                           **self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(256,144,128,96))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.02 for i in range(3000))):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['UP_LITTLE','HOLD_BOX','DOWN_BOX'])
        self.assertEqual([call.args[0] for call in servo.begin_vertical.call_args_list],[120])

    def test_initial_scores_choose_blue_over_dark_blue_black_and_strip(self):
        frame=np.zeros((240,420,3),np.uint8)
        blue=Box(40,70,100,110);dark=Box(190,70,100,110);strip=Box(340,70,25,110)
        # 蓝盒HSV与截图相近；黑盒有明显偏蓝，但亮度低。
        hsv=np.zeros_like(frame)
        hsv[70:180,40:140]=(105,161,159)
        hsv[70:180,190:290]=(123,122,64)
        hsv[70:180,340:365]=(105,161,159)
        frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker=BlockTracker()
        for now in (0,.1,.2):
            self.assertEqual(tracker.update([dark,strip,blue],now=now,frame=frame,color='blue'),blue)
        self.assertEqual(tracker.frames,3)
        scores={box:score for box,score,_,_ in tracker.scores}
        self.assertGreater(scores[blue],scores[dark])

    def test_equal_quality_blue_blocks_wait_and_dark_only_does_not_lock(self):
        hsv=np.zeros((240,420,3),np.uint8)
        first,second=Box(40,70,100,110),Box(190,70,100,110)
        hsv[70:180,40:140]=(105,161,159)
        hsv[70:180,190:290]=(105,161,159)
        frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker=BlockTracker()
        self.assertIsNone(tracker.update([first,second],frame=frame,color='blue'))
        self.assertIsNone(tracker.identity_box)
        hsv[70:180,40:140]=(123,122,64)
        frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        self.assertIsNone(tracker.update([first],frame=frame,color='blue'))
        self.assertIsNone(tracker.identity_box)

    def test_locked_blue_does_not_switch_to_new_brighter_box(self):
        hsv=np.zeros((240,420,3),np.uint8)
        old,new=Box(40,70,100,110),Box(160,70,100,110)
        hsv[70:180,40:140]=(105,161,159)
        frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker=BlockTracker();tracker.update([old],now=0,frame=frame,color='blue')
        tracker.after_body_move()
        hsv[70:180,160:260]=(105,220,240)
        frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        self.assertEqual(tracker.update([new,old],now=.1,frame=frame,color='blue'),old)

    def test_nested_candidates_collapse_but_separate_boxes_remain(self):
        outer,inner,other=Box(100,100,120,140),Box(110,100,40,20),Box(250,100,90,110)
        self.assertEqual(remove_nested_boxes([inner,outer,other]),[outer,other])
        tracker=BlockTracker()
        self.assertEqual(tracker.update([inner,outer]),outer)
        self.assertEqual(len(tracker.scores),1)
        self.assertIsNone(BlockTracker().update([outer,other]))

    def test_contained_lid_merges_even_when_widths_differ(self):
        # 外接框包含顶盖，但色轮廓不相连，顶盖远窄于盒身。
        body=np.array([[[100,100]],[[200,100]],[[200,220]],[[100,220]],[[100,140]],
                       [[150,140]],[[150,120]],[[100,120]]],np.int32)
        lid=np.array([[[110,123]],[[140,123]],[[140,136]],[[110,136]]],np.int32)
        self.assertEqual(len(merge_box_parts([body,lid])),1)

    def test_shadowed_locked_box_passes_when_initial_quality_is_low(self):
        box=Box(50,50,100,100)
        hsv=np.zeros((200,200,3),np.uint8)
        hsv[50:150,50:150]=(105,170,170)
        frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker=BlockTracker();tracker.update([box],now=0,frame=frame,color='blue')
        # 外观仍蓝，整体变暗到低于首次选择质量门槛。
        hsv[50:150,50:150]=(105,170,65)
        hsv[50:150,50:110]=(105,170,45)
        frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        c,shape=block_quality(frame,box,'blue')
        self.assertLess(.65*c+.35*shape,.45)
        self.assertIsNone(BlockTracker().update([box],frame=frame,color='blue'))
        tracker.after_body_move()
        self.assertEqual(tracker.update([box],now=.1,frame=frame,color='blue'),box)
        self.assertEqual(tracker.frames,1)

    def test_visible_blue_face_is_not_diluted_by_dark_side_and_white_label(self):
        box=Box(50,50,100,100)
        hsv=np.zeros((200,200,3),np.uint8)
        hsv[50:150,50:150]=(105,80,75)
        hsv[50:150,110:150]=(105,170,150)
        hsv[90:110,120:135]=(0,0,255)
        frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker=BlockTracker()
        self.assertEqual(tracker.update([box],frame=frame,color='blue'),box)

    def test_head_wrong_lock_expires_and_requires_fresh_confirmation(self):
        tracker = BlockTracker(recover_head=True)
        wrong = Box(10,10,20,20)
        actual = Box(200,250,100,100)
        tracker.update([wrong],now=0)
        tracker.after_body_move()
        self.assertIsNone(tracker.update([actual],now=.1))
        self.assertIn('mismatch',tracker.reason)
        self.assertIsNone(tracker.update([actual],now=2.0))
        self.assertEqual(tracker.identity_box,wrong)
        self.assertIsNone(tracker.update([actual],now=2.1))
        self.assertIsNone(tracker.identity_box)
        self.assertEqual(tracker.frames,0)
        for index in range(3):
            self.assertEqual(tracker.update([actual],now=2.2+index*.1),actual)
            self.assertEqual(tracker.frames,index+1)

    def test_belly_lock_is_not_replaced_after_head_recovery_timeout(self):
        tracker = BlockTracker()
        original = Box(10,10,40,40)
        tracker.update([original],now=0)
        self.assertIsNone(tracker.update([Box(300,300,80,80)],now=.1))
        self.assertIsNone(tracker.update([Box(300,300,80,80)],now=3))
        self.assertEqual(tracker.identity_box,original)

    def test_head_acquisition_prefers_box_over_small_bright_background(self):
        frame = np.zeros((480,640,3),np.uint8)
        small = Box(100,60,20,20)
        actual = Box(300,300,100,120)
        tracker = BlockTracker(recover_head=True)
        with patch.object(carry_vision,'block_quality',side_effect=[(.29,.74),(.29,.67)]):
            self.assertEqual(tracker.update([small,actual],now=0,frame=frame,color='blue'),actual)
        self.assertEqual(tracker.frames,1)

    def test_rejected_visible_candidate_does_not_block_head_search(self):
        frame = np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot = Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value = belly.getImage.return_value = (True,frame)
        servo.is_moving.return_value = False
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),
                target_qr='DROP',shapes=dict(head=[480,640],belly=[480,640]),
                **self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=lambda frame,color,near=False: [] if near else [Box(200,200,100,100)]), \
                 patch.object(BlockTracker,'update',return_value=None), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=[-1]*30+[ord('q')]), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(3000))):
                self.assertFalse(carry_vision.run('blue','DROP',actions=True,robot=robot))
        servo.begin_vertical.assert_called()
        robot.robotMove.assert_not_called()

    def test_qr_decoder_keeps_exact_content_and_positions(self):
        detector=Mock()
        corners=np.array([[[10,10],[50,10],[50,50],[10,50]],
                          [[100,10],[140,10],[140,50],[100,50]]],np.float32)
        detector.detectAndDecodeMulti.return_value=(True,['DROP','OTHER'],corners,[])
        codes=read_qr_codes(np.zeros((100,200,3),np.uint8),detector)
        self.assertEqual([text for text,_ in codes],['DROP','OTHER'])
        self.assertAlmostEqual(codes[0][1].cx,30,delta=1)

    def reference(self):
        return dict(pickup=[.4,.5,.1,.1],drop=[.4,.3,.2,.2],drop_head_position=120)

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

    def test_lost_target_and_action_limit_do_not_drop(self):
        planner=CarryPlanner(self.reference(),'0');planner.phase='DELIVER'
        self.assertEqual(planner.decide(None,0),'WAIT')
        self.assertEqual(self.confirmed_action(planner,Box(.3,.3,.4,.4)),'DOWN_BOX')
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
        near_qr=False
        def show_belly_after_approach(action):
            nonlocal approached,near_qr
            if action == 'UP_LITTLE': approached=True
            if action == 'UP_HOLDBOX': near_qr=True
        robot.robotMove.side_effect=show_belly_after_approach
        robot_factory=Mock(return_value=robot)
        reference=dict(version=2,cameras=settings,head_position=129,target_qr='DROP',
                       shapes=dict(head=[480,640],belly=[480,640]),**self.reference())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry_reference.json';path.write_text(json.dumps(reference))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[eye,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=lambda image,color,near=False: [Box(288,150,64,48)] if image is head_frame else ([pickup] if approached else [])), \
                 patch.object(carry_vision,'read_qr_codes',side_effect=lambda *args: [('OTHER',Box(10,10,30,30)),('DROP',drop if near_qr else Box(256,72,128,96))]), \
                 patch.dict('sys.modules',{
                     'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                     'robotmove':types.SimpleNamespace(RobotMove=robot_factory)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(1000))):
                self.assertTrue(carry_vision.run('red','DROP',actions=True))
            self.assertEqual(json.loads(path.read_text()),reference)
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],['UP_LITTLE','HOLD_BOX','UP_HOLDBOX','DOWN_BOX'])
        servo.turn_vertical.assert_called_once_with(123)
        servo.begin_vertical.assert_called_with(120)
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
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
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
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=lambda image,color,near=False: [Box(256,240,64,48)] if image is bf else []), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(128,72,64,48))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=[-1]*6+[ord('h'),ord('d')]+[-1]*6+[ord('d'),ord('q')]):
                self.assertFalse(carry_vision.run('blue','DROP'))
            reference=json.loads(path.read_text())
        self.assertEqual(reference['version'],3)
        self.assertEqual(reference['drop_head_position'],120)
        self.assertEqual(reference['pickup'],[.4,.5,.1,.1])
        self.assertEqual(reference['drop'],[.4,.3,.2,.2])
        self.assertEqual(reference['shapes'],dict(head=[240,320],belly=[480,640]))
        head.close.assert_called_once();belly.close.assert_called_once()

    def test_pickup_search_moves_only_head_and_stops_when_block_appears(self):
        settings=camera_settings();frame=np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=belly.getImage.return_value=(True,frame)
        servo.is_moving.return_value=False
        def blocks(image,color,near=False):
            # 已发一个搜索小步后出现未稳定方块，立即停止新的头部搜索。
            return [Box(288,150,64,48)] if servo.begin_vertical.call_count else []
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry_dual_reference.json'
            path.write_text(json.dumps(dict(version=2,cameras=settings,head_position=121,target_qr='DROP',
                                           shapes=dict(head=[480,640],belly=[480,640]),**self.reference())))
            keys=iter([-1]*25+[ord('q')])
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                                          'robotmove':types.SimpleNamespace(RobotMove=lambda *args,**kwargs:robot)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=lambda _:next(keys)), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(1000))):
                self.assertFalse(carry_vision.run('blue','DROP',actions=True))
        servo.begin_vertical.assert_called_once_with(126,settle_seconds=.18)
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
