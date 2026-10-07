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
    def test_original_h_trial_approaches_and_requires_exact_decode_to_release(self):
        reference = dict(pickup=[.4,.5,.1,.1])
        planner = CarryPlanner(reference,'none',use_original_drop=True)
        planner.phase = 'DELIVER'
        self.assertEqual(self.confirmed_action(planner,Box(.45,.3,.1,.1)),'UP_HOLDBOX')
        self.assertEqual(self.confirmed_action(planner,Box(.1,.5,.1,.1)),'LEFT_HOLDBOX')
        box = Box(.45,.5,.1,.1)
        self.assertEqual(planner.decide(box,3,decoded=False),'WAIT')
        self.assertEqual([planner.decide(box,3) for _ in range(3)],['WAIT','WAIT','DOWN_BOX'])
        self.assertEqual(reference,dict(pickup=[.4,.5,.1,.1]))

    def test_original_drop_option_still_prefers_valid_d(self):
        planner = CarryPlanner(self.reference(),'none',use_original_drop=True)
        planner.phase = 'DELIVER'
        self.assertEqual(self.confirmed_action(planner,Box(.48,.46,.04,.04)),'UP_HOLDBOX')

    def test_original_h_trial_passes_reference_gate_without_modifying_file(self):
        reference = dict(version=3,cameras=camera_settings(),target_qr='DROP',
                         shapes=dict(head=[480,640],belly=[480,640]),pickup=[.4,.5,.1,.1])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json';path.write_text(json.dumps(reference))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=RuntimeError('camera reached')):
                with self.assertRaisesRegex(RuntimeError,'camera reached'):
                    carry_vision.run('blue','DROP',actions=True,robot=Mock(),use_original_drop=True)
            self.assertEqual(json.loads(path.read_text()),reference)

    def test_small_qr_at_old_release_line_never_releases(self):
        planner = CarryPlanner(self.reference(),'none');planner.phase = 'DELIVER'
        # 与参考相同下沿，但二维码宽高仅20%，原下沿方案会直接放下。
        self.assertEqual(self.confirmed_action(planner,Box(.48,.46,.04,.04)),'UP_HOLDBOX')
        # 宽高不一致变化可能是斜拍或错误框，不能继续盲目接近。
        self.assertEqual(self.confirmed_action(planner,Box(.48,.3,.04,.2)),'WAIT')

    def test_placement_near_band_requires_alignment_and_exact_decode(self):
        planner = CarryPlanner(self.reference(),'none');planner.phase = 'DELIVER'
        box = Box(.41,.31,.18,.18)
        self.assertEqual([planner.decide(box,3) for _ in range(3)],['WAIT','WAIT','DOWN_BOX'])
        self.assertEqual(self.confirmed_action(planner,Box(.53,.3,.2,.2)),'RIGHT_HOLDBOX')
        self.assertEqual(planner.decide(box,3,decoded=False),'WAIT')
        self.assertEqual(planner.confirm_frames,0)
        self.assertEqual([planner.decide(box,3) for _ in range(3)],['WAIT','WAIT','DOWN_BOX'])

    def test_invalid_or_missing_d_prevents_camera_and_body_startup(self):
        for changes in ({'drop_head_position':129}, {'drop':None}, {'drop':[.4,.3,0,.2]}):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as folder:
                reference = dict(version=3,cameras=camera_settings(),target_qr='DROP',
                                 shapes=dict(head=[480,640],belly=[480,640]),**self.reference())
                reference.update(changes)
                path=Path(folder)/'carry.json';path.write_text(json.dumps(reference))
                robot=Mock()
                with patch.object(carry_vision,'REFERENCE_FILE',path),patch.object(carry_vision,'RobotEye') as eye:
                    with self.assertRaisesRegex(ValueError,'D参考'):
                        carry_vision.run('blue','DROP',actions=True,robot=robot)
                eye.assert_not_called();robot.robotMove.assert_not_called()

    def test_saving_d_with_changed_target_preserves_old_h_and_reference(self):
        frame=np.zeros((480,640,3),np.uint8)
        head,belly,servo=Mock(),Mock(),Mock()
        head.getImage.return_value=belly.getImage.return_value=(True,frame)
        servo.is_moving.return_value=False
        reference=dict(version=3,cameras=camera_settings(),target_qr='OLD',
                       shapes=dict(head=[480,640],belly=[480,640]),**self.reference())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry.json';path.write_text(json.dumps(reference))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',return_value=[]), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('NEW',Box(256,144,128,96))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kw:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=[ord('d')]+[-1]*6+[ord('d'),ord('q')]):
                self.assertFalse(carry_vision.run('blue','NEW'))
            self.assertEqual(json.loads(path.read_text()),reference)

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
        # 宽度不同但中心对齐且上沿到达横线，仍可抱取。
        self.assertEqual(self.confirmed_action(planner,Box(.3,.2,.3,.3)),'HOLD_BOX')
        self.assertEqual(self.confirmed_action(planner,Box(.4,.65,.1,.1)),'UP_LITTLE')

    def test_old_head129_drop_reference_stops_instead_of_using_default_line(self):
        reference = self.reference();reference.pop('drop_head_position')
        planner = CarryPlanner(reference,'none');planner.phase = 'DELIVER'
        self.assertEqual(self.confirmed_action(planner,Box(.4,.3,.2,.2)),'STOP')
        self.assertEqual(self.confirmed_action(planner,Box(.4,.7,.2,.15)),'STOP')
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
        box = Box(.4,.3,.2,.2)
        self.assertEqual([planner.approach(box,frames) for frames in (1,2)],['WAIT','UP_LITTLE'])
        planner.reset_confirmation()
        pickup = Box(.4,.2,.1,.1)
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
        self.assertEqual([planner.decide(drop,frames) for frames in range(1,4)],
                         ['WAIT','WAIT','DOWN_BOX'])

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
                        return [Box(200,72,200,408) if robot.robotMove.call_count else Box(200,384,200,96)] if clipped_belly else []
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
                self.assertEqual(sent, ['UP_LITTLE','HOLD_BOX','DOWN_BOX'] if clipped_belly else ['UP_LITTLE']*4)
                self.assertEqual([call.args[0] for call in servo.begin_vertical.call_args_list],
                                 [120] if clipped_belly else [129,135,140])

    def test_transfer_crosses_missing_head_view_then_belly_grasps_and_delivers(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        def blocks(frame,color,near=False):
            forwards = sum(call.args[0] == 'UP_LITTLE' for call in robot.robotMove.call_args_list)
            if not near:
                return [Box(200,320,240,160)] if forwards == 0 else []
            return [Box(256,72,64,48)] if forwards >= 2 else []
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
        self.assertEqual(servo.begin_vertical.call_args_list[-1].args[0],120)
        self.assertTrue(all(85 <= call.args[0] <= 180 for call in servo.begin_vertical.call_args_list))

    def test_qr_search_budget_resets_after_target_returns(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        def blocks(frame,color,near=False):
            return [Box(256,72,64,48)] if near else []
        clock = [0.0]
        first_missing = [None]
        def now():
            clock[0] += .1
            return clock[0]
        def moved(action):
            if action == 'RIGHT_HOLDBOX':
                self.assertIsNotNone(first_missing[0])
                self.assertGreaterEqual(clock[0]-first_missing[0],5.0)
                first_missing[0] = None
        robot.robotMove.side_effect = moved
        def read_qr(frame,detector):
            sent = [call.args[0] for call in robot.robotMove.call_args_list]
            turns = sent.count('RIGHT_HOLDBOX')
            if 'UP_HOLDBOX' not in sent and turns >= 2:
                return [('DROP',Box(288,144,64,48))]
            if 'UP_HOLDBOX' in sent and turns >= 5:
                return [('DROP',Box(256,336,128,96))]
            if first_missing[0] is None: first_missing[0] = clock[0]
            return []
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),head_position=121,
                                           target_qr='DROP',shapes=dict(head=[480,640],belly=[480,640]),
                                           pickup=[.4,.5,.1,.1],drop=[.4,.7,.2,.2],drop_head_position=120)))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'read_qr_codes',side_effect=read_qr), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=now):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['HOLD_BOX']+['RIGHT_HOLDBOX']*2+['UP_HOLDBOX']+['RIGHT_HOLDBOX']*3+['DOWN_BOX'])
        self.assertEqual(servo.begin_vertical.call_args_list[-1].args[0],120)
        self.assertTrue(all(85 <= call.args[0] <= 180 for call in servo.begin_vertical.call_args_list))

    def test_visible_head_target_can_approach_five_steps_before_belly_takes_over(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        def blocks(frame,color,near=False):
            forwards = sum(call.args[0] == 'UP_LITTLE' for call in robot.robotMove.call_args_list)
            if not near:
                return [Box(200,320,240,160)] if forwards < 5 else []
            return [Box(256,72,64,48)] if forwards >= 5 else []
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
        self.assertEqual(servo.begin_vertical.call_args_list[-1].args[0],120)
        self.assertTrue(all(85 <= call.args[0] <= 180 for call in servo.begin_vertical.call_args_list))

    def test_small_low_head_target_does_not_trigger_blind_transfer_limit(self):
        hf=np.zeros((480,640,3),np.uint8);bf=hf.copy()
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value=(True,hf);belly.getImage.return_value=(True,bf)
        servo.is_moving.return_value=False
        def blocks(frame,color,near=False):
            forwards = sum(call.args[0] == 'UP_LITTLE' for call in robot.robotMove.call_args_list)
            if not near:
                return [Box(280,400,80,80)] if forwards < 5 else []
            return [Box(256,72,64,48)] if forwards >= 5 else []
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
        self.assertEqual(servo.begin_vertical.call_args_list[-1].args[0],120)
        self.assertTrue(all(85 <= call.args[0] <= 180 for call in servo.begin_vertical.call_args_list))

    def test_partial_reference_matches_only_current_local_region(self):
        reference = dict(pickup=[.3,.6,.3,.4],pickup_mode='visible_region',
                         pickup_clipped=[False,False,False,True])
        planner = CarryPlanner(reference,'none')
        self.assertEqual(self.confirmed_action(planner,Box(.3,.6,.3,.4)),'UP_LITTLE')
        self.assertEqual(planner.decide(None,0),'WAIT')
        # 裁切方式变化不改变横线规则，上沿仍在横线下时继续接近。
        self.assertEqual(self.confirmed_action(planner,Box(.3,.58,.3,.4)),'UP_LITTLE')
        self.assertEqual(self.confirmed_action(planner,Box(.345,.5,.21,.28)),'UP_LITTLE')

    def test_clipped_old_reference_does_not_require_same_size_or_high_shape_score(self):
        for box in [Box(.2,.6,.6,.4),Box(.3,.7,.4,.3),Box(.4,.9,.2,.1)]:
            with self.subTest(box=box):
                planner = CarryPlanner(self.reference(),'none')
                self.assertEqual(planner.decide(box,1),'WAIT')
                self.assertEqual(planner.decide(box,2),'HOLD_BOX' if box.y <= .2 else 'UP_LITTLE')
                planner.reset_confirmation()
                close = Box(box.x,.15,box.width,.85)
                self.assertEqual(planner.decide(close,1),'WAIT')
                self.assertEqual(planner.decide(close,2),'HOLD_BOX')

    def test_copied_field_reference_and_screenshot_region_no_longer_wait(self):
        reference = dict(pickup=[.278125,.1604166667,.446875,.55625])
        planner = CarryPlanner(reference,'none')
        box = Box(218/640,366/480,286/640,114/480)
        self.assertEqual(planner.decide(box,1),'WAIT')
        self.assertEqual(planner.decide(box,2),'UP_LITTLE')
        # 更小的盒子到同一个可见底边位置，仍按位置而不是参考宽高判断。
        planner.reset_confirmation()
        small = Box(.4515625,.6366666667,.1,.08)
        self.assertEqual(planner.decide(small,1),'WAIT')
        self.assertEqual(planner.decide(small,2),'UP_LITTLE')

    def test_old_h_bottom_does_not_stop_belly_approach(self):
        planner = CarryPlanner(self.reference(),'none')
        self.assertEqual(self.confirmed_action(planner,Box(.43,.43,.1,.1)),'UP_LITTLE')
        self.assertEqual(self.confirmed_action(planner,Box(.4,.7,.1,.1)),'UP_LITTLE')

    def test_logged_98_8_percent_bottom_keeps_approaching_until_top_line(self):
        for reference in (
            dict(pickup=[.278125,.1604166667,.446875,.55625]),
            dict(pickup=[.278125,.1604166667,.446875,.55625],
                 pickup_mode='visible_region',pickup_clipped=[False]*4),
        ):
            with self.subTest(reference=reference):
                planner = CarryPlanner(reference,'none')
                logged = Box(.3,.262,.4,.726)
                self.assertEqual(self.confirmed_action(planner,logged),'UP_LITTLE')
                planner.mark_sent('UP_LITTLE')
                self.assertEqual(planner.belly_approach_steps,1)
                # 完整框的下沿仍未出画，但上沿到线后应抱；不再依赖旧H下沿。
                at_line = Box(.3,.20,.4,.788)
                self.assertEqual(self.confirmed_action(planner,at_line),'HOLD_BOX')

    def test_full_box_approach_keeps_six_step_limit_and_waits_on_missing_target(self):
        planner = CarryPlanner(self.reference(),'none')
        below = Box(.3,.262,.4,.726)
        for _ in range(6):
            self.assertEqual(self.confirmed_action(planner,below),'UP_LITTLE')
            planner.mark_sent('UP_LITTLE')
        self.assertEqual(planner.decide(None,0),'WAIT')
        self.assertEqual(planner.decide(below,2),'STOP')
        self.assertIn('6 belly approach steps',planner.reason)

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
            return [Box(192,72,192,408)] if robot.robotMove.call_count else [Box(192,384,192,96)]
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
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],['UP_LITTLE','HOLD_BOX','DOWN_BOX'])
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
            if near: return [Box(256,72,64,48)] if approached else []
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
        self.assertEqual(servo.begin_vertical.call_args_list[-1].args[0],120)
        self.assertTrue(all(85 <= call.args[0] <= 180 for call in servo.begin_vertical.call_args_list))

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

    def test_floor_like_head_region_does_not_beat_actual_box(self):
        frame = np.zeros((480,640,3),np.uint8)
        floor,box = Box(25,137,265,148),Box(495,239,130,151)
        tracker = BlockTracker(recover_head=True)
        with patch.object(carry_vision,'block_quality',side_effect=[(.56,.616),(.328,.753)]):
            self.assertEqual(tracker.update([floor,box],frame=frame,color='blue'),box)
        self.assertEqual(tracker.rejections[floor],'wide floor-like region')

    def test_partial_pickup_requires_upper_edge_at_line_and_has_step_limit(self):
        planner = CarryPlanner(self.reference(),'none')
        far = Box(.3,.74,.4,.26)
        for _ in range(6):
            self.assertEqual(planner.decide(far,1),'WAIT')
            self.assertEqual(planner.decide(far,2),'UP_LITTLE')
            planner.mark_sent('UP_LITTLE')
        self.assertEqual(planner.decide(far,2),'STOP')
        close = Box(.3,.2,.4,.8)
        planner.reset_confirmation()
        self.assertEqual(planner.decide(close,1),'WAIT')
        self.assertEqual(planner.decide(close,2),'HOLD_BOX')

    def test_belly_tracks_growing_clipped_box_and_reconfirms_before_pickup(self):
        frame = np.zeros((480,640,3),np.uint8)
        tracker = BlockTracker(near=True)
        old = Box(200,384,100,96)
        close = Box(0,72,600,408)
        with patch.object(carry_vision,'block_quality',return_value=(.84,.75)), \
             patch.object(carry_vision,'block_appearance',return_value=None):
            tracker.update([old],frame=frame,color='blue',now=0)
            tracker.after_body_move()
            self.assertEqual(tracker.update([close],frame=frame,color='blue',now=.1),close)
            planner = CarryPlanner(dict(pickup=[.278125,.16,.446875,.55625]),'none')
            self.assertEqual(planner.decide(close.normalized(frame.shape),tracker.frames),'WAIT')
            tracker.update([close],frame=frame,color='blue',now=.2)
            self.assertEqual(planner.decide(close.normalized(frame.shape),tracker.frames),'HOLD_BOX')

    def test_partial_belly_does_not_switch_to_distant_or_ambiguous_box(self):
        frame = np.zeros((480,640,3),np.uint8)
        tracker = BlockTracker(near=True)
        old = Box(200,384,100,96)
        tracker.update([old],frame=frame,now=0)
        tracker.after_body_move()
        self.assertIsNone(tracker.update([Box(510,100,100,380)],frame=frame,now=.1))
        self.assertIsNone(tracker.update([Box(160,170,130,310),Box(210,170,130,310)],
                                         frame=frame,now=.2))
        self.assertEqual(tracker.identity_box,old)
        self.assertEqual(tracker.frames,0)

    def test_partial_belly_still_rejects_wrong_color_appearance(self):
        frame = np.zeros((480,640,3),np.uint8)
        old = Box(200,384,100,96)
        frame[360:480,200:300] = (255,0,0)
        tracker = BlockTracker(near=True)
        tracker.update([old],frame=frame,now=0)
        tracker.after_body_move()
        frame[:] = (0,0,255)
        self.assertIsNone(tracker.update([Box(0,100,490,380)],frame=frame,now=.1))
        self.assertEqual(tracker.reason,'appearance mismatch')

    def test_belly_appearance_ignores_saturation_change(self):
        box = Box(200,72,200,408)
        hsv = np.zeros((480,640,3),np.uint8)
        hsv[120:480,200:400] = (105,95,170)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker = BlockTracker(near=True)
        tracker.update([box],frame=frame,color='blue',now=0)
        tracker.after_body_move()
        hsv[120:480,200:400] = (105,240,170)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        self.assertEqual(tracker.update([box],frame=frame,color='blue',now=.1),box)
        self.assertEqual(tracker.frames,1)
        tracker.update([box],frame=frame,color='blue',now=.2)
        self.assertEqual(tracker.frames,2)

    def test_strong_single_belly_region_reconfirms_changed_hue_before_grasp(self):
        box = Box(200,72,200,408)
        hsv = np.zeros((480,640,3),np.uint8)
        hsv[120:480,200:400] = (105,240,170)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker = BlockTracker(near=True)
        for now in (0,.1,.2): tracker.update([box],frame=frame,color='blue',now=now)
        hsv[120:480,200:400] = (125,240,170)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        self.assertEqual(tracker.update([box],frame=frame,color='blue',now=.3),box)
        self.assertEqual(tracker.frames,1)
        self.assertEqual(tracker.reason,'box appearance changed; reconfirm current region')
        planner = CarryPlanner(self.reference(),'none')
        self.assertEqual(planner.decide(box.normalized(frame.shape),tracker.frames),'WAIT')
        tracker.update([box],frame=frame,color='blue',now=.4)
        self.assertEqual(tracker.frames,2)
        self.assertEqual(planner.decide(box.normalized(frame.shape),tracker.frames),'HOLD_BOX')
        np.testing.assert_allclose(tracker.appearance,carry_vision.block_appearance(frame,box,'blue'))

    def test_changed_appearance_cannot_recover_multiple_or_weak_candidates(self):
        hsv = np.zeros((480,640,3),np.uint8)
        box = Box(200,72,200,408)
        hsv[120:480,200:400] = (105,240,170)
        old_frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        hsv[:] = (125,240,170)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker = BlockTracker(near=True)
        tracker.update([box],frame=old_frame,color='blue',now=0)
        appearance = tracker.appearance.copy()
        candidates = [Box(160,120,130,360),Box(210,120,130,360)]
        self.assertIsNone(tracker.update(candidates,frame=frame,color='blue',now=.1))
        np.testing.assert_array_equal(tracker.appearance,appearance)
        with patch.object(carry_vision,'block_quality',return_value=(.2,.8)):
            self.assertIsNone(tracker.update([box],frame=frame,color='blue',now=.2))
        np.testing.assert_array_equal(tracker.appearance,appearance)
        self.assertEqual(tracker.identity_box,box)

    def test_selected_belly_appearance_excludes_other_colored_background(self):
        box = Box(0,0,100,100)
        frame = np.zeros((100,100,3),np.uint8)
        frame[:] = (255,0,0)
        expected = carry_vision.block_appearance(frame,box,'blue')
        frame[:,50:] = (0,0,255)
        actual = carry_vision.block_appearance(frame,box,'blue')
        np.testing.assert_allclose(actual,expected)
        frame[:] = (0,0,255)
        self.assertIsNone(carry_vision.block_appearance(frame,box,'blue'))

    def test_belly_position_recovery_needs_three_current_observations(self):
        frame = np.full((480,640,3),(255,0,0),np.uint8)
        old = Box(450,380,100,100)
        current = Box(180,210,240,270)
        tracker = BlockTracker(near=True)
        tracker.update([old],frame=frame,color='blue',now=0)
        tracker.after_body_move()
        for now in (.1,.2):
            self.assertIsNone(tracker.update([current],frame=frame,color='blue',now=now))
            self.assertEqual(tracker.identity_box,old)
            self.assertEqual(tracker.frames,0)
        self.assertEqual(tracker.update([current],frame=frame,color='blue',now=.3),current)
        self.assertEqual(tracker.frames,3)
        planner = CarryPlanner(dict(pickup=[.278125,.16,.446875,.55625]),'none')
        self.assertEqual(planner.decide(current.normalized(frame.shape),tracker.frames),'WAIT')
        self.assertEqual(planner.decide(current.normalized(frame.shape),tracker.frames),'UP_LITTLE')

    def test_belly_position_recovery_resets_on_missing_multiple_or_weak_candidates(self):
        frame = np.full((480,640,3),(255,0,0),np.uint8)
        old = Box(450,380,100,100)
        current = Box(180,210,240,270)
        for interruption in ([],[current,Box(0,100,100,380)]):
            with self.subTest(interruption=interruption):
                tracker = BlockTracker(near=True)
                tracker.update([old],frame=frame,color='blue',now=0)
                self.assertIsNone(tracker.update([current],frame=frame,color='blue',now=.1))
                tracker.update(interruption,frame=frame,color='blue',now=.2)
                self.assertIsNone(tracker.update([current],frame=frame,color='blue',now=.3))
                self.assertIsNone(tracker.update([current],frame=frame,color='blue',now=.4))
                self.assertEqual(tracker.update([current],frame=frame,color='blue',now=.5),current)
        tracker = BlockTracker(near=True)
        tracker.update([old],frame=frame,color='blue',now=0)
        with patch.object(carry_vision,'block_quality',return_value=(.2,.8)):
            for now in (.1,.2,.3,.4):
                self.assertIsNone(tracker.update([current],frame=frame,color='blue',now=now))
        self.assertEqual(tracker.identity_box,old)

    def test_unaccepted_belly_candidate_does_not_block_reliable_head_approach(self):
        frame = np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot = Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value = belly.getImage.return_value = (True,frame)
        servo.is_moving.return_value = False
        original = BlockTracker.update
        def update(tracker,candidates,**kwargs):
            if tracker.near:
                tracker.reason = 'position mismatch'
                return None
            return original(tracker,candidates,**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),target_qr='DROP',
                shapes=dict(head=[480,640],belly=[480,640]),**self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=lambda image,color,near=False:
                              [Box(220,180,200,120)] if not near else [Box(200,380,100,100)]), \
                 patch.object(BlockTracker,'update',new=update), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=[-1]*20+[ord('q')]), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(3000))):
                self.assertFalse(carry_vision.run('blue','DROP',actions=True,robot=robot))
        actions = [call.args[0] for call in robot.robotMove.call_args_list]
        self.assertTrue(actions)
        self.assertEqual(set(actions),{'UP_LITTLE'})

    def test_belly_takeover_disables_head_boxes_and_grasp_disables_all_boxes(self):
        frame = np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot = Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value = belly.getImage.return_value = (True,frame)
        servo.is_moving.return_value = False
        def blocks(image,color,near=False):
            actions = [call.args[0] for call in robot.robotMove.call_args_list]
            self.assertNotIn('HOLD_BOX',actions,'delivery must not run block detection')
            if near:
                return [Box(192,72,192,408)] if actions else [Box(192,384,192,96)]
            self.assertFalse(actions,'belly takeover must disable head block detection')
            return [Box(220,180,200,120)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),target_qr='DROP',
                shapes=dict(head=[480,640],belly=[480,640]),**self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(256,144,128,96))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.02 for i in range(3000))):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['UP_LITTLE','HOLD_BOX','DOWN_BOX'])

    def test_partial_head_side_edge_keeps_box_but_rejects_full_screen(self):
        frame = np.full((480,640,3),100,np.uint8)
        cv2.rectangle(frame,(0,280),(250,479),(255,0,0),-1)
        self.assertEqual(len(find_blocks(frame,'blue')),1)
        frame[:] = (255,0,0)
        self.assertEqual(find_blocks(frame,'blue'),[])

    def test_two_steps_then_lost_view_recovered_by_head_before_next_step(self):
        frame = np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot = Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value = belly.getImage.return_value = (True,frame)
        servo.is_moving.return_value = False
        angle = 123
        forwards = 0
        def turn(value,**kwargs):
            nonlocal angle
            angle = value
        def move(action):
            nonlocal forwards
            if action == 'UP_LITTLE':
                if forwards == 2: self.assertGreaterEqual(angle,129)
                forwards += 1
        def blocks(image,color,near=False):
            if near:
                if forwards >= 4: return [Box(0,72,600,408)]
                if forwards >= 3: return [Box(200,384,100,96)]
                return []
            if forwards == 2 and angle < 129: return []
            return [Box(220,180,200,120)]
        servo.begin_vertical.side_effect = turn
        robot.robotMove.side_effect = move
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),target_qr='DROP',
                shapes=dict(head=[480,640],belly=[480,640]),**self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'block_quality',return_value=(.84,.75)), \
                 patch.object(carry_vision,'block_appearance',return_value=None), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[('DROP',Box(256,144,128,96))]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.02 for i in range(4000))):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['UP_LITTLE']*4+['HOLD_BOX','DOWN_BOX'])
        self.assertEqual([call.args[0] for call in servo.begin_vertical.call_args_list],[126,129,120])

    def test_pickup_line_at_upper_20_percent_boundary(self):
        reference = self.reference()
        self.assertEqual(carry_vision.pickup_action(Box(.3,.20,.4,.80),reference,'none'),'HOLD_BOX')
        self.assertEqual(carry_vision.pickup_action(Box(.3,.21,.4,.79),reference,'none'),'UP_LITTLE')
        self.assertEqual(carry_vision.pickup_action(Box(.3,.19,.4,.81),reference,'none'),'HOLD_BOX')
        self.assertEqual(carry_vision.follow_head_angle(Box(.3,.6,.4,.2),123),129)
        self.assertEqual(carry_vision.follow_head_angle(Box(.3,.1,.4,.2),123),120)
        self.assertEqual(carry_vision.follow_head_angle(Box(.3,.4,.4,.2),123),123)
        self.assertEqual(carry_vision.follow_head_angle(Box(.3,.6,.4,.2),140),140)

    def test_screenshot_upper_edge_at_74_percent_does_not_grasp(self):
        for reference in (self.reference(),dict(pickup=[.3,.6,.4,.4],
                         pickup_mode='visible_region',pickup_clipped=[False,False,False,True])):
            with self.subTest(reference=reference):
                planner = CarryPlanner(reference,'none')
                below = Box(.3,.74,.4,.26)
                self.assertEqual(planner.decide(below,8),'WAIT')
                self.assertEqual(planner.decide(below,9),'UP_LITTLE')
                planner.reset_confirmation()
                above = Box(.3,.15,.4,.85)
                self.assertEqual(planner.decide(above,1),'WAIT')
                self.assertEqual(planner.decide(above,2),'HOLD_BOX')

    def test_placement_requires_size_position_and_three_exact_decodes(self):
        planner = CarryPlanner(self.reference(),'none');planner.phase = 'DELIVER'
        close = Box(.4,.3,.2,.2)
        self.assertEqual([planner.decide(close,i) for i in (1,2,3)],['WAIT','WAIT','DOWN_BOX'])
        planner.reset_confirmation()
        self.assertEqual(planner.decide(close,5,decoded=False),'WAIT')
        self.assertEqual([planner.decide(close,5) for _ in range(3)],['WAIT','WAIT','DOWN_BOX'])
        self.assertEqual(self.confirmed_action(planner,Box(.46,.4,.08,.08)),'UP_HOLDBOX')
        self.assertEqual(self.confirmed_action(planner,Box(.4,.65,.2,.2)),'WAIT')
        self.assertEqual(self.confirmed_action(planner,Box(.3,.3,.4,.4)),'STOP')

    def test_lost_head_search_stays_between_120_and_140_and_observes_each_pose(self):
        frame = np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot = Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value = belly.getImage.return_value = (True,frame)
        servo.is_moving.return_value = False
        clock = [0.0]
        turns = []
        def now():
            clock[0] += .1
            return clock[0]
        def turn(angle,**kwargs):
            turns.append((angle,clock[0]))
        servo.begin_vertical.side_effect = turn
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),target_qr='DROP',
                shapes=dict(head=[480,640],belly=[480,640]),**self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',return_value=[]), \
                 patch.object(carry_vision,'read_qr_codes',return_value=[]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=[-1]*240+[ord('q')]), \
                 patch.object(carry_vision.time,'monotonic',side_effect=now):
                self.assertFalse(carry_vision.run('blue','DROP',actions=True,robot=robot))
        angles = [angle for angle,_ in turns]
        self.assertIn(120,angles)
        self.assertIn(140,angles)
        self.assertTrue(all(120 <= angle <= 140 for angle in angles))
        for (old,t0),(new,t1) in zip(turns,turns[1:]):
            self.assertLessEqual(abs(new-old),3)
            self.assertGreaterEqual(t1-t0,1.5)
        robot.robotMove.assert_not_called()

    def test_follow_move_resets_old_upward_search_direction(self):
        frame = np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot = Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value = belly.getImage.return_value = (True,frame)
        servo.is_moving.return_value = False
        turns = []
        target_visible = False
        followed = False
        done = False
        def turn(angle,**kwargs):
            nonlocal target_visible,followed,done
            turns.append(angle)
            if followed:
                done = True
            elif target_visible and angle == 129:
                target_visible = False
                followed = True
            elif angle == 123 and 140 in turns:
                target_visible = True
        servo.begin_vertical.side_effect = turn
        def blocks(image,color,near=False):
            return [Box(220,360,200,100)] if target_visible and not near else []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),target_qr='DROP',
                shapes=dict(head=[480,640],belly=[480,640]),**self.reference())))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=blocks), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=lambda _:ord('q') if done else -1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=iter(i*.1 for i in range(10000))):
                self.assertFalse(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertTrue(followed)
        self.assertEqual(turns[-3:],[123,129,132])
        robot.robotMove.assert_not_called()

    def test_delivery_observes_for_1_2_seconds_between_body_steps(self):
        frame = np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot = Mock(),Mock(),Mock(),Mock()
        head.getImage.return_value = belly.getImage.return_value = (True,frame)
        servo.is_moving.return_value = False
        clock = [0.0]
        sent = []
        def now():
            clock[0] += .02
            return clock[0]
        def moved(action):
            if sent and sent[-1][0] == 'UP_HOLDBOX':
                self.assertGreaterEqual(clock[0]-sent[-1][1],1.2)
            sent.append((action,clock[0]))
        def qr_codes(image,detector):
            steps = sum(action == 'UP_HOLDBOX' for action,_ in sent)
            return [('DROP',Box(256,168,128,144) if steps >= 2 else Box(288,120,64,72))]
        robot.robotMove.side_effect = moved
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'carry.json'
            path.write_text(json.dumps(dict(version=3,cameras=camera_settings(),target_qr='DROP',
                shapes=dict(head=[480,640],belly=[480,640]),pickup=[.4,.5,.1,.1],drop=[.4,.35,.2,.3],drop_head_position=120)))
            with patch.object(carry_vision,'REFERENCE_FILE',path), \
                 patch.object(carry_vision,'RobotEye',side_effect=[head,belly]), \
                 patch.object(carry_vision,'find_blocks',side_effect=lambda image,color,near=False:
                              [Box(256,72,64,48)] if near else []), \
                 patch.object(carry_vision,'block_quality',return_value=(1.0,1.0)), \
                 patch.object(carry_vision,'read_qr_codes',side_effect=qr_codes), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=-1), \
                 patch.object(carry_vision.time,'monotonic',side_effect=now):
                self.assertTrue(carry_vision.run('blue','DROP',actions=True,robot=robot))
        self.assertEqual([action for action,_ in sent],
                         ['HOLD_BOX','UP_HOLDBOX','UP_HOLDBOX','DOWN_BOX'])

    def test_extended_run_time_retains_five_minute_limit(self):
        planner = CarryPlanner(self.reference(),'none')
        planner.started_at = 0
        self.assertEqual(planner.decide(None,0,now=299),'WAIT')
        self.assertEqual(planner.decide(None,0,now=300),'STOP')
        self.assertEqual(planner.reason,'40 actions / 300 seconds limit')

    def test_qr_visual_tracking_requires_current_pattern_and_reconfirms_after_move(self):
        tracker = carry_vision.DestinationTracker()
        frame = np.zeros((160,240,3),np.uint8)
        pattern = np.random.default_rng(1).integers(0,256,(40,40,3),dtype=np.uint8)
        frame[40:80,50:90] = pattern
        tracker.update([Box(50,40,40,40)],frame=frame,now=0)
        tracker.after_move()
        moved = np.zeros_like(frame);moved[65:105,80:120] = pattern
        for frames in (1,2):
            self.assertEqual(tracker.update([],frame=moved,now=.1),Box(80,65,40,40))
            self.assertEqual(tracker.frames,frames)
            self.assertEqual(tracker.source,'tracked')
        self.assertIsNone(tracker.update([],frame=np.zeros_like(frame),now=.2))
        self.assertEqual(tracker.frames,0)
        self.assertIsNone(tracker.update([],frame=moved,now=4))
        self.assertIsNone(tracker.update([Box(50,40,40,40),Box(80,65,40,40)],frame=moved,now=4))

    def test_qr_template_does_not_choose_between_identical_patterns(self):
        tracker = carry_vision.DestinationTracker()
        frame = np.zeros((160,240,3),np.uint8)
        pattern = np.random.default_rng(2).integers(0,256,(40,40,3),dtype=np.uint8)
        frame[40:80,50:90] = pattern
        tracker.update([Box(50,40,40,40)],frame=frame,now=0)
        frame[100:140,160:200] = pattern
        self.assertIsNone(tracker.update([],frame=frame,now=.1))

    def test_far_qr_center_band_prevents_turning_at_small_jitter(self):
        planner = CarryPlanner(self.reference(),'none');planner.phase = 'DELIVER'
        self.assertEqual(self.confirmed_action(planner,Box(.45,.3,.1,.1)),'UP_HOLDBOX')
        planner.mark_sent('UP_HOLDBOX')
        self.assertEqual(self.confirmed_action(planner,Box(.49,.3,.1,.1)),'UP_HOLDBOX')

    def reference(self):
        return dict(pickup=[.4,.5,.1,.1],drop=[.4,.3,.2,.2],drop_head_position=120)

    def confirmed_action(self, planner, box):
        for _ in range(5): action=planner.decide(box,5)
        return action

    def test_pickup_deliver_release_once(self):
        planner=CarryPlanner(self.reference(),'0')
        pickup=Box(.4,.2,.1,.1)
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
        self.assertEqual(self.confirmed_action(planner,Box(.3,.3,.4,.4)),'STOP')
        planner.actions=40
        self.assertEqual(planner.decide(Box(.4,.3,.2,.2),5),'STOP')

    def test_real_entry_selects_destination_and_closes_resources(self):
        settings=camera_settings()
        frame=np.zeros((480,640,3),np.uint8)
        pickup=Box(256,72,64,48);drop=Box(256,144,128,96)
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
                 patch.object(carry_vision,'read_qr_codes',side_effect=lambda *args: [('OTHER',Box(10,10,30,30)),('DROP',drop if near_qr else Box(288,72,64,48))]), \
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
        servo.begin_vertical.assert_called_once_with(126,settle_seconds=.5)
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
