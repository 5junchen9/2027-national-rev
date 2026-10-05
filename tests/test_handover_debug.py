import unittest
from pathlib import Path
import cv2
import json
import tempfile
import types
from unittest.mock import Mock,patch
import numpy as np
from copy import deepcopy
from handover_debug import Handover, at_right_foot, foot_match_details, validate
from ball_debug import PatchTracker
import handover_debug
from dual_kick import camera_settings


class HandoverTests(unittest.TestCase):
    def test_head_passes_old_down_limit_and_returns_after_belly_confirmation(self):
        s = Handover(dict(forward=119,handover=129,down_sign=1,bounds=[85,137]))
        self.assertIsNone(s.step(None,0,480))
        below=(100,350,40,40)
        angles = [value for i in range(150) if (value:=s.step(below,0,480,now=i*.2)) is not None]
        self.assertEqual(angles,list(range(121,180,2))+[180])
        self.assertIsNone(s.step(None,0,480))
        self.assertIsNone(s.step(None,5,480))
        self.assertIsNone(s.step(None,5,480))
        self.assertEqual(s.step(None,5,480),119)
        self.assertEqual(s.phase,'BELLY')
        self.assertIsNone(s.step(None,5,480))

    def test_center_and_lost_ball_do_not_continue_down_sequence(self):
        s=Handover(dict(forward=117,handover=127,down_sign=1,bounds=[85,137]))
        for i in range(3): s.step((100,350,40,40),0,480,now=i*.25)
        self.assertEqual(s.angle,119)
        # Allow the short median window to replace old off-center measurements.
        for i in range(3): s.step((100,220,40,40),0,480,now=.6+i*.01)
        for _ in range(20): self.assertIsNone(s.step((100,220,40,40),0,480))
        for _ in range(20): self.assertIsNone(s.step(None,0,480))
        self.assertEqual(s.angle,119)
        self.assertEqual(s.phase,'HEAD')

    def test_upward_ball_raises_head_and_jitter_does_not_move(self):
        s=Handover(dict(forward=117,handover=127,down_sign=1,bounds=[85,137]))
        for i in range(3): s.step((100,60,40,40),0,480,now=i*.25)
        self.assertEqual(s.angle,114)
        for i,box in enumerate([(100,350,40,40),(100,60,40,40)]*10):
            self.assertIsNone(s.step(box,0,480,now=1+i*.1))
        self.assertEqual(s.angle,114)

    def test_fast_frames_and_single_outlier_do_not_trigger_moves(self):
        s=Handover(dict(forward=117,handover=127,down_sign=1,bounds=[85,137]))
        for i in range(3): self.assertIsNone(s.step((100,350,40,40),0,480,now=i*.02))
        s.reset_follow()
        center=(100,220,40,40)
        for i in range(5): s.step(center,0,480,now=.1+i*.1)
        self.assertIsNone(s.step((100,60,40,40),0,480,now=.7))
        self.assertEqual(s.angle,117)

    def test_reverse_direction_waits_but_can_respond_within_half_second(self):
        s=Handover(dict(forward=117,handover=127,down_sign=1,bounds=[85,137]))
        for i in range(3): s.step((100,350,40,40),0,480,now=i*.06)
        self.assertEqual(s.angle,119)
        for i in range(5):
            self.assertIsNone(s.step((100,60,40,40),0,480,now=.18+i*.06))
        self.assertEqual(s.step((100,60,40,40),0,480,now=.53),116)

    def test_fast_follow_clamps_at_down_limit_and_does_not_move_on_loss(self):
        s=Handover(dict(forward=119,handover=129,down_sign=1,bounds=[85,137]))
        for i in range(240): s.step((100,420,40,40),0,480,now=i/30)
        self.assertEqual(s.angle,180)  # 底层指令上限，旧137不再限制。
        for i in range(30): self.assertIsNone(s.step(None,0,480,now=1+i/30))
        self.assertEqual(s.angle,180)

    def test_saved_partial_box_matches_but_missing_ball_or_wrong_position_does_not(self):
        ref=dict(visible_patch_box=[25/640,0,257/640,249/480],clipped_edges=['top'])
        self.assertTrue(at_right_foot((25,0,257,249),(480,640),ref,['top']))
        self.assertFalse(at_right_foot(None,(480,640),ref,['top']))
        self.assertTrue(at_right_foot((25,0,257,249),(480,640),ref,[]))
        self.assertTrue(at_right_foot((28,22,252,239),(480,640),ref,[]))
        self.assertFalse(at_right_foot((25,65,257,249),(480,640),ref,[]))
        self.assertFalse(at_right_foot((300,0,257,249),(480,640),ref,['top']))

    def test_actual_near_top_frame_matches_without_clipped_edge_flag(self):
        frame=cv2.imread(str(Path(__file__).parent/'fixtures/tennis_foot_near_top.png'))
        self.assertIsNotNone(frame)
        tracker=PatchTracker()
        ref=dict(visible_patch_box=[25/640,0,257/640,249/480],clipped_edges=['top'])
        for _ in range(10):
            box=tracker.update(frame)
            self.assertTrue(at_right_foot(box,frame.shape[:2],ref,tracker.edges))
        self.assertGreaterEqual(tracker.stable_frames,5)
        self.assertEqual(tracker.edges,[])

    def test_failed_geometry_explains_the_actual_mismatch(self):
        ref=dict(visible_patch_box=[25/640,0,257/640,249/480],clipped_edges=['top'])
        info=foot_match_details((300,0,257,249),(480,640),ref,['top'])
        self.assertFalse(info['match'])
        self.assertEqual(info['reason'],'X position')
        self.assertIn('X err=+275px',info['detail'])

    def test_expanded_region_accepts_moderate_offsets_but_not_far_ball(self):
        ref=dict(visible_patch_box=[25/640,0,257/640,249/480],clipped_edges=['top'])
        self.assertTrue(at_right_foot((55,20,257,249),(480,640),ref,[]))
        self.assertTrue(at_right_foot((0,0,330,249),(480,640),ref,['top']))
        self.assertFalse(at_right_foot((120,20,257,249),(480,640),ref,[]))

    def test_startup_resets_saved_forward_without_waiting_for_g(self):
        settings=camera_settings()
        head_profile=dict(version=1,gpio_bcm=27,forward=117,handover=127,down_sign=1,
                          bounds=[85,137],cameras=settings,shapes=dict(head=[480,640],belly=[480,640]))
        foot=dict(version=2,method='yellow_green_visible_region',camera=settings['belly'],
                  shape=[480,640],visible_patch_box=[.0390625,0,.4015625,.51875],
                  clipped_edges=['top'],color=dict(hue=40,hue_width=14,min_s=55,min_v=100))
        head,belly,servo=Mock(),Mock(),Mock()
        servo.is_moving.return_value=False
        frame=np.zeros((480,640,3),dtype=np.uint8)
        cv2.circle(frame,(200,200),25,(0,240,210),-1)
        head.getImage.return_value=(True,frame)
        belly.getImage.return_value=(True,frame)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'config').mkdir()
            (root/'config/head_calibration.json').write_text(json.dumps(head_profile),encoding='utf-8')
            (root/'config/right_foot_reference.json').write_text(json.dumps(foot),encoding='utf-8')
            with patch.object(handover_debug,'ROOT',root), \
                 patch.object(handover_debug,'RobotEye',side_effect=[head,belly]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'selectROI',return_value=(190,190,20,20)) as roi, \
                 patch.object(cv2,'waitKey',side_effect=[ord('s'),ord('b'),ord('q')]):
                self.assertTrue(handover_debug.run())
                self.assertEqual([call.args[0] for call in roi.call_args_list],['head','belly'])
                for call in roi.call_args_list:
                    np.testing.assert_array_equal(call.args[1],frame)
            self.assertEqual(json.loads((root/'config/head_calibration.json').read_text()),head_profile)
            self.assertEqual(json.loads((root/'config/right_foot_reference.json').read_text()),foot)
        servo.turn_vertical.assert_called_once_with(129)
        servo.cleanup.assert_called_once()
        head.close.assert_called_once();belly.close.assert_called_once()

    def test_preview_and_belly_reads_continue_during_head_move(self):
        settings=camera_settings()
        profile=dict(version=1,gpio_bcm=27,forward=117,handover=127,down_sign=1,
                     bounds=[85,137],cameras=settings,shapes=dict(head=[480,640],belly=[480,640]))
        foot=dict(version=2,method='yellow_green_visible_region',camera=settings['belly'],
                  shape=[480,640],visible_patch_box=[.0390625,0,.4015625,.51875],
                  clipped_edges=['top'],color=dict(hue=40,hue_width=14,min_s=55,min_v=100))
        head,belly,servo=Mock(),Mock(),Mock()
        servo.is_moving.side_effect=[False,False,False,True,True]
        frame=np.zeros((480,640,3),dtype=np.uint8)
        cv2.circle(frame,(200,350),25,(0,240,210),-1)
        head.getImage.return_value=belly.getImage.return_value=(True,frame)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'config').mkdir()
            (root/'config/head_calibration.json').write_text(json.dumps(profile),encoding='utf-8')
            (root/'config/right_foot_reference.json').write_text(json.dumps(foot),encoding='utf-8')
            with patch.object(handover_debug,'ROOT',root), \
                 patch.object(handover_debug,'RobotEye',side_effect=[head,belly]), \
                 patch.dict('sys.modules',{'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo)}), \
                 patch.object(handover_debug.time,'monotonic',side_effect=iter(i*.25 for i in range(100))), \
                 patch.object(cv2,'imshow') as show,patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=[-1,-1,-1,-1,ord('q')]):
                self.assertTrue(handover_debug.run())
        servo.turn_vertical.assert_called_once_with(129)
        servo.begin_vertical.assert_called_once_with(131,settle_seconds=.18)
        self.assertEqual(show.call_count,10)  # Both windows, including moving frames.
        self.assertEqual(belly.getImage.call_count,6)
        head.discard_frames.assert_called_once_with(1)
        belly.discard_frames.assert_called_once_with(1)
        servo.cleanup.assert_called_once()

    def test_invalid_or_old_reference_is_rejected(self):
        settings=camera_settings()
        head=dict(version=1,gpio_bcm=27,forward=117,handover=127,down_sign=1,
                  bounds=[85,137],cameras=settings,shapes=dict(head=[480,640],belly=[480,640]))
        foot=dict(version=2,method='yellow_green_visible_region',camera=settings['belly'],
                  shape=[480,640],visible_patch_box=[.0390625,0,.4015625,.51875],
                  color=dict(hue=40,hue_width=14,min_s=55,min_v=100))
        validate(head,foot,settings)
        old=deepcopy(foot);old['version']=1
        with self.assertRaises(ValueError):validate(head,old,settings)
        bad=deepcopy(head);bad['forward']=181
        with self.assertRaises(ValueError):validate(bad,foot,settings)
        # 初始位置允许越过旧低头137，但仍在驱动范围内。
        adjusted=deepcopy(head);adjusted['forward']=130
        validate(adjusted,foot,settings)
        adjusted['forward']=138
        validate(adjusted,foot,settings)
        adjusted['forward']=84
        with self.assertRaises(ValueError):validate(adjusted,foot,settings)
        faster=deepcopy(settings)
        for view in faster.values(): view['fps']=30
        validate(head,foot,faster)
        for key,value in [('flip','1'),('device','different'),('backend','usb')]:
            changed=deepcopy(faster);changed['belly'][key]=value
            with self.assertRaises(ValueError):validate(head,foot,changed)

    def test_growing_partial_belly_ball_can_take_over_without_pose_stability(self):
        head_tracker,belly_tracker=PatchTracker(),PatchTracker()
        state=Handover(dict(forward=129,down_sign=1,bounds=[85,137]))
        for i,y in enumerate((-19,-16,-13,-10,-7,-4,0)):
            hf=np.zeros((240,320,3),np.uint8);bf=hf.copy()
            cv2.circle(hf,(160,238),30,(0,240,210),-1)
            cv2.circle(bf,(160,y),30,(0,240,210),-1)
            hb=head_tracker.update(hf);bb=belly_tracker.update(bf)
            self.assertIsNotNone(hb);self.assertIsNotNone(bb)
            state.step(hb,belly_tracker.frames,240,now=i*.05)
            if belly_tracker.frames >= 5 and belly_tracker.frames < 7:
                self.assertLess(belly_tracker.stable_frames,5)
        self.assertEqual(state.phase,'BELLY')
        self.assertEqual(state.angle,129)
