"""身体动作决定与真实入口连线检查，使用伪摄像头和伪串口。"""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2
import handover_debug
import robotmove
from handover_debug import Handover, foot_match_details
from dual_kick import camera_settings
from walk_kick import WalkKick, enlarged_reference
from kick_shapes import Box


class WalkKickTests(unittest.TestCase):
    def test_start_robot_handshakes_then_sends_one_stand(self):
        controller=Mock()
        with patch.object(robotmove,'DreamMakerController',return_value=controller):
            robot=handover_debug.start_robot()
            try:
                controller.open.assert_called_once()
                controller.stand.assert_called_once()
                self.assertEqual([call[0] for call in controller.method_calls],['open','stand'])
            finally:
                robot.close()

    def test_startup_handshake_failure_does_not_send_stand(self):
        controller=Mock()
        controller.open.side_effect=RuntimeError('no FB response')
        with patch.object(robotmove,'DreamMakerController',return_value=controller):
            with self.assertRaisesRegex(RuntimeError,'no FB response'):
                handover_debug.start_robot()
        controller.stand.assert_not_called()
        controller.close.assert_called_once()

    def decide(self, planner, head=None, belly=None, phase='HEAD', ready=0, match=None, flip='0',
               elapsed=0,head_score=1.0,belly_score=1.0,ambiguous=False,
               goal=Box(.3,.1,.4,.3),goal_stable=5,head_flip='0'):
        if match is None:
            match = foot_match_details(belly,(480,640),self.reference,[])
        return planner.decide(phase,head,belly,5,5,(480,640),flip,
                              now=planner.started_at+elapsed,head_score=head_score,
                              belly_score=belly_score,ambiguous=ambiguous,goal=goal,
                              goal_stable=goal_stable,head_flip=head_flip)

    def setUp(self):
        self.reference = dict(visible_patch_box=[.2,.2,.2,.2])

    def test_loss_search_uses_only_head_and_never_body_actions(self):
        planner = WalkKick()
        self.assertEqual(self.decide(planner),'WAIT')
        self.assertEqual(self.decide(planner,elapsed=.15),'WAIT')
        self.assertEqual(self.decide(planner,elapsed=.31),'REACQUIRE')
        self.assertEqual(self.decide(planner,elapsed=.8),'WAIT')
        for elapsed in (1.32,2.33,3.34):
            self.assertEqual(self.decide(planner,elapsed=elapsed),'LOWER_HEAD')
            planner.mark_head_search(True,now=planner.started_at+elapsed)
        self.assertEqual(self.decide(planner,elapsed=4.35),'STOP')
        self.assertEqual(planner.actions,0)

    def test_visible_ball_resets_gap_but_belly_loss_never_kicks(self):
        planner = WalkKick(); planner.blind_steps=2
        self.decide(planner,head=(290,200,60,60))
        self.assertEqual(planner.blind_steps,0)
        self.assertEqual(self.decide(planner,phase='BELLY',ready=100),'WAIT')

    def test_low_score_cannot_trigger_kick_and_causes_visual_recovery(self):
        planner=WalkKick()
        for elapsed in (0,.15):
            self.assertEqual(self.decide(planner,belly=(128,96,128,96),phase='BELLY',
                                         ready=100,belly_score=.4,elapsed=elapsed),'WAIT')
        self.assertEqual(self.decide(planner,belly=(128,96,128,96),phase='BELLY',
                                     ready=100,belly_score=.4,elapsed=.31),'REACQUIRE')

    def test_reliable_reacquisition_cancels_search_and_restores_tracking(self):
        planner=WalkKick();planner.search_stage='VISUAL';planner.blind_steps=2
        planner.search_head_moves=3
        self.decide(planner,head=(290,200,60,60))
        self.assertEqual(planner.search_stage,'TRACK')
        self.assertEqual(planner.blind_steps,0)
        self.assertEqual(planner.search_head_moves,0)

    def test_ambiguous_search_never_walks_and_head_limit_is_respected(self):
        planner=WalkKick()
        for elapsed in (0,.15,.31): action=self.decide(planner,elapsed=elapsed,ambiguous=True)
        self.assertEqual(action,'REACQUIRE')
        self.assertEqual(self.decide(planner,elapsed=1.32,ambiguous=True),'LOWER_HEAD')
        planner.mark_head_search(False,now=planner.started_at+1.32)
        self.assertEqual(planner.search_head_moves,3)
        self.assertEqual(self.decide(planner,elapsed=2.33,ambiguous=True),'STOP')

    def test_belly_alignment_and_horizontal_flip(self):
        for flip,expected in [('none','SIDE_LEFT'),('1','SIDE_RIGHT')]:
            planner=WalkKick()
            for _ in range(5):
                action=self.decide(planner,belly=(400,96,128,96),phase='BELLY',flip=flip)
            self.assertEqual(action,expected)
        planner=WalkKick()
        for _ in range(5): action=self.decide(planner,belly=(288,110,64,48),phase='BELLY')
        self.assertEqual(action,'UP_LITTLE')

    def test_head_alignment_uses_corrected_real_robot_direction(self):
        for box,expected in [((60,200,60,60),'SIDE_RIGHT'),((500,200,60,60),'SIDE_LEFT')]:
            planner=WalkKick()
            for _ in range(5): action=self.decide(planner,head=box)
            self.assertEqual(action,expected)

    def test_centered_goal_and_ball_keep_walking_without_kick(self):
        planner=WalkKick()
        for _ in range(5): action=self.decide(planner,belly=(256,96,128,96),phase='BELLY',ready=100)
        self.assertEqual(action,'UP_LITTLE')
        planner.mark_sent(action)
        for _ in range(5): action=self.decide(planner,belly=(256,96,128,96),phase='BELLY',ready=100)
        self.assertEqual(action,'UP_LITTLE')
        # 原右脚尺寸上限不再阻止行走带球。
        for _ in range(5): action=self.decide(planner,belly=(200,70,240,200),phase='BELLY')
        self.assertEqual(action,'UP_LITTLE')

    def test_goal_missing_or_unstable_holds_and_clears_old_action(self):
        planner=WalkKick()
        for _ in range(4): self.decide(planner,belly=(256,96,128,96),phase='BELLY')
        self.assertEqual(self.decide(planner,belly=(256,96,128,96),phase='BELLY',goal=None),'WAIT')
        self.assertEqual(planner.confirm_frames,0)
        self.assertEqual(self.decide(planner,belly=(256,96,128,96),phase='BELLY',goal_stable=4),'WAIT')

    def test_goal_turn_precedes_ball_alignment_and_uses_head_flip(self):
        for flip,expected in [('0','TURN_RIGHT'),('1','TURN_LEFT')]:
            planner=WalkKick()
            for _ in range(5):
                action=self.decide(planner,belly=(40,96,128,96),phase='BELLY',
                                   goal=Box(.55,.1,.4,.3),head_flip=flip)
            self.assertEqual(action,expected)

    def test_lost_ball_cannot_blind_walk_without_goal(self):
        planner=WalkKick();planner.search_stage='VISUAL';planner.search_head_moves=3
        self.assertEqual(self.decide(planner,elapsed=2,goal=None),'STOP')
        self.assertEqual(planner.blind_steps,0)

    def test_action_count_and_deadline_stop(self):
        planner=WalkKick();planner.actions=30
        self.assertEqual(self.decide(planner,head=(290,200,60,60)),'STOP')
        planner=WalkKick()
        self.assertEqual(planner.decide('HEAD',None,None,0,0,(480,640),'0',
                                        now=planner.started_at+121),'STOP')

    def test_box_grows_three_percent_without_changing_saved_reference(self):
        expanded=enlarged_reference(self.reference)
        self.assertEqual(self.reference['visible_patch_box'],[.2,.2,.2,.2])
        self.assertAlmostEqual(expanded['visible_patch_box'][2],.206)
        self.assertAlmostEqual(expanded['visible_patch_box'][3],.206)
        self.assertAlmostEqual(expanded['visible_patch_box'][0],.197)
        edge=enlarged_reference(dict(visible_patch_box=[0,0,.4,.5]))
        self.assertEqual(edge['visible_patch_box'][:2],[0,0])

    def test_handover_before_any_fixed_down_position(self):
        state=Handover(dict(forward=121,handover=127,down_sign=1,bounds=[85,137]))
        for _ in range(2): self.assertIsNone(state.step(None,5,480))
        self.assertEqual(state.step(None,5,480),121)
        self.assertEqual(state.phase,'BELLY')

    def test_belly_can_take_over_when_head_still_sees_centered_ball(self):
        state=Handover(dict(forward=129,down_sign=1,bounds=[85,137]))
        for _ in range(3): angle=state.step((290,200,60,60),5,480)
        self.assertEqual(angle,129)
        self.assertEqual(state.phase,'BELLY')

    def run_loop(self, scenario):
        settings=camera_settings()
        profile=dict(version=1,gpio_bcm=27,forward=117,handover=127,down_sign=1,
                     bounds=[85,137],cameras=settings,shapes=dict(head=[480,640],belly=[480,640]))
        foot=dict(version=2,method='yellow_green_visible_region',camera=settings['belly'],
                  shape=[480,640],visible_patch_box=[.2,.2,.2,.2],
                  color=dict(hue=40,hue_width=14,min_s=55,min_v=100))
        frame=np.zeros((480,640,3),np.uint8)
        head,belly,servo,robot=Mock(),Mock(),Mock(),Mock()
        robot_factory=Mock(return_value=robot)
        head.getImage.return_value=belly.getImage.return_value=(True,frame)
        servo.is_moving.return_value=False
        def tracker(box):
            result=Mock(frames=10,stable_frames=10,score=.8,edges=[],reason='visible region confirmed')
            result.update.return_value=box
            return result
        head_tracker=tracker(None if scenario != 'walk_then_disconnect' else (290,200,60,60))
        belly_tracker=tracker((256,96,128,96) if scenario not in ('walk_then_disconnect','startup_no_ball','search_stop') else None)
        if scenario == 'walk_then_disconnect':
            def disconnect_after_move(action):
                head.getImage.side_effect = RuntimeError('USB disconnected')
            robot.robotMove.side_effect = disconnect_after_move
        elif scenario == 'walk_failure':
            robot.robotMove.side_effect = RuntimeError('serial disconnected during walk')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'config').mkdir()
            for name,data in [('head_calibration',profile),('right_foot_reference',foot)]:
                (root/'config'/f'{name}.json').write_text(json.dumps(data))
            with patch.object(handover_debug,'ROOT',root), \
                 patch.object(handover_debug,'RobotEye',side_effect=[head,belly]), \
                 patch.object(handover_debug,'PatchTracker',side_effect=[head_tracker,belly_tracker]), \
                 patch('goal_debug.find_goal_frames',return_value=[Box(192,60,256,180)]), \
                 patch.dict('sys.modules',{
                     'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                     'robotmove':types.SimpleNamespace(RobotMove=robot_factory)}), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',return_value=ord('q') if scenario == 'startup_no_ball' else -1), \
                 patch.object(handover_debug.time,'monotonic',side_effect=iter(i*.1 for i in range(2000))):
                if scenario == 'walk':
                    self.assertFalse(handover_debug.run(actions=True))
                elif scenario == 'startup_no_ball':
                    self.assertFalse(handover_debug.run(actions=True))
                elif scenario == 'search_stop':
                    self.assertFalse(handover_debug.run(actions=True))
                else:
                    expected='USB disconnected' if scenario == 'walk_then_disconnect' else 'serial disconnected'
                    with self.assertRaisesRegex(RuntimeError,expected):
                        handover_debug.run(actions=True)
            self.assertEqual(json.loads((root/'config/right_foot_reference.json').read_text()),foot)
        if scenario == 'startup_no_ball':
            robot.robotMove.assert_not_called()
        elif scenario == 'search_stop':
            robot.robotMove.assert_not_called()
            self.assertEqual([call.args[0] for call in servo.begin_vertical.call_args_list],[132,135,138])
        elif scenario == 'walk':
            self.assertGreaterEqual(robot.robotMove.call_count,2)
            self.assertLessEqual(robot.robotMove.call_count,30)
            self.assertTrue(all(call.args == ('UP_LITTLE',) for call in robot.robotMove.call_args_list))
        else:
            robot.robotMove.assert_called_once_with('UP_LITTLE')
        robot_factory.assert_called_once_with(None,port='/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-port0')
        robot.close.assert_called_once()
        servo.turn_vertical.assert_called_once_with(129)
        servo.cleanup.assert_called_once()
        head.close.assert_called_once();belly.close.assert_called_once()

    def test_real_loop_walks_without_kick_and_releases_resources(self):
        self.run_loop('walk')

    def test_startup_initializes_body_even_without_a_ball(self):
        self.run_loop('startup_no_ball')

    def test_real_loop_searches_with_head_only_and_sends_no_body_actions(self):
        self.run_loop('search_stop')

    def test_camera_disconnect_after_step_sends_no_more_actions(self):
        self.run_loop('walk_then_disconnect')

    def test_serial_failure_during_walk_does_not_retry(self):
        self.run_loop('walk_failure')

    def test_ball_search_cannot_walk_toward_off_center_goal(self):
        planner=WalkKick();planner.search_stage='VISUAL';planner.search_head_moves=3
        self.assertEqual(self.decide(planner,elapsed=2,goal=Box(.55,.1,.4,.3)),'STOP')
