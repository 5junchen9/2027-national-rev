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
from walk_kick import WalkKick, BallDeparture, enlarged_reference


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
               elapsed=0,head_score=1.0,belly_score=1.0,ambiguous=False):
        if match is None:
            match = foot_match_details(belly,(480,640),self.reference,[])
        return planner.decide(phase,head,belly,5,5,(480,640),flip,
                              now=planner.started_at+elapsed,head_score=head_score,
                              belly_score=belly_score,ambiguous=ambiguous)

    def setUp(self):
        self.reference = dict(visible_patch_box=[.2,.2,.2,.2])

    def test_loss_search_uses_only_head_and_never_body_actions(self):
        planner = WalkKick()
        self.assertEqual(self.decide(planner),'WAIT')
        self.assertEqual(self.decide(planner,elapsed=.15),'WAIT')
        self.assertEqual(self.decide(planner,elapsed=.31),'REACQUIRE')
        self.assertEqual(self.decide(planner,elapsed=.8),'WAIT')
        for index in range(80):
            elapsed = .31+(index+1)*1.51
            expected = 'LOWER_HEAD' if (index//4)%2 == 0 else 'RAISE_HEAD'
            self.assertEqual(self.decide(planner,elapsed=elapsed),expected)
            planner.mark_head_search(True,now=planner.started_at+elapsed)
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
        self.assertEqual(self.decide(planner,elapsed=1.82,ambiguous=True),'LOWER_HEAD')
        planner.mark_head_search(False,now=planner.started_at+1.82)
        self.assertEqual(planner.search_head_direction,-1)
        self.assertEqual(self.decide(planner,elapsed=3.33,ambiguous=True),'RAISE_HEAD')

    def test_belly_alignment_and_horizontal_flip(self):
        for flip,expected in [('none','TURN_RIGHT'),('1','TURN_LEFT'),('-1','TURN_LEFT'),('0','TURN_RIGHT')]:
            planner=WalkKick()
            for _ in range(5):
                action=self.decide(planner,belly=(400,96,128,96),phase='BELLY',flip=flip)
            self.assertEqual(action,expected)
        planner=WalkKick()
        for _ in range(5): action=self.decide(planner,belly=(288,110,64,48),phase='BELLY')
        self.assertEqual(action,'UP_LITTLE')

    def test_head_alignment_turns_toward_ball(self):
        for box,expected in [((60,200,60,60),'TURN_LEFT'),((500,200,60,60),'TURN_RIGHT')]:
            planner=WalkKick()
            for _ in range(5): action=self.decide(planner,head=box)
            self.assertEqual(action,expected)

    def test_head_turn_mirror_and_confirmation_reset(self):
        for flip in ('none','0','1','-1'):
            planner = WalkKick()
            for _ in range(4):
                self.assertEqual(self.decide(planner,head=(60,200,60,60),flip=flip),'WAIT')
            expected = 'TURN_RIGHT' if flip in ('1','-1') else 'TURN_LEFT'
            self.assertEqual(self.decide(planner,head=(60,200,60,60),flip=flip),expected)
            planner.mark_sent(expected)
            # 发完转向后，重新观察、确认居中的球，才允许前进。
            for _ in range(4):
                self.assertEqual(self.decide(planner,head=(290,200,60,60),flip=flip),'WAIT')
            self.assertEqual(self.decide(planner,head=(290,200,60,60),flip=flip),'UP_LITTLE')

    def test_fixed_head_loss_recovery_never_requests_pitch_search(self):
        planner = WalkKick(fixed_head=True)
        self.assertEqual(self.decide(planner),'WAIT')
        self.assertEqual(self.decide(planner,elapsed=.15),'WAIT')
        self.assertEqual(self.decide(planner,elapsed=.31),'REACQUIRE')
        for elapsed in (2,10,60,300):
            self.assertEqual(self.decide(planner,elapsed=elapsed),'WAIT')
        self.assertEqual(planner.actions,0)

    def test_turn_action_reaches_shared_body_controller(self):
        self.run_loop('turn_then_disconnect')

    def test_centered_ball_keeps_walking_without_kick(self):
        planner=WalkKick()
        for _ in range(5): action=self.decide(planner,belly=(256,96,128,96),phase='BELLY',ready=100)
        self.assertEqual(action,'UP_LITTLE')
        planner.mark_sent(action)
        for _ in range(5): action=self.decide(planner,belly=(256,96,128,96),phase='BELLY',ready=100)
        self.assertEqual(action,'UP_LITTLE')
        # 原右脚尺寸上限不再阻止行走带球。
        for _ in range(5): action=self.decide(planner,belly=(200,70,240,200),phase='BELLY')
        self.assertEqual(action,'UP_LITTLE')

    def test_belly_ball_alone_turns_or_walks_without_goal(self):
        planner=WalkKick()
        for _ in range(5): action=self.decide(planner,belly=(288,110,64,48),phase='BELLY')
        self.assertEqual(action,'UP_LITTLE')
        for box in ((40,100,60,60),(500,100,60,60)):
            for _ in range(5): action=self.decide(planner,belly=box,phase='BELLY')
            self.assertIn(action,('TURN_LEFT','TURN_RIGHT'))

    def test_lost_ball_cannot_blind_walk(self):
        planner=WalkKick();planner.search_stage='VISUAL';planner.search_head_moves=3
        self.assertEqual(self.decide(planner,elapsed=2),'LOWER_HEAD')
        self.assertEqual(planner.blind_steps,0)

    def test_action_count_stops_body_but_elapsed_time_does_not_stop_search(self):
        planner=WalkKick();planner.actions=30
        self.assertEqual(self.decide(planner,head=(290,200,60,60)),'STOP')
        planner=WalkKick()
        self.assertEqual(planner.decide('HEAD',None,None,0,0,(480,640),'0',
                                        now=planner.started_at+600),'WAIT')

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
        self.assertIsNone(state.step(None,5,480))
        self.assertEqual(state.phase,'BELLY')

    def test_belly_can_take_over_when_head_still_sees_centered_ball(self):
        state=Handover(dict(forward=129,down_sign=1,bounds=[85,137]))
        for _ in range(3): angle=state.step((290,200,60,60),5,480)
        self.assertIsNone(angle)
        self.assertEqual(state.phase,'BELLY')

    def test_competition_returns_on_first_valid_belly_detection(self):
        self.run_loop('competition_belly')

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
            result=Mock(frames=10,stable_frames=10,score=.8,partial=False,edges=[],reason='visible region confirmed')
            result.update.return_value=box
            return result
        head_tracker=tracker((290,200,60,60) if scenario in ('walk_then_disconnect','visible_head_unstable') else None)
        if scenario == 'turn_then_disconnect':
            settings['head']['flip'] = 'none'
            head_tracker.update.return_value = (60,200,60,60)
        if scenario == 'visible_head_unstable': head_tracker.stable_frames=1
        belly_tracker=tracker((256,96,128,96) if scenario not in ('walk_then_disconnect','turn_then_disconnect','startup_no_ball','search_stop','lock_recovery','visible_head_unstable') else None)
        if scenario == 'lock_recovery':
            head_tracker.reason='outside target lock: retry=3/3 confirm=1/3'
        if scenario == 'partial_handover':
            belly_tracker.score=.8
            belly_tracker.stable_frames=1
        if scenario == 'ball_sent':
            def show_sent_ball(action):
                belly_tracker.update.return_value=None
            robot.robotMove.side_effect=show_sent_ball
        if scenario in ('walk_then_disconnect','turn_then_disconnect'):
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
                 patch.object(handover_debug,'camera_settings',side_effect=lambda: json.loads(json.dumps(settings))), \
                 patch.object(handover_debug,'RobotEye',side_effect=[head,belly]), \
                 patch.object(handover_debug,'PatchTracker',side_effect=[head_tracker,belly_tracker]), \
                 patch.dict('sys.modules',{
                     'Head':types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs:servo),
                     'robotmove':types.SimpleNamespace(RobotMove=robot_factory)}), \
                 patch('goal_debug.GoalTracker',side_effect=AssertionError('goal detection removed')), \
                 patch.object(cv2,'imshow'),patch.object(cv2,'destroyAllWindows'), \
                 patch.object(cv2,'waitKey',side_effect=lambda _: ord('q') if (scenario == 'startup_no_ball'
                     or (scenario in ('partial_handover','visible_head_unstable') and head.getImage.call_count >= 20)
                     or (scenario == 'search_stop' and head.getImage.call_count >= 30)
                     or (scenario == 'lock_recovery' and head_tracker.begin_search.call_count+belly_tracker.begin_search.call_count >= 1)) else -1), \
                 patch.object(handover_debug.time,'monotonic',side_effect=iter(i*(.01 if scenario == 'ball_sent' else .1) for i in range(10000))):
                if scenario == 'competition_belly':
                    self.assertTrue(handover_debug.run(actions=True, finish_on_belly=True))
                elif scenario == 'ball_sent':
                    self.assertTrue(handover_debug.run(actions=True))
                elif scenario == 'walk':
                    self.assertFalse(handover_debug.run(actions=True))
                elif scenario in ('startup_no_ball','partial_handover','lock_recovery','visible_head_unstable'):
                    self.assertFalse(handover_debug.run(actions=True))
                elif scenario == 'search_stop':
                    self.assertFalse(handover_debug.run(actions=True))
                else:
                    expected='USB disconnected' if scenario in ('walk_then_disconnect','turn_then_disconnect') else 'serial disconnected'
                    with self.assertRaisesRegex(RuntimeError,expected):
                        handover_debug.run(actions=True)
            self.assertEqual(json.loads((root/'config/right_foot_reference.json').read_text()),foot)
        if scenario in ('startup_no_ball','partial_handover','lock_recovery','visible_head_unstable','competition_belly'):
            robot.robotMove.assert_not_called()
            if scenario in ('partial_handover','visible_head_unstable'): servo.begin_vertical.assert_not_called()
            if scenario == 'lock_recovery':
                head_tracker.begin_search.assert_not_called()
                belly_tracker.begin_search.assert_called_once()
        elif scenario == 'search_stop':
            robot.robotMove.assert_not_called()
            servo.begin_vertical.assert_not_called()
        elif scenario == 'walk':
            self.assertGreaterEqual(robot.robotMove.call_count,2)
            self.assertLessEqual(robot.robotMove.call_count,30)
            self.assertTrue(all(call.args == ('UP_LITTLE',) for call in robot.robotMove.call_args_list))
        elif scenario == 'turn_then_disconnect':
            robot.robotMove.assert_called_once_with('TURN_LEFT')
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

    def test_visible_belly_ball_does_not_trigger_lost_ball_search_when_moving(self):
        planner=WalkKick()
        for i in range(50):
            action=planner.decide('BELLY',None,(256,0,128,60),0,1,(480,640),'0',
                                  now=planner.started_at+i*.1,belly_score=.8)
            self.assertEqual(action,'WAIT')
        self.assertEqual(planner.search_stage,'TRACK')
        self.assertEqual(planner.lost_frames,0)

    def test_visible_head_ball_does_not_reset_tracker_or_scan_before_stable(self):
        planner=WalkKick()
        planner.search_stage='VISUAL'
        for i in range(100):
            action=planner.decide('HEAD',(290,200,60,60),None,1,0,(480,640),'0',
                                  now=planner.started_at+i*.1,head_score=.8)
            self.assertEqual(action,'WAIT')
        self.assertEqual(planner.search_stage,'TRACK')
        self.assertEqual(planner.lost_frames,0)

    def test_real_loop_hands_over_partial_ball_without_body_action_or_loss_exit(self):
        self.run_loop('partial_handover')

    def test_automatic_visual_search_preserves_pending_position_confirmation(self):
        self.run_loop('lock_recovery')

    def test_visible_unstable_head_ball_stops_head_search_without_body_action(self):
        self.run_loop('visible_head_unstable')

    def test_real_loop_stops_after_ball_disappears_without_another_forward_action(self):
        self.run_loop('ball_sent')


class BallDepartureTests(unittest.TestCase):
    def test_five_missing_frames_spanning_point_three_seconds_finish(self):
        check=BallDeparture();check.arm(129,now=0)
        for i in range(4):
            self.assertEqual(check.observe(None,129,now=.5+i*.1),'WAIT')
        self.assertEqual(check.observe(None,129,now=.91),'DONE')

    def test_missing_before_forward_step_never_finishes(self):
        check=BallDeparture()
        self.assertIsNone(check.observe(None,129,now=0))

    def test_visible_ball_resets_missing_confirmation(self):
        check=BallDeparture();check.arm(129,now=0)
        for i in range(4):check.observe(None,129,now=.1+i*.1)
        check.observe((250,280,140,40),129,now=.6)
        self.assertEqual(check.frames,0)
        self.assertEqual(check.observe(None,129,now=.7),'WAIT')

    def test_visible_smaller_receding_ball_does_not_finish(self):
        check=BallDeparture();check.arm(129,now=0)
        for i in range(15):
            self.assertNotEqual(check.observe((270,160,90,90),129,now=.2+i*.1),'DONE')

    def test_head_move_or_expired_window_cancels_observation(self):
        for angle,now in [(132,.4),(129,3.1)]:
            check=BallDeparture();check.arm(129,now=0)
            self.assertIsNone(check.observe(None,angle,now=now))
            self.assertIsNone(check.angle)

    def test_five_fast_misses_alone_are_not_enough(self):
        check=BallDeparture();check.arm(129,now=0)
        for i in range(5):
            self.assertEqual(check.observe(None,129,now=.5+i*.01),'WAIT')

    def test_new_forward_step_resets_previous_confirmation(self):
        check=BallDeparture();check.arm(129,now=0)
        for i in range(4):check.observe(None,129,now=.1+i*.1)
        check.arm(129,now=.6)
        self.assertEqual(check.frames,0)
        self.assertEqual(check.observe(None,129,now=.7),'WAIT')
