import unittest
from itertools import count, cycle, chain, repeat
from unittest.mock import Mock, patch
import numpy as np

from line_search_route import detect_line, line_mask, route_segments, near_line_sample, LinePlanner, LineWidthReference, stripe_width, run_course
from competition_line_main import LineCompetitionIO, run_line_test, main, GuardedRobot
import types
import cv2
from competition_main import load_settings, CONFIG_FILE
from kick_shapes import Box


class LineWidthTests(unittest.TestCase):
    def make_reference(self):
        mask = np.zeros((480,640),np.uint8)
        # 上方细、下方粗，模拟透视；分开记录各高度宽度。
        mask[:240,316:325] = 255
        mask[240:,312:329] = 255
        reference = LineWidthReference()
        segments = [(320,479,320,0)]
        for _ in range(9):
            reference.sample(mask, segments)
        self.assertFalse(reference.ready)
        reference.sample(mask, segments)
        self.assertTrue(reference.ready)
        return reference, mask, segments

    def test_samples_ten_frames_and_locks_width_per_height(self):
        reference, mask, segments = self.make_reference()
        self.assertEqual(reference.widths[108],9)
        self.assertEqual(reference.widths[396],17)
        self.assertEqual(reference.filter(mask,segments),segments)
        saved = reference.widths.copy()
        reference.sample(np.zeros_like(mask),segments)
        self.assertEqual(reference.widths,saved)

    def test_rejects_both_thin_and_wide_false_lines(self):
        reference, _, segments = self.make_reference()
        for stripe_width_pixels in (3,35):
            mask = np.zeros((480,640),np.uint8)
            mask[:,320:320+stripe_width_pixels] = 255
            self.assertEqual(reference.filter(mask,segments),[])

    def test_missing_or_jumping_line_skips_frame_without_losing_samples(self):
        mask = np.zeros((480,640),np.uint8)
        mask[:,312:329] = 255
        reference = LineWidthReference()
        for _ in range(5):
            reference.sample(mask,[(320,479,320,0)])
        reference.sample(mask,[])
        self.assertEqual(len(reference.frames),5)
        self.assertFalse(reference.ready)
        reference.sample(mask,[(320,479,320,0)])
        shifted = np.roll(mask,80,axis=1)
        reference.sample(shifted,[(400,479,400,0)])
        self.assertEqual(len(reference.frames),6)

    def test_no_actual_white_pixels_cannot_finish_sampling(self):
        reference = LineWidthReference()
        for _ in range(12):
            reference.sample(np.zeros((480,640),np.uint8),[(320,479,320,0)])
        self.assertFalse(reference.ready)

    def test_diagonal_width_is_measured_perpendicular_to_stripe(self):
        straight = np.zeros((480,640),np.uint8)
        diagonal = np.zeros_like(straight)
        cv2.line(straight,(320,0),(320,479),255,12)
        cv2.line(diagonal,(100,400),(400,100),255,12)
        self.assertLessEqual(abs(stripe_width(straight,320,250,0)-
                                 stripe_width(diagonal,250,250,45)),2)
        reference = LineWidthReference()
        for _ in range(10):
            reference.sample(straight,[(320,479,320,0)])
        segments = [(100,400,400,100)]
        self.assertEqual(reference.filter(diagonal,segments),segments)

    def test_unobserved_height_is_not_rejected(self):
        reference, _, _ = self.make_reference()
        reference.widths = {396:17}
        segment = [(100,200,100,50)]
        self.assertEqual(reference.filter(np.zeros((480,640),np.uint8),segment),segment)


class LineTests(unittest.TestCase):
    def test_hough_coordinate_shapes_are_read_as_four_coordinates_per_segment(self):
        mask=np.zeros((480,640),np.uint8)
        segments=np.array([[320,0,320,220],[180,479,400,220]],dtype=np.int32)
        for shape in ((2,1,4),(2,4),(1,2,4),(8,)):
            with patch('line_search_route.cv2.HoughLinesP',return_value=segments.reshape(shape)):
                vertical,diagonal=route_segments(mask)
            self.assertEqual(len(vertical),1)
            self.assertEqual(len(diagonal),1)

    def corner_mask(self, vertical=True, diagonal=True, top_only=False):
        mask=np.zeros((480,640),np.uint8)
        if vertical:
            cv2.line(mask,(400,0),(400,100 if top_only else 220),255,12)
        if diagonal:
            cv2.line(mask,(180,479),(400,220),255,12)
        return mask

    def test_joined_corner_is_split_into_vertical_and_diagonal_segments(self):
        vertical,diagonal=route_segments(self.corner_mask())
        self.assertTrue(vertical)
        self.assertTrue(diagonal)

    def test_vertical_line_at_top_of_entire_frame_still_blocks_turn(self):
        vertical,diagonal=route_segments(self.corner_mask(top_only=True))
        self.assertTrue(vertical)
        self.assertTrue(diagonal)
        self.assertTrue(all(max(segment[1],segment[3])<384 for segment in vertical))

    def test_only_diagonal_and_mirror_direction_are_distinguished(self):
        mask=self.corner_mask(vertical=False)
        vertical,diagonal=route_segments(mask)
        self.assertFalse(vertical)
        self.assertTrue(diagonal)
        self.assertFalse(route_segments(cv2.flip(mask,1))[1])
        self.assertTrue(route_segments(cv2.flip(mask,1),mirrored=True)[1])
        broad=np.zeros_like(mask)
        broad[150:450,200:450]=255
        self.assertEqual(route_segments(broad),([],[]))

    def test_vertical_stays_forward_even_with_diagonal_then_two_turns_once(self):
        planner=LinePlanner()
        self.assertEqual(planner.decide(400,640,True),'WAIT')
        self.assertEqual(planner.decide(400,640,True),'SIDE_LEFT')
        self.assertEqual(planner.decide(400,640,True),'SIDE_LEFT')
        self.assertEqual(planner.decide(None,640,True),'WAIT')
        self.assertEqual(planner.decide(None,640,True),'TURN_RIGHT_2')
        for _ in range(4):self.assertEqual(planner.decide(None,640,True),'WAIT')
        self.assertEqual(planner.decide(320,640),'WAIT')
        self.assertEqual(planner.decide(320,640),'UP_LITTLE')
        self.assertEqual(planner.decide(None,640,True),'WAIT')
        self.assertEqual(planner.decide(None,640,True),'TURN_RIGHT_2')

    def test_small_angle_offset_uses_sideways_actions_and_respects_mirror(self):
        for x,angle,mirrored,expected in (
                (230,5,False,'SIDE_RIGHT'), (410,-5,False,'SIDE_LEFT'),
                (230,5,True,'SIDE_LEFT'), (410,-5,True,'SIDE_RIGHT'),
                (350,0,False,'UP_LITTLE'), (410,12,False,'SIDE_LEFT')):
            with self.subTest(x=x,angle=angle,mirrored=mirrored):
                planner=LinePlanner()
                self.assertEqual(planner.decide(x,640,angle=angle,mirrored=mirrored),'WAIT')
                self.assertEqual(planner.decide(x,640,angle=angle,mirrored=mirrored),expected)

    def test_centered_line_uses_angle_correction_and_handles_mirror(self):
        for angle,mirrored,expected in ((18,False,'TURN_RIGHT'),(-18,False,'TURN_LEFT'),
                                        (18,True,'TURN_LEFT'),(-18,True,'TURN_RIGHT')):
            planner=LinePlanner()
            planner.decide(320,640,angle=angle,mirrored=mirrored)
            self.assertEqual(planner.decide(320,640,angle=angle,mirrored=mirrored),expected)
            self.assertEqual(planner.decide(320,640,angle=angle,mirrored=mirrored),'WAIT')

    def test_translation_precedes_turning_when_both_errors_exist(self):
        for x,angle,mirrored,expected in ((410,18,False,'SIDE_LEFT'),
                                         (230,-18,False,'SIDE_RIGHT'),
                                         (410,18,True,'SIDE_RIGHT'),
                                         (230,-18,True,'SIDE_LEFT')):
            planner=LinePlanner()
            planner.decide(x,640,angle=angle,mirrored=mirrored)
            self.assertEqual(planner.decide(x,640,angle=angle,mirrored=mirrored),expected)
            self.assertEqual(planner.decide(320,640,angle=angle,mirrored=mirrored),'WAIT')
            expected_turn='TURN_RIGHT' if (angle > 0) != mirrored else 'TURN_LEFT'
            self.assertEqual(planner.decide(320,640,angle=angle,mirrored=mirrored),expected_turn)

    def test_angle_gap_and_post_bend_tilt_are_detected(self):
        for angle,recovering in ((18,False),(35,True),(-35,True)):
            mask=np.zeros((480,640),np.uint8)
            top_x=round(320+400*np.tan(np.radians(angle)))
            cv2.line(mask,(320,450),(top_x,50),255,12)
            vertical,_=route_segments(mask,recovering=recovering)
            self.assertTrue(vertical)

    def test_post_bend_correction_does_not_rearm_corner_until_line_is_upright(self):
        planner=LinePlanner()
        planner.decide(320,640)
        planner.decide(320,640)
        planner.decide(None,640,True)
        self.assertEqual(planner.decide(None,640,True),'TURN_RIGHT_2')
        planner.decide(400,640,angle=35)
        self.assertEqual(planner.decide(400,640,angle=35),'SIDE_LEFT')
        self.assertTrue(planner.after_bend)
        self.assertEqual(planner.decide(None,640,True),'WAIT')
        self.assertTrue(planner.after_bend)
        planner.decide(400,640,angle=5)
        self.assertEqual(planner.decide(400,640,angle=5),'SIDE_LEFT')
        self.assertFalse(planner.after_bend)

    def test_near_sample_uses_bottom_quarter_and_ignores_upper_segment(self):
        upper=(100,220,100,0)
        lower=(320,479,400,0)
        x,angle=near_line_sample([upper,lower],480,640)
        self.assertGreater(x,320)
        self.assertLess(x,331)
        self.assertLess(abs(angle),10)
        self.assertEqual(near_line_sample([upper],480,640),(None,0))

    def test_upper_original_line_blocks_corner_when_near_sample_is_missing(self):
        planner=LinePlanner()
        planner.decide(320,640)
        planner.decide(320,640)
        for _ in range(3):
            self.assertEqual(planner.decide(None,640,True,original_visible=True),'UP_LITTLE')
        self.assertEqual(planner.decide(None,640,True),'WAIT')
        self.assertEqual(planner.decide(None,640,True),'TURN_RIGHT_2')

    def test_blank_frame_and_single_frame_loss_never_trigger_corner(self):
        planner=LinePlanner()
        for _ in range(3):self.assertEqual(planner.decide(None,640,True),'WAIT')
        planner.decide(320,640)
        planner.decide(320,640)
        for _ in range(3):self.assertEqual(planner.decide(None,640),'WAIT')
        self.assertEqual(planner.decide(None,640,True),'WAIT')
        planner.decide(320,640,True)
        planner.decide(320,640,True)
        self.assertEqual(planner.decide(None,640,True),'WAIT')

    @patch('competition_line_main.cv2.imshow')
    def test_real_masks_trigger_exactly_two_turns_only_after_vertical_leaves(self, show):
        io=self.make_io()
        head=np.zeros((480,640,3),np.uint8)
        straight=np.zeros_like(head)
        cv2.line(straight,(400,479),(400,0),(255,255,255),12)
        diagonal=cv2.cvtColor(self.corner_mask(vertical=False),cv2.COLOR_GRAY2BGR)
        frames=chain([straight]*2, [diagonal]*4, [np.zeros_like(straight)]*200)
        def observe():
            try:
                return head,next(frames),{}
            except StopIteration:
                raise KeyboardInterrupt
        io.observe_ready.side_effect=observe
        with self.assertRaises(KeyboardInterrupt):
            io.follow_to_carry('red')
        self.assertEqual([call.args[0] for call in io.move.call_args_list],
                         ['SIDE_LEFT','TURN_RIGHT','TURN_RIGHT'])

    @patch('competition_line_main.cv2.imshow')
    def test_restored_upper_original_line_keeps_forward_until_it_leaves(self, show):
        io=self.make_io()
        head=np.zeros((480,640,3),np.uint8)
        straight=np.zeros_like(head)
        cv2.line(straight,(320,479),(320,0),(255,255,255),12)
        corner=cv2.cvtColor(self.corner_mask(top_only=True),cv2.COLOR_GRAY2BGR)
        diagonal=cv2.cvtColor(self.corner_mask(vertical=False),cv2.COLOR_GRAY2BGR)
        io.observe_ready.side_effect=([(head,straight,{})]*2+[(head,corner,{})]*3
                                     +[(head,diagonal,{})]*4+[KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            io.follow_to_carry('red')
        self.assertEqual([call.args[0] for call in io.move.call_args_list],
                         ['UP_LITTLE']*4+['TURN_RIGHT','TURN_RIGHT'])

    @patch('competition_line_main.cv2.imshow')
    def test_after_two_corner_turns_tilted_route_gets_single_correction(self, show):
        io=self.make_io()
        head=np.zeros((480,640,3),np.uint8)
        straight=np.zeros_like(head)
        cv2.line(straight,(400,479),(400,0),(255,255,255),12)
        diagonal=cv2.cvtColor(self.corner_mask(vertical=False),cv2.COLOR_GRAY2BGR)
        tilted=np.zeros_like(head)
        cv2.line(tilted,(300,479),(600,0),(255,255,255),12)
        upright=np.zeros_like(head)
        cv2.line(upright,(320,479),(320,0),(255,255,255),12)
        frames=chain([straight]*2,[diagonal]*4,[tilted]*2,[upright]*2,[np.zeros_like(head)]*200)
        def observe():
            try:
                return head,next(frames),{}
            except StopIteration:
                raise KeyboardInterrupt
        io.observe_ready.side_effect=observe
        with self.assertRaises(KeyboardInterrupt):
            io.follow_to_carry('red')
        self.assertEqual([call.args[0] for call in io.move.call_args_list],
                         ['SIDE_LEFT','TURN_RIGHT','TURN_RIGHT','TURN_RIGHT','UP_LITTLE'])

    @patch('competition_line_main.cv2.imshow')
    def test_camera_failure_during_two_turns_prevents_remaining_actions(self, show):
        io=self.make_io()
        head=np.zeros((480,640,3),np.uint8)
        straight=np.zeros_like(head)
        cv2.line(straight,(400,479),(400,0),(255,255,255),12)
        diagonal=cv2.cvtColor(self.corner_mask(vertical=False),cv2.COLOR_GRAY2BGR)
        io.observe_ready.side_effect=[(head,straight,{})]*2+[(head,diagonal,{})]*3+[RuntimeError('camera failed')]
        with self.assertRaisesRegex(RuntimeError,'camera failed'):
            io.follow_to_carry('red')
        self.assertEqual([call.args[0] for call in io.move.call_args_list],
                         ['SIDE_LEFT','TURN_RIGHT'])

    def test_diagonal_lines_crossing_near_band_are_not_rejected_as_wide_regions(self):
        for width,height in ((640,480),(320,240)):
            for start,end in ((.28,.47),(.72,.53)):
                image=np.zeros((height,width,3),np.uint8)
                cv2.line(image,(round(width*start),height-1),
                         (round(width*end),round(height*.8)),(255,255,255),
                         max(3,round(width*12/640)))
                x=detect_line(image)
                self.assertIsNotNone(x)
                self.assertLess(abs(x-width*(start+end)/2),width*.03)


    def test_large_bright_region_is_rejected_even_if_it_crosses_near_band(self):
        image=np.zeros((480,640,3),np.uint8)
        image[384:,220:420]=255
        self.assertIsNone(detect_line(image))


    def test_line_only_requires_explicit_real_action_flags(self):
        with patch('sys.argv',['competition_line.py','--run','--line-only']), \
             patch('competition_line_main.run_line_test') as run:
            with self.assertRaises(SystemExit):main()
        run.assert_not_called()


    def test_line_only_dispatches_without_full_competition_preflight(self):
        with patch('sys.argv',['competition_line.py','--run','--actions','--line-only','--color','red']), \
             patch('competition_line_main.run_line_test',return_value=True) as run, \
             patch('competition_line_main.preflight') as preflight:
            self.assertTrue(main())
        self.assertEqual(run.call_args.args[1:],('red','white'))
        preflight.assert_not_called()


    @patch('competition_line_main.cv2.destroyAllWindows')
    def test_line_only_checks_camera_before_body_and_never_runs_other_phases(self, windows):
        io,robot=Mock(),Mock()
        events=[]
        io.observe_ready.side_effect=lambda:events.append('camera')
        def connect(*args,**kwargs):
            events.append('body')
            return robot
        with patch('competition_line_main.LineCompetitionIO',return_value=io), \
             patch.dict('sys.modules',{'robotmove':types.SimpleNamespace(RobotMove=connect)}):
            self.assertTrue(run_line_test(load_settings(CONFIG_FILE),'red','white'))
        self.assertEqual(events,['camera','body'])
        io.follow_to_carry.assert_called_once_with('red')
        for name in ('prepare_identity','scan_until','identity','carry','sport','dance'):
            getattr(io,name).assert_not_called()
        robot.close.assert_called_once()
        io.close_views.assert_called_once()


    @patch('competition_line_main.cv2.destroyAllWindows')
    def test_line_only_camera_failure_never_opens_body(self, windows):
        io=Mock()
        io.observe_ready.side_effect=RuntimeError('camera failed')
        factory=Mock()
        with patch('competition_line_main.LineCompetitionIO',return_value=io), \
             patch.dict('sys.modules',{'robotmove':types.SimpleNamespace(RobotMove=factory)}):
            with self.assertRaisesRegex(RuntimeError,'camera failed'):
                run_line_test(load_settings(CONFIG_FILE),'red','white')
        factory.assert_not_called()
        io.close_views.assert_called_once()


    def test_white_line_on_blue_floor_and_black_line_on_white_floor(self):
        white = np.full((480,640,3), (100,30,20), np.uint8)
        white[384:,310:330] = 255
        self.assertAlmostEqual(detect_line(white),319.5)
        black = np.full((480,640,3),255,np.uint8)
        black[384:,310:330] = 0
        self.assertAlmostEqual(detect_line(black,'black'),319.5)


    def test_whole_bright_floor_is_not_a_line(self):
        self.assertIsNone(detect_line(np.full((480,640,3),255,np.uint8)))


    def test_follow_previous_local_candidate(self):
        image = np.zeros((480,640,3),np.uint8)
        image[384:,200:220] = 255
        image[384:,360:380] = 255
        self.assertLess(detect_line(image,previous_x=210),230)


    def test_white_paper_with_black_islands_excluded_when_qr_localization_fails(self):
        image=np.zeros((480,640,3),np.uint8)
        image[260:480,460:520]=255
        for y in (280,330,400):
            image[y:y+12,480:495]=0
        image[384:,310:330]=255
        with patch('line_search_route.cv2.QRCodeDetector') as detector:
            detector.return_value.detectMulti.return_value=(False,None)
            mask=line_mask(image)
        self.assertFalse(mask[384:,460:520].any())
        self.assertAlmostEqual(detect_line(image,mask=mask),319.5)


    def make_io(self):
        clock=patch('competition_line_main.time.monotonic',side_effect=count())
        clock.start()
        self.addCleanup(clock.stop)
        io=LineCompetitionIO(load_settings(CONFIG_FILE),GuardedRobot(Mock(),float('inf')),float('inf'))
        io.open_views=Mock()
        io.stream=Mock()
        frame=np.zeros((480,640,3),np.uint8)
        io.observe_ready=Mock(side_effect=[(frame,frame,{})]*200+[KeyboardInterrupt()])
        io.move=Mock()
        io.set_head=Mock()
        return io


    @patch('competition_line_main.cv2.imshow')
    def test_initial_missing_line_waits_past_timeout_until_manual_stop(self, show):
        io=self.make_io()
        with self.assertRaises(KeyboardInterrupt):
            io.follow_to_carry('red')
        io.move.assert_not_called()
        self.assertGreater(io.observe_ready.call_count,30)  # 可以持续等待新图，直到手动退出。


    @patch('competition_line_main.cv2.imshow')
    def test_search_suspends_deadlines_and_restores_remaining_time(self, show):
        io=self.make_io()
        io.deadline=10
        io.robot.deadline=10
        def finish(color):
            self.assertEqual(io.deadline,float('inf'))
            self.assertEqual(io.robot.deadline,float('inf'))
        io.follow_line=Mock(side_effect=finish)
        with patch('competition_line_main.time.monotonic',side_effect=[100,500]):
            io.follow_to_carry('red')
        self.assertEqual(io.deadline,410)
        self.assertEqual(io.robot.deadline,410)

    @patch('competition_line_main.cv2.imshow')
    def test_width_sampling_starts_on_new_image_after_forward_returns(self, show):
        for frame_count, samples, moves in ((2,0,1),(3,1,2),(13,10,12)):
            with self.subTest(frame_count=frame_count):
                io = self.make_io()
                frame = np.zeros((480,640,3),np.uint8)
                cv2.line(frame,(320,0),(320,479),(255,255,255),12)
                io.observe_ready.side_effect = [(frame,frame,{})]*frame_count+[KeyboardInterrupt()]
                reference = LineWidthReference()
                original_sample = reference.sample
                def sample_after_move(mask, segments):
                    self.assertGreater(io.move.call_count,0)
                    original_sample(mask,segments)
                reference.sample = sample_after_move
                with patch('competition_line_main.LineWidthReference',return_value=reference):
                    with self.assertRaises(KeyboardInterrupt):
                        io.follow_to_carry('red')
                self.assertEqual(len(reference.frames),samples)
                self.assertEqual(io.move.call_count,moves)
                self.assertEqual(reference.ready,samples == 10)

    @patch('competition_line_main.cv2.imshow')
    def test_failed_forward_does_not_start_width_sampling(self, show):
        io = self.make_io()
        frame = np.zeros((480,640,3),np.uint8)
        cv2.line(frame,(320,0),(320,479),(255,255,255),12)
        io.observe_ready.side_effect = [(frame,frame,{})]*3
        io.move.side_effect = RuntimeError('move failed')
        reference = LineWidthReference()
        with patch('competition_line_main.LineWidthReference',return_value=reference):
            with self.assertRaisesRegex(RuntimeError,'move failed'):
                io.follow_to_carry('red')
        self.assertEqual(reference.frames,[])
        self.assertFalse(reference.ready)

    @patch('competition_line_main.cv2.imshow')
    def test_search_continues_beyond_configured_action_limit(self, show):
        io=self.make_io()
        io.settings['route_max_steps']=2
        frame=np.zeros((480,640,3),np.uint8)
        cv2.line(frame,(320,0),(320,479),(255,255,255),12)
        io.observe_ready.side_effect=[(frame,frame,{})]*41+[KeyboardInterrupt()]
        with patch('competition_line_main.route_segments',return_value=([(320,479,320,0)],[])):
            with self.assertRaises(KeyboardInterrupt):
                io.follow_to_carry('red')
        self.assertEqual(io.move.call_count,40)  # 采样不增加WAIT，第二帧即开始前进。
        self.assertTrue(all(call.args==('UP_LITTLE',) for call in io.move.call_args_list))

    @patch('competition_line_main.cv2.imshow')
    def test_same_color_block_confirms_before_any_search_motion(self, show):
        io=self.make_io()
        with patch('competition_line_main.BlockTracker') as tracker:
            tracker.return_value.update.return_value=Box(100,100,60,60)
            io.follow_to_carry('red')
        io.move.assert_not_called()
        self.assertEqual(io.observe_ready.call_count,3)
        io.set_head.assert_called_once_with(127)


    @patch('competition_line_main.cv2.imshow')
    def test_different_blocks_do_not_accumulate_confirmation(self, show):
        io=self.make_io()
        boxes=[[Box(10 if i%2 else 200,100,60,60)] for i in range(30)]
        with patch('competition_line_main.BlockTracker') as tracker:
            tracker.return_value.update.side_effect=cycle([items[0] for items in boxes])
            with self.assertRaises(KeyboardInterrupt):
                io.follow_to_carry('red')
        io.move.assert_not_called()


    def test_course_has_no_left_or_action1_scan_and_preserves_later_tasks(self):
        io=Mock();io.settings=load_settings(CONFIG_FILE)
        run_course(io,'red')
        io.follow_to_carry.assert_called_once_with('red')
        self.assertEqual([call.args[0] for call in io.forward.call_args_list],[3,10])
        self.assertEqual([call.args[0] for call in io.scan_until.call_args_list],['face'])
        io.carry.assert_called_once_with('red','action2')
        names=[call[0] for call in io.method_calls]
        self.assertLess(names.index('stand'),names.index('follow_to_carry'))
        self.assertLess(names.index('left'),names.index('forward'))
        self.assertLess(names.index('forward'),names.index('follow_to_carry'))
        self.assertLess(names.index('follow_to_carry'),names.index('carry'))
        self.assertLess(names.index('carry'),names.index('sport'))
        io.dance.assert_called_once()
