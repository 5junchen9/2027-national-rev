import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import competition_main
from competition_main import CompetitionIO, GuardedRobot, load_settings
from competition_route import QRStepPlanner, RouteVision, BlueEntry, blue_ratios, run_course
from competition_identity import choose_face, recognize, load_source, load_models, infer_identity


class CompetitionTests(unittest.TestCase):
    def settings(self):
        return load_settings(competition_main.CONFIG_FILE)

    def test_target_qr_stops_walking_and_requires_consecutive_frames(self):
        planner = QRStepPlanner('face', 3, 2, 100)
        self.assertEqual(planner.decide(['sber'], 1), 'UP_LITTLE')
        planner.mark_step()
        self.assertEqual(planner.decide(['face'], 2), 'WAIT')
        self.assertEqual(planner.decide([], 3), 'UP_LITTLE')
        planner.mark_step()
        self.assertEqual([planner.decide(['face'], t) for t in (4, 5, 6)], ['WAIT', 'WAIT', 'DONE'])

    def test_multiple_codes_motion_cooldown_and_limits(self):
        planner = QRStepPlanner('face', 2, 1, 10)
        self.assertEqual(planner.decide(['face', 'face'], 1), 'WAIT')
        self.assertEqual(planner.decide(['face'], 2, ready=False), 'WAIT')
        self.assertEqual(planner.frames, 0)
        planner.mark_step()
        self.assertEqual(planner.decide([], 3), 'STOP')
        self.assertEqual(planner.decide(['face'], 10), 'STOP')

    def test_blue_cube_does_not_count_as_blue_floor(self):
        settings = self.settings()
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.rectangle(frame, (130, 150), (180, 200), (255, 0, 0), -1)
        ratios = blue_ratios(frame, settings['blue_roi_near'], settings['blue_roi_far'])
        entry = BlueEntry(.65, 3)
        self.assertFalse(entry.update(ratios))
        frame[:] = (255, 0, 0)
        ratios = blue_ratios(frame, settings['blue_roi_near'], settings['blue_roi_far'])
        self.assertEqual([entry.update(ratios) for _ in range(3)], [False, False, True])
        self.assertFalse(entry.update([1, 0]))

    def test_same_course_order_for_each_field_color(self):
        for color in ('red', 'blue', 'yellow'):
            io = Mock()
            io.settings = self.settings()
            self.assertTrue(run_course(io, color))
            calls = [call[0] for call in io.method_calls]
            self.assertLess(calls.index('identity'), calls.index('carry'))
            self.assertLess(calls.index('carry'), calls.index('sport'))
            self.assertLess(calls.index('sport'), calls.index('enter_blue'))
            self.assertLess(calls.index('enter_blue'), calls.index('dance'))
            io.dance.assert_called_once()
            io.carry.assert_called_once_with(color, io.settings['drop_qr'])
            self.assertEqual([(call.args[0], call.args[1]) for call in io.scan_until.call_args_list],
                             [('face', 'belly'), ('sber', 'belly'), ('action1', 'belly'), ('dance', 'head')])

    def test_failed_identity_never_enters_carry_or_dance(self):
        io = Mock();io.settings = self.settings()
        io.identity.side_effect = RuntimeError('identity failed')
        with self.assertRaises(RuntimeError): run_course(io, 'blue')
        io.carry.assert_not_called();io.sport.assert_not_called();io.dance.assert_not_called()
        io.left.assert_called_once_with(1)
        io.right.assert_not_called()

    def test_actual_qr_loop_holds_at_candidate(self):
        io = CompetitionIO(self.settings(), Mock(), float('inf'))
        io.open_views = Mock();io.ready_at = 0
        io.observe = Mock(side_effect=[(None, None, {'belly': [('face', None)]})] * 3)
        io.move = Mock()
        io.scan_until('face', 'belly')
        io.move.assert_not_called()

    def test_background_qr_remembers_code_after_it_leaves_view(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        head, belly = Mock(), Mock()
        head.getImage.return_value = (True, frame)
        decode = Mock(side_effect=[[], [('face', None)]] * 3 + [[], []])
        with patch('competition_route.threading.Thread'):
            stream = RouteVision(head, belly, decode, 3)
        reads = 0
        def belly_read():
            nonlocal reads
            reads += 1
            if reads == 4: stream.stopped.set()
            return True, frame
        belly.getImage.side_effect = belly_read
        stream.watch('face', 'belly')
        stream.capture()
        self.assertTrue(stream.saw_target())
        self.assertEqual(stream.latest[2]['belly'], [])
        stream.watch('sber', 'belly')
        self.assertFalse(stream.saw_target())

    def test_code_seen_during_last_allowed_step_still_completes_stage(self):
        settings = self.settings();settings['route_max_steps'] = 0
        io = CompetitionIO(settings, Mock(), float('inf'))
        io.open_views = Mock();io.ready_at = 0
        io.stream = Mock();io.stream.saw_target.return_value = True
        io.observe = Mock(return_value=(None, None, {'belly': []}))
        io.move = Mock()
        io.scan_until('face', 'belly')
        io.move.assert_not_called()
        io.stream.watch.assert_any_call(None, None)

    def test_failed_preflight_never_opens_body_connection(self):
        robot_factory = Mock()
        with patch.object(competition_main, 'preflight', return_value=False), \
             patch.dict('sys.modules', {'robotmove': types.SimpleNamespace(RobotMove=robot_factory)}), \
             patch('sys.argv', ['competition.py', '--run', '--actions', '--color', 'blue']):
            self.assertFalse(competition_main.main())
        robot_factory.assert_not_called()

    def test_fixed_moves_wait_for_fresh_observation_after_previous_action(self):
        for method, action in (('forward', 'UP_LITTLE'), ('right', 'TURN_RIGHT'), ('left', 'TURN_LEFT')):
            io = CompetitionIO(self.settings(), Mock(), float('inf'))
            io.ready_at = 2
            io.observe = Mock(return_value=(None, None, {}))
            io.move = Mock()
            with patch('competition_main.time.monotonic', side_effect=[0, 1, 2]):
                getattr(io, method)(1)
            self.assertEqual(io.observe.call_count, 3)
            io.move.assert_called_once_with(action)

    def test_carry_receives_shared_body_connection(self):
        robot = Mock()
        io = CompetitionIO(self.settings(), robot, 999)
        io.close_views = Mock();io.open_views = Mock()
        with patch('carry_vision.run', return_value=True) as carry:
            io.carry('blue', io.settings['drop_qr'])
        carry.assert_called_once_with('blue', io.settings['drop_qr'], actions=True,
                                      robot=robot, search_right_actions=5, deadline=999,
                                      qr_reader=competition_main.read_codes)
        io.close_views.assert_called_once();io.open_views.assert_called_once()

    def test_failure_in_each_task_never_starts_later_tasks(self):
        tasks = ['identity', 'carry', 'sport', 'enter_blue', 'dance']
        for index, task in enumerate(tasks):
            io = Mock();io.settings = self.settings()
            getattr(io, task).side_effect = RuntimeError('task failed')
            with self.assertRaises(RuntimeError): run_course(io, 'blue')
            for later in tasks[index+1:]:
                getattr(io, later).assert_not_called()

    def test_complete_route_uses_exact_markers_color_return_and_one_dance(self):
        for color, count in [('red', 2), ('blue', 3), ('yellow', 5)]:
            io = Mock();io.settings = self.settings()
            io.settings['return_right_actions'][color] = count
            self.assertTrue(run_course(io, color))
            self.assertEqual([call.args for call in io.scan_until.call_args_list],
                             [('face', 'belly'), ('sber', 'belly'),
                              ('action1', 'belly'), ('dance', 'head')])
            self.assertEqual([call.args[0] for call in io.right.call_args_list], [1, 4, 4, count, 7])
            io.left.assert_called_once_with(1)
            self.assertEqual([call.args[0] for call in io.forward.call_args_list], [1, 3, 1])
            calls = [call[0] for call in io.method_calls]
            carry_index = calls.index('carry')
            self.assertEqual(calls[carry_index-2:carry_index], ['forward', 'phase'])
            io.carry.assert_called_once_with(color, io.settings['drop_qr'])
            io.dance.assert_called_once()

    def test_face_turns_are_single_turn_actions_around_identity(self):
        from robotmove import ACTIONS
        self.assertEqual(ACTIONS['TURN_LEFT'],('左转.dzz',1))
        self.assertEqual(ACTIONS['TURN_RIGHT'],('右转.dzz',1))
        io = Mock();io.settings = self.settings()
        run_course(io,'blue')
        calls = [call for call in io.method_calls if call[0] != 'phase']
        self.assertEqual([call[0] for call in calls[:5]],
                         ['scan_until','left','identity','right','scan_until'])
        io.forward.assert_any_call(io.settings['factory_entry_forward_steps'])

    def test_time_guard_rejects_motion_before_sending(self):
        robot = Mock()
        with patch('time.monotonic', return_value=100):
            guarded = GuardedRobot(robot, 101)
            with self.assertRaises(RuntimeError): guarded.robotMove('HOLD_BOX')
        robot.robotMove.assert_not_called()

    def test_validated_pickup_is_the_real_hold_box_action(self):
        from robotmove import ACTIONS
        from dreammaker_protocol import load_dzz
        frames = load_dzz(competition_main.ROOT / 'assets/actions' / ACTIONS['HOLD_BOX'][0])
        self.assertEqual([milliseconds for _, milliseconds in frames], [600] * 4)
        self.assertEqual((frames[0][0][8], frames[0][0][14]), (-150, 150))

    def test_identity_searches_entire_image_using_colleague_face_order(self):
        left = (10, 20, 50, 50, .9);right = (520, 350, 80, 80, .9)
        self.assertEqual(choose_face([left]), left)
        self.assertEqual(choose_face([right]), right)
        self.assertEqual(choose_face([right, left]), right)
        self.assertIsNone(choose_face([]))

    def test_identity_matches_standalone_intervals_and_clears_cache_on_lost_face(self):
        root = competition_main.legacy_root(self.settings())
        source = load_source(root, 'app/face_main.py', 'test_identity_intervals')
        frame = np.zeros((480,640,3),np.uint8)
        detector, gender, ocr = Mock(), Mock(), Mock()
        detector.detect.return_value = [(100,100,80,80,.9)]
        gender.classify.return_value = ('female', .9)
        ocr.read_name.side_effect = [('李娜', .9), ('王芳', .95), ('李娜', .9)]
        state = dict(frame_index=0, name=None, name_score=None,
                     gender_label=None, gender_score=0.0)
        results = [infer_identity(frame,detector,gender,ocr,source,state)[0] for _ in range(4)]
        self.assertEqual(results,['李娜','李娜','李娜','王芳'])
        self.assertEqual(detector.detect.call_count,4)
        self.assertEqual(gender.classify.call_count,2)
        self.assertEqual(ocr.read_name.call_count,2)
        detector.detect.return_value = []
        self.assertIsNone(infer_identity(frame,detector,gender,ocr,source,state)[0])
        self.assertIsNone(state['name'])
        detector.detect.return_value = [(100,100,80,80,.9)]
        self.assertEqual(infer_identity(frame,detector,gender,ocr,source,state)[0],'李娜')
        self.assertEqual(ocr.read_name.call_count,3)

    def test_competition_face_head_is_124_and_dance_head_remains_120(self):
        self.assertEqual(self.settings()['face_head_position'],124)
        self.assertEqual(self.settings()['dance_head_position'],120)

    def test_voice_face_task_uses_current_identity_entry(self):
        import face_main
        import voice_main
        with patch.object(face_main,'recognize',return_value=True) as recognize:
            self.assertTrue(voice_main._run_face(None))
        recognize.assert_called_once_with(competition_main.legacy_root(self.settings()),124,60)

    def check_colleague_identity(self, missing_frame=False):
        root = competition_main.legacy_root(self.settings())
        eye, head, detector, gender, ocr = Mock(), Mock(), Mock(), Mock(), Mock()
        speech = Mock(return_value=True)
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[0, 0] = [10, 20, 30]
        frame[0, -1] = [40, 50, 60]
        eye.getImage.return_value = (True, frame)
        face = [(520, 350, 80, 80, .9)]
        detector.detect.side_effect = [face, [], face, face, face] if missing_frame else [face] * 3
        gender.classify.return_value = ('female', .9)
        ocr.read_name.return_value = ('李晓明', .9)
        with patch.dict('sys.modules', {
            'Head': types.SimpleNamespace(RobotHeadServoOnly=lambda **kw: head),
            'roboteye': types.SimpleNamespace(RobotEye=lambda **kw: eye),
            'chinese_speech': types.SimpleNamespace(speak_chinese=speech)}), \
             patch('competition_identity.load_models', return_value=(detector, gender, ocr)), \
             patch('robot_audio.configure') as configure, \
             patch.object(cv2, 'imshow'), patch.object(cv2, 'destroyAllWindows'), \
             patch.object(cv2, 'waitKey', return_value=-1):
            self.assertTrue(recognize(root, 124, 5))
        head.turn_vertical.assert_called_once_with(124)
        configure.assert_called_once_with(volume=127)
        speech.assert_called_once_with('李晓明，女性')
        count = 2 if missing_frame else 1
        self.assertEqual(gender.classify.call_count, count)
        self.assertEqual(ocr.read_name.call_count, count)
        self.assertEqual(ocr.read_name.call_args.args[1], (520, 350, 80, 80))
        np.testing.assert_array_equal(detector.detect.call_args.args[0][0, 0], [40, 50, 60])
        np.testing.assert_array_equal(ocr.read_name.call_args.args[0][0, 0], [40, 50, 60])
        np.testing.assert_array_equal(frame[0, 0], [10, 20, 30])
        eye.close.assert_called_once();head.cleanup.assert_called_once()
        for model in (detector, gender, ocr):
            model.close.assert_called_once()

    def test_colleague_ocr_name_is_announced_without_old_name_model(self):
        self.check_colleague_identity()

    def test_lost_face_resets_actual_ocr_confirmation(self):
        self.check_colleague_identity(missing_frame=True)

    def test_slow_identity_keeps_preview_live_without_cached_confirmations(self):
        root = competition_main.legacy_root(self.settings())
        eye, head, detector, gender, ocr = Mock(), Mock(), Mock(), Mock(), Mock()
        eye.getImage.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        detector.detect.return_value = [(100, 100, 80, 80, .9)]
        started, release = threading.Event(), threading.Event()
        preview_count = 0

        def slow_gender(image):
            started.set()
            if not release.wait(2):
                raise RuntimeError('预览没有在推理等待期间刷新')
            return 'female', .9

        def show(window, image):
            nonlocal preview_count
            if started.is_set():
                preview_count += 1
            if preview_count >= 8:
                release.set()

        gender.classify.side_effect = slow_gender
        ocr.read_name.return_value = ('李晓明', .9)
        speech = Mock()
        with patch.dict('sys.modules', {
            'Head': types.SimpleNamespace(RobotHeadServoOnly=lambda **kw: head),
            'roboteye': types.SimpleNamespace(RobotEye=lambda **kw: eye),
            'chinese_speech': types.SimpleNamespace(speak_chinese=speech)}), \
             patch('competition_identity.load_models', return_value=(detector, gender, ocr)), \
             patch('robot_audio.configure'), patch.object(cv2, 'imshow', side_effect=show), \
             patch.object(cv2, 'destroyAllWindows'), \
             patch.object(cv2, 'waitKey', side_effect=lambda delay: ord('q') if release.is_set() else -1):
            self.assertFalse(recognize(root, 120, 5))
        self.assertGreaterEqual(preview_count, 8)
        # 多次预览既没有排队提交推理，也没有把一个结果重复确认并播报。
        detector.detect.assert_called_once()
        gender.classify.assert_called_once()
        ocr.read_name.assert_called_once()
        speech.assert_not_called()
        eye.close.assert_called_once()
        for model in (detector, gender, ocr):
            model.close.assert_called_once()

    def test_rapidocr_empty_text_does_not_break_colleague_name_parser(self):
        root = competition_main.legacy_root(self.settings())
        engine = Mock(return_value=types.SimpleNamespace(txts=None, scores=None))
        with patch.dict('sys.modules', {
            'rapidocr_onnxruntime': types.SimpleNamespace(RapidOCR=lambda: engine)}), \
             patch.object(cv2.dnn, 'readNetFromTensorflow'), \
             patch.object(cv2.dnn, 'readNetFromONNX'):
            detector, gender, ocr = load_models(root)
        image = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(ocr.read_name(image, (100, 100, 100, 100)), (None, None))
        self.assertEqual(engine.call_count, 2)  # 下方无字后，整帧也检查一次。

    def test_colleague_detector_searches_right_half_and_rejects_low_confidence(self):
        root = competition_main.legacy_root(self.settings())
        source = load_source(root, 'hardware/face_detector_dnn.py', 'test_colleague_detector')
        net = Mock()
        net.forward.return_value = np.array([[[[0, 1, .9, .5, .5, 1., 1.],
                                              [0, 1, .2, .1, .1, .4, .4]]]], np.float32)
        with patch.object(cv2.dnn, 'readNetFromTensorflow', return_value=net):
            detector = source.FaceDetectorDNN()
        faces = detector.detect(np.zeros((100, 200, 3), np.uint8))
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0][:4], (100, 50, 100, 50))

    def test_colleague_gender_uses_named_output(self):
        root = competition_main.legacy_root(self.settings())
        source = load_source(root, 'hardware/face_detector_dnn.py', 'test_colleague_gender')
        net = Mock();net.forward.return_value = np.array([[1., 4.]], np.float32)
        with patch.object(cv2.dnn, 'readNetFromONNX', return_value=net):
            gender = source.GenderClassifier()
        label, confidence = gender.classify(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(label, 'female');self.assertGreater(confidence, .9)
        net.forward.assert_called_once_with('gender_output')

    def test_carry_calibration_injects_real_qr_reader_without_body_actions(self):
        with patch('carry_vision.run', return_value=False) as calibrate, \
             patch('sys.argv', ['competition.py', '--run', '--calibrate-carry', '--color', 'blue']):
            self.assertTrue(competition_main.main())
        calibrate.assert_called_once_with('blue', self.settings()['drop_qr'],
                                          qr_reader=competition_main.read_codes)

    def test_wechat_url_is_kept_as_exact_text_without_network_execution(self):
        from competition_qr import read_codes
        payload = self.settings()['drop_qr']
        decoder = Mock(return_value=[types.SimpleNamespace(data=payload.encode(), rect=(10, 20, 30, 40))])
        module = types.SimpleNamespace(decode=decoder, ZBarSymbol=types.SimpleNamespace(QRCODE='QR'))
        with patch.dict('sys.modules', {'pyzbar': types.ModuleType('pyzbar'), 'pyzbar.pyzbar': module}):
            codes = read_codes(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(codes[0][0], payload)
        self.assertEqual((codes[0][1].x, codes[0][1].y), (10, 20))
        decoder.assert_called_once()

    def test_qr_enhancement_restores_coordinates_and_preserves_payload(self):
        from competition_qr import read_codes
        payload = self.settings()['drop_qr']
        code = types.SimpleNamespace(data=payload.encode(),rect=(20,40,60,80))
        for misses in (1,2):
            with self.subTest(misses=misses):
                decoder = Mock(side_effect=[[]]*misses+[[code]])
                module = types.SimpleNamespace(decode=decoder,ZBarSymbol=types.SimpleNamespace(QRCODE='QR'))
                with patch.dict('sys.modules',{'pyzbar':types.ModuleType('pyzbar'),'pyzbar.pyzbar':module}):
                    codes = read_codes(np.zeros((100,100,3),np.uint8))
                self.assertEqual(codes[0][0],payload)
                box = codes[0][1]
                self.assertEqual((box.x,box.y,box.width,box.height),(10,20,30,40))
                self.assertEqual(decoder.call_count,misses+1)
                image = decoder.call_args.args[0]
                self.assertEqual(image.shape,(200,200))
                if misses == 2: self.assertTrue(np.isin(image,[0,255]).all())

    def test_qr_enhancement_does_not_invent_payload_when_all_attempts_fail(self):
        from competition_qr import read_codes
        decoder = Mock(return_value=[])
        module = types.SimpleNamespace(decode=decoder,ZBarSymbol=types.SimpleNamespace(QRCODE='QR'))
        with patch.dict('sys.modules',{'pyzbar':types.ModuleType('pyzbar'),'pyzbar.pyzbar':module}):
            self.assertEqual(read_codes(np.zeros((100,100,3),np.uint8)),[])
        self.assertEqual(decoder.call_count,3)

    def test_shared_carry_search_turns_once_without_new_connection(self):
        import carry_vision
        from dual_kick import camera_settings
        from kick_shapes import Box
        head, belly, servo, robot = Mock(), Mock(), Mock(), Mock()
        hf = np.zeros((480, 640, 3), np.uint8);bf = hf.copy()
        head.getImage.return_value = (True, hf);belly.getImage.return_value = (True, bf)
        servo.is_moving.return_value = False
        pickup, drop = Box(256, 72, 64, 48), Box(256, 144, 128, 96)
        found_qr = False
        def moved(action):
            nonlocal found_qr
            if action == 'RIGHT_HOLDBOX': found_qr = True
        robot.robotMove.side_effect = moved
        reference = dict(version=2, cameras=camera_settings(), head_position=129,
                         target_qr='action2', shapes=dict(head=[480, 640], belly=[480, 640]),
                         pickup=[.4, .5, .1, .1], drop=[.4, .3, .2, .2], drop_head_position=120)
        factory = Mock()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'carry.json';path.write_text(json.dumps(reference))
            with patch.object(carry_vision, 'REFERENCE_FILE', path), \
                 patch.object(carry_vision, 'block_quality', return_value=(1.0,1.0)), \
                 patch.object(carry_vision, 'RobotEye', side_effect=[head, belly]), \
                 patch.object(carry_vision, 'find_blocks', side_effect=lambda image, color, near=False: [pickup] if image is bf else []), \
                 patch.object(carry_vision, 'read_qr_codes', side_effect=lambda *args: [('action2', drop)] if found_qr else []), \
                 patch.dict('sys.modules', {'Head': types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs: servo),
                                           'robotmove': types.SimpleNamespace(RobotMove=factory)}), \
                 patch.object(cv2, 'imshow'), patch.object(cv2, 'destroyAllWindows'), \
                 patch.object(cv2, 'waitKey', return_value=-1), \
                 patch('time.monotonic', side_effect=iter(i*.1 for i in range(2000))):
                self.assertTrue(carry_vision.run('blue', 'action2', actions=True, robot=robot,
                                                search_right_actions=2, deadline=999))
        self.assertEqual([call.args[0] for call in robot.robotMove.call_args_list],
                         ['HOLD_BOX', 'RIGHT_HOLDBOX', 'DOWN_BOX'])
        factory.assert_not_called();robot.close.assert_not_called()
        head.close.assert_called_once();belly.close.assert_called_once();servo.cleanup.assert_called_once()

    def test_shared_sport_stops_after_ball_recedes_without_repeating_stand(self):
        import handover_debug
        from dual_kick import camera_settings
        settings = camera_settings()
        head_profile = dict(version=1, gpio_bcm=27, forward=129, down_sign=1, bounds=[85, 137],
                            cameras=settings, shapes=dict(head=[480, 640], belly=[480, 640]))
        foot = dict(version=2, method='yellow_green_visible_region', camera=settings['belly'],
                    shape=[480, 640], visible_patch_box=[.1, .1, .2, .2], clipped_edges=[],
                    color=dict(hue=40, hue_width=14, min_s=55, min_v=100))
        head, belly, servo, robot = Mock(), Mock(), Mock(), Mock()
        frame = np.zeros((480, 640, 3), np.uint8)
        head.getImage.return_value = belly.getImage.return_value = (True, frame)
        servo.is_moving.return_value = False
        trackers = [Mock(frames=10, stable_frames=10, score=1, reason='ok', edges=[]) for _ in range(2)]
        for tracker in trackers: tracker.update.return_value = (256, 96, 128, 96)
        def ball_sent(action):
            trackers[1].update.return_value = (280, 20, 80, 60)
        robot.robotMove.side_effect = ball_sent
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);(root/'config').mkdir()
            (root/'config/head_calibration.json').write_text(json.dumps(head_profile))
            (root/'config/right_foot_reference.json').write_text(json.dumps(foot))
            with patch.object(handover_debug, 'ROOT', root), \
                 patch.object(handover_debug, 'RobotEye', side_effect=[head, belly]), \
                 patch.object(handover_debug, 'PatchTracker', side_effect=trackers), \
                 patch.object(handover_debug, 'start_robot') as factory, \
                 patch.dict('sys.modules', {'Head': types.SimpleNamespace(RobotHeadServoOnly=lambda **kwargs: servo)}), \
                 patch.object(cv2, 'imshow'), patch.object(cv2, 'destroyAllWindows'), \
                 patch.object(cv2, 'waitKey', return_value=-1), \
                 patch('time.monotonic', side_effect=iter(i*.01 for i in range(10000))):
                self.assertTrue(handover_debug.run(fps=settings['head']['fps'], actions=True,
                                                  robot=robot, deadline=999))
        robot.robotMove.assert_called_once_with('UP_LITTLE')
        factory.assert_not_called();robot.close.assert_not_called()

    def test_missing_music_does_not_send_dance(self):
        io = CompetitionIO(self.settings(), Mock(), float('inf'))
        io.close_views = Mock();io.move = Mock()
        process = Mock();process.poll.return_value = 1
        with patch('robot_audio.start', return_value=process), patch('time.sleep'):
            with self.assertRaises(RuntimeError): io.dance()
        io.move.assert_not_called()

    def test_dance_music_process_released_when_motion_fails(self):
        io = CompetitionIO(self.settings(), Mock(), float('inf'))
        io.close_views = Mock();io.move = Mock(side_effect=RuntimeError('serial failed'))
        process = Mock();process.poll.return_value = None
        with patch('robot_audio.start', return_value=process), patch('time.sleep'):
            with self.assertRaises(RuntimeError): io.dance()
        process.terminate.assert_called_once();process.wait.assert_called_once()

    def test_music_failure_during_last_action_does_not_report_completion(self):
        import dance_main
        io = CompetitionIO(self.settings(), Mock(), float('inf'))
        io.close_views = Mock();io.move = Mock()
        process = Mock()
        process.poll.side_effect = [None] * (1 + len(dance_main.DANCE_ACTIONS)) + [1, 1]
        with patch('robot_audio.start', return_value=process), patch('time.sleep'):
            with self.assertRaises(RuntimeError): io.dance()
        self.assertEqual(io.move.call_count, len(dance_main.DANCE_ACTIONS))

    def test_empty_or_out_of_frame_blue_roi_rejected(self):
        settings = self.settings();settings['blue_roi_near'] = [0, .6, 0, .9]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json';path.write_text(json.dumps(settings))
            with self.assertRaises(ValueError): load_settings(path)


if __name__ == '__main__':
    unittest.main()
