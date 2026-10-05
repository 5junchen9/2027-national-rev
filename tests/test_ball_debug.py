import unittest
from pathlib import Path
import cv2
import numpy as np
from ball_debug import PatchTracker


class BallDebugTests(unittest.TestCase):
    def test_body_move_reacquires_displaced_ball_without_old_pixel_lock(self):
        tracker=PatchTracker()
        frame=np.zeros((240,640,3),np.uint8)
        cv2.circle(frame,(60,100),22,(0,240,210),-1)
        for _ in range(5): self.assertIsNotNone(tracker.update(frame))
        old_color=tracker.color_profile()
        displaced=np.zeros_like(frame)
        cv2.circle(displaced,(350,170),22,(0,240,210),-1)
        self.assertIsNone(tracker.update(displaced))
        tracker.notify_body_move()
        self.assertEqual(tracker.color_profile(),old_color)
        for _ in range(5): box=tracker.update(displaced)
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box[0]+box[2]/2,350,delta=2)
        self.assertGreaterEqual(tracker.stable_frames,5)

    def test_full_search_selects_highest_score_instead_of_largest_region(self):
        tracker=PatchTracker();tracker.begin_search()
        frame=np.zeros((240,640,3),np.uint8)
        cv2.ellipse(frame,(100,100),(40,28),0,0,360,(0,240,210),-1)
        cv2.circle(frame,(400,100),25,(0,240,210),-1)
        box=tracker.update(frame)
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box[0]+box[2]/2,400,delta=2)
        self.assertNotIn('competing',tracker.reason)

    def test_motion_prediction_tracks_acceleration_without_jumping_to_distractor(self):
        tracker=PatchTracker()
        for i,x in enumerate((60,80,115,165,230,310)):
            frame=np.zeros((240,640,3),np.uint8)
            cv2.circle(frame,(x,100),20,(0,240,210),-1)
            if i >= 2: cv2.circle(frame,(550,190),20,(0,240,210),-1)
            box=tracker.update(frame,now=i*.05)
            self.assertIsNotNone(box)
            self.assertAlmostEqual(box[0]+box[2]/2,x,delta=2)
        self.assertEqual(tracker.stable_frames,1)
        for i in range(8): tracker.update(np.zeros_like(frame),now=.3+i*.05)
        remote=np.zeros_like(frame)
        cv2.circle(remote,(550,190),20,(0,240,210),-1)
        self.assertIsNone(tracker.update(remote,now=.8))

    def test_square_green_background_is_rejected(self):
        frame=np.zeros((240,320,3),np.uint8)
        cv2.rectangle(frame,(80,70),(140,130),(0,240,210),-1)
        self.assertIsNone(PatchTracker().update(frame))

    def test_moving_target_wins_over_distractor_at_previous_position(self):
        tracker=PatchTracker()
        for i,x in enumerate((80,100,130,170)):
            frame=np.zeros((240,640,3),np.uint8)
            cv2.circle(frame,(x,100),12,(0,240,210),-1)
            self.assertIsNotNone(tracker.update(frame,now=i*.05))
        frame[:]=0
        cv2.circle(frame,(215,100),12,(0,240,210),-1)
        cv2.circle(frame,(170,100),12,(0,240,210),-1)
        box=tracker.update(frame,now=.20)
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box[0]+box[2]/2,215,delta=2)

    def test_ball_touching_white_shirt_does_not_merge_into_shirt(self):
        frame = np.zeros((240,320,3),dtype=np.uint8)
        frame[70:230,110:310] = (220,220,220)
        cv2.circle(frame,(110,100),35,(0,240,210),-1)
        tracker = PatchTracker()
        for _ in range(8):
            x,y,w,h = tracker.update(frame)
            self.assertLess(w,80)
            self.assertLess(h,80)
            self.assertAlmostEqual(x+w/2,110,delta=3)
            self.assertAlmostEqual(y+h/2,100,delta=3)
        self.assertGreaterEqual(tracker.stable_frames,5)

    def test_actual_head_ball_on_white_shirt_stays_local(self):
        # This crop contains preview annotations and lacks the rightmost columns.
        frame = cv2.imread(str(Path(__file__).parent/'fixtures/tennis_head_white_shirt.png'))
        self.assertIsNotNone(frame)
        for sampled in (False,True):
            tracker = PatchTracker()
            if sampled: tracker.select(frame,(180,235,25,25))
            for _ in range(8):
                box = tracker.update(frame)
                self.assertIsNotNone(box)
                x,y,w,h = box
                self.assertLess(w,205)  # Previous code included >500px of shirt.
                self.assertLess(h,150)
                self.assertAlmostEqual(x+w/2,195,delta=30)
                self.assertAlmostEqual(y+h/2,250,delta=25)
            self.assertGreaterEqual(tracker.stable_frames,5)

    def test_current_right_ball_can_be_reselected_without_unlocking_automatically(self):
        # Screenshot only captures 502 of the 640 camera columns.
        frame = cv2.imread(str(Path(__file__).parent/'fixtures/tennis_belly_right.png'))
        self.assertIsNotNone(frame)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        tracker = PatchTracker()
        tracker.last_box = (25,0,257,249)
        self.assertIsNone(tracker.update(frame))
        self.assertEqual(tracker.last_box,(25,0,257,249))
        tracker.select(frame,(310,230,35,35))
        for _ in range(5):
            box = tracker.update(frame)
            self.assertIsNotNone(box)
            self.assertGreater(box[0],250)
        self.assertGreaterEqual(tracker.stable_frames,5)

    def test_far_round_target_reports_lock_rejection(self):
        frame = np.zeros((240,320,3),dtype=np.uint8)
        cv2.circle(frame,(60,120),30,(0,240,210),-1)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        frame[:] = 0
        cv2.circle(frame,(240,120),30,(0,240,210),-1)
        self.assertIsNone(tracker.update(frame))
        self.assertIn('outside target lock',tracker.reason)

    def test_remote_ball_relocks_only_after_repeated_rejection_and_confirmation(self):
        old=np.zeros((240,640,3),np.uint8)
        cv2.circle(old,(80,100),25,(0,240,210),-1)
        new=np.zeros_like(old)
        cv2.circle(new,(400,180),25,(0,240,210),-1)
        tracker=PatchTracker()
        tracker.update(old)
        old_box=tracker.last_box
        color=tracker.color_profile()
        for i in range(4):
            self.assertIsNone(tracker.update(new))
            self.assertEqual(tracker.last_box,old_box)
            self.assertEqual(tracker.frames,0)
        box=tracker.update(new)
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box[0]+box[2]/2,400,delta=2)
        self.assertEqual(tracker.frames,1)
        self.assertEqual(tracker.stable_frames,1)
        self.assertEqual(tracker.color_profile(),color)
        self.assertEqual(tracker.outside_frames,0)

    def test_changing_remote_candidates_cannot_relock(self):
        frame=np.zeros((240,640,3),np.uint8)
        cv2.circle(frame,(80,100),25,(0,240,210),-1)
        tracker=PatchTracker();tracker.update(frame)
        old_box=tracker.last_box
        for i in range(12):
            frame[:]=0
            cv2.circle(frame,(350 if i%2 else 550,100),25,(0,240,210),-1)
            self.assertIsNone(tracker.update(frame))
        self.assertEqual(tracker.last_box,old_box)
        self.assertEqual(tracker.recovery_frames,1)

    def test_old_ball_wins_over_remote_candidate_and_cancels_recovery(self):
        old=np.zeros((240,640,3),np.uint8)
        cv2.circle(old,(80,100),25,(0,240,210),-1)
        new=np.zeros_like(old)
        cv2.circle(new,(400,100),25,(0,240,210),-1)
        tracker=PatchTracker();tracker.update(old)
        for _ in range(4): self.assertIsNone(tracker.update(new))
        box=tracker.update(old|new)
        self.assertAlmostEqual(box[0]+box[2]/2,80,delta=2)
        self.assertEqual(tracker.recovery_frames,0)
        self.assertEqual(tracker.outside_frames,0)

    def test_empty_frame_breaks_remote_confirmation(self):
        old=np.zeros((240,640,3),np.uint8)
        cv2.circle(old,(80,100),25,(0,240,210),-1)
        new=np.zeros_like(old)
        cv2.circle(new,(400,100),25,(0,240,210),-1)
        tracker=PatchTracker();tracker.update(old)
        for _ in range(4): self.assertIsNone(tracker.update(new))
        self.assertIsNone(tracker.update(np.zeros_like(old)))
        self.assertEqual(tracker.outside_frames,0)
        self.assertEqual(tracker.recovery_frames,0)
        self.assertIsNone(tracker.update(new))

    def test_partial_patch_tracks_then_stops_when_lost(self):
        frame = np.zeros((120,160,3),dtype=np.uint8)
        cv2.circle(frame,(70,115),25,(0,240,210),-1)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        self.assertIn('bottom',tracker.edges)
        moved = np.zeros_like(frame)
        cv2.circle(moved,(85,125),32,(0,240,210),-1)
        self.assertIsNotNone(tracker.update(moved))  # Size and visible proportion change.
        for _ in range(5): self.assertIsNone(tracker.update(np.zeros_like(frame)))
        self.assertIsNotNone(tracker.update(moved))  # Unique nearby target can recover.
        self.assertEqual(tracker.frames,1)  # Saving still needs five fresh confirmations.

    def test_uniform_patch_is_rejected(self):
        with self.assertRaises(ValueError):
            PatchTracker().select(np.zeros((100,100,3),dtype=np.uint8),(10,10,30,30))

    def test_background_sized_selection_is_rejected(self):
        frame = np.zeros((480,640,3),dtype=np.uint8)
        tracker = PatchTracker()
        with self.assertRaisesRegex(ValueError,'too large'):
            tracker.select(frame,(19,33,447,433))
        self.assertIsNone(tracker.box)
        cv2.circle(frame,(230,230),40,(0,240,210),-1)
        tracker.select(frame,(210,210,30,30))
        self.assertIsNotNone(tracker.update(frame))

    def test_two_balls_select_one_but_yellow_rectangle_is_rejected(self):
        frame = np.zeros((120,160,3),dtype=np.uint8)
        cv2.circle(frame,(35,60),18,(0,240,210),-1)
        cv2.circle(frame,(110,60),18,(0,240,210),-1)
        self.assertIsNotNone(PatchTracker().update(frame))
        frame[:] = 0
        cv2.rectangle(frame,(20,40),(140,65),(0,240,210),-1)
        self.assertIsNone(PatchTracker().update(frame))

    def test_surface_color_sampling_handles_green_tennis_ball(self):
        hsv = np.zeros((120,160,3),dtype=np.uint8)
        cv2.circle(hsv,(80,60),30,(65,180,220),-1)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        tracker.select(frame,(65,45,30,30))
        self.assertIsNotNone(tracker.update(frame))
        self.assertAlmostEqual(tracker.hue,65,delta=1)

    def test_selected_ball_ignores_separate_same_color_region(self):
        frame = np.zeros((240,320,3),dtype=np.uint8)
        cv2.circle(frame,(90,120),30,(0,240,210),-1)
        cv2.circle(frame,(170,120),25,(0,240,210),-1)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        tracker.select(frame,(80,110,20,20))
        for _ in range(5):
            box = tracker.update(frame)
            self.assertIsNotNone(box)
            self.assertLess(box[0]+box[2]/2,120)
        self.assertEqual(tracker.frames,5)

    def test_manually_selected_occluded_shape_inside_frame_is_allowed(self):
        frame = np.zeros((160,240,3),dtype=np.uint8)
        cv2.ellipse(frame,(100,80),(40,40),0,0,180,(0,240,210),-1)
        tracker = PatchTracker()
        tracker.select(frame,(90,85,20,20))
        self.assertIsNotNone(tracker.update(frame))

    def test_real_belly_frame_white_seam_and_background(self):
        path = Path(__file__).parent/'fixtures/tennis_belly_seam.png'
        frame = cv2.imread(str(path))
        self.assertIsNotNone(frame)
        tracker = PatchTracker()
        for _ in range(5):
            box = tracker.update(frame)
            self.assertIsNotNone(box)
            x,y,w,h = box
            self.assertTrue(260 < x+w/2 < 340)
            self.assertTrue(210 < y+h/2 < 290)
        self.assertEqual(tracker.frames,5)
        for _ in range(8): tracker.update(np.zeros_like(frame))
        self.assertIsNotNone(tracker.update(frame))
        self.assertEqual(tracker.frames,1)

    def test_after_loss_cannot_jump_to_remote_yellow_ball(self):
        frame = np.zeros((240,640,3),dtype=np.uint8)
        cv2.circle(frame,(80,100),25,(0,240,210),-1)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        for _ in range(6): tracker.update(np.zeros_like(frame))
        far = np.zeros_like(frame)
        cv2.circle(far,(560,100),25,(0,240,210),-1)
        self.assertIsNone(tracker.update(far))

    def test_real_full_to_top_edge_cannot_become_tiny_fragment(self):
        fixtures = Path(__file__).parent/'fixtures'
        full = cv2.imread(str(fixtures/'tennis_belly_full.png'))
        top = cv2.imread(str(fixtures/'tennis_belly_top.png'))
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(full))
        # The lock deliberately rejects jumping between distant snapshots;
        # feed gradual movement before the new top-edge frame.
        for dx,dy in [(-35,-40),(-70,-80),(-105,-120)]:
            shifted = cv2.warpAffine(full,np.float32([[1,0,dx],[0,1,dy]]),(640,480))
            self.assertIsNotNone(tracker.update(shifted))
        for _ in range(5):
            box = tracker.update(top)
            self.assertIsNotNone(box)
            x,y,w,h = box
            self.assertGreater(w,180)
            self.assertGreater(h,180)
            self.assertGreater(x,10)  # Exclude the left border's floor fragments.
            self.assertIn('top',tracker.edges)
        self.assertEqual(tracker.stable_frames,5)

    def test_tiny_fragment_is_not_replacement_for_tracked_ball(self):
        frame = np.zeros((240,320,3),dtype=np.uint8)
        cv2.circle(frame,(150,110),50,(0,240,210),-1)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        fragment = np.zeros_like(frame)
        cv2.circle(fragment,(150,110),8,(0,240,210),-1)
        self.assertIsNone(tracker.update(fragment))
        self.assertEqual(tracker.stable_frames,0)

    def test_moving_ball_cannot_save_as_stable_reference(self):
        tracker = PatchTracker()
        for x in (40,65,90,115,140,165):
            frame = np.zeros((160,320,3),dtype=np.uint8)
            cv2.circle(frame,(x,80),20,(0,240,210),-1)
            self.assertIsNotNone(tracker.update(frame))
        self.assertGreaterEqual(tracker.frames,5)
        self.assertEqual(tracker.stable_frames,1)

    def test_small_pale_yellow_ball_acquires_without_white_background(self):
        hsv = np.zeros((240,320,3),dtype=np.uint8)
        cv2.circle(hsv,(90,90),12,(40,18,230),-1)
        cv2.rectangle(hsv,(220,40),(280,110),(0,0,240),-1)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker = PatchTracker()
        box = tracker.update(frame)
        self.assertIsNotNone(box)
        self.assertLess(box[0]+box[2]/2,130)

    def test_yellow_lock_recovers_nearby_white_ball_but_not_white_paper(self):
        frame = np.zeros((160,320,3),dtype=np.uint8)
        cv2.circle(frame,(90,80),20,(0,240,210),-1)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        white = np.zeros_like(frame)
        cv2.circle(white,(96,80),20,(220,220,220),-1)
        self.assertIsNotNone(tracker.update(white))
        paper = np.zeros_like(frame)
        cv2.rectangle(paper,(76,60),(116,100),(220,220,220),-1)
        self.assertIsNone(tracker.update(paper))

    def test_lock_does_not_follow_same_color_far_after_drop(self):
        frame = np.zeros((200,320,3),dtype=np.uint8)
        cv2.circle(frame,(80,100),20,(0,240,210),-1)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        changed = np.zeros_like(frame)
        cv2.circle(changed,(145,100),20,(0,240,210),-1)
        self.assertIsNone(tracker.update(changed))

    def test_pitch_change_keeps_horizontal_lock_but_allows_vertical_change(self):
        frame = np.zeros((240,320,3),dtype=np.uint8)
        cv2.circle(frame,(80,190),20,(0,240,210),-1)
        tracker = PatchTracker()
        self.assertIsNotNone(tracker.update(frame))
        tracker.notify_pitch_change()
        moved = np.zeros_like(frame)
        cv2.circle(moved,(85,45),20,(0,240,210),-1)
        self.assertIsNotNone(tracker.update(moved))

    def test_orange_blue_and_plain_white_balls_are_not_initial_candidates(self):
        for hue,saturation in [(10,180),(110,180),(0,0)]:
            hsv = np.zeros((160,240,3),dtype=np.uint8)
            cv2.circle(hsv,(100,80),25,(hue,saturation,220),-1)
            frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            tracker = PatchTracker()
            self.assertIsNone(tracker.update(frame))
            with self.assertRaises(ValueError): tracker.select(frame,(90,70,20,20))

    def test_green_white_ball_is_combined_and_color_learned_from_green(self):
        hsv = np.zeros((180,240,3),dtype=np.uint8)
        cv2.circle(hsv,(100,90),35,(0,0,230),-1)
        cv2.ellipse(hsv,(100,90),(35,35),0,90,270,(62,160,230),-1)
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        tracker = PatchTracker()
        box = tracker.update(frame)
        self.assertIsNotNone(box)
        self.assertGreater(box[2],60)
        self.assertTrue(tracker.color_learned)
        self.assertAlmostEqual(tracker.hue,62,delta=1)
