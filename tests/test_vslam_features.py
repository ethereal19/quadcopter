"""Test suite for Landmark Tracking, Breadcrumb Navigation, and Visual SLAM Loop Closure."""

import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from resqnav.landmark_tracker import LandmarkTracker, SurveyedLandmark, VisualLandmark
from resqnav.breadcrumb_nav import BreadcrumbNavigator, Waypoint
from resqnav.slam_loop_closure import VisualSLAM, Keyframe


def test_landmark_tracker_initialization_and_surveyed_distinction():
    """Verify landmark tracker initializes, detects features, and maintains surveyed distinction."""
    tracker = LandmarkTracker(num_features=400)
    assert len(tracker.surveyed_landmarks) >= 2
    assert tracker.surveyed_landmarks[0].name == "DATUM-BASE-ORIGIN"

    # Synthetic textured frame with high-contrast corner features
    np.random.seed(42)
    frame1 = np.full((480, 640, 3), 30, dtype=np.uint8)
    for _ in range(60):
        cx, cy = np.random.randint(40, 600), np.random.randint(40, 440)
        cv2.circle(frame1, (cx, cy), 8, (220, 220, 220), -1)
        cv2.rectangle(frame1, (cx - 10, cy - 10), (cx + 10, cy + 10), (180, 180, 180), 2)

    ann1, metrics1 = tracker.process_frame(frame1, frame_idx=0)

    assert metrics1.tracking_ok is True
    assert metrics1.status == "INITIALIZED"
    assert metrics1.active_tracks > 0
    assert len(tracker.all_landmarks) > 0

    # Second frame with slight shift
    frame2 = np.roll(frame1, shift=3, axis=1)
    ann2, metrics2 = tracker.process_frame(frame2, frame_idx=1)

    assert metrics2.matched_features > 0
    # Inliers should be detected
    summary = tracker.get_landmark_summary()
    assert summary["total_landmarks_seen"] > 0
    assert summary["surveyed_landmarks_registered"] >= 2


def test_breadcrumb_navigation_and_persistence():
    """Verify trajectory building, distance auto-drop, and CSV/SQLite persistence."""
    nav = BreadcrumbNavigator(auto_drop_interval_dist=0.8)
    assert len(nav.waypoints) == 1  # Base origin waypoint
    assert nav.waypoints[0].label == "BASE-DEPLOYMENT-DATUM"

    # Step rover forward
    nav.update_pose_step(dx_rel=0.0, dz_rel=0.5, dyaw_deg=0.0, frame_idx=1, tracking_ok=True)
    nav.update_pose_step(dx_rel=0.0, dz_rel=0.5, dyaw_deg=0.0, frame_idx=2, tracking_ok=True)

    # Since 0.5 + 0.5 = 1.0 > 0.8, an auto-waypoint should have been dropped
    assert len(nav.waypoints) >= 2
    assert nav.current_pose[1] == pytest.approx(1.0, abs=0.05)

    # Add a manual waypoint
    manual_wp = nav.add_manual_waypoint(label="SURVIVOR-SIGHTING-A", frame_idx=3)
    assert manual_wp.is_manual is True
    assert manual_wp.label == "SURVIVOR-SIGHTING-A"

    # Test CSV Export
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "waypoints.csv"
        saved_csv = nav.save_to_csv(str(csv_file))
        assert saved_csv.exists()
        content = saved_csv.read_text(encoding="utf-8")
        assert "SURVIVOR-SIGHTING-A" in content
        assert "BASE-DEPLOYMENT-DATUM" in content

        # Test SQLite Export & Load
        db_file = Path(tmpdir) / "waypoints.db"
        saved_db = nav.save_to_sqlite(str(db_file))
        assert saved_db.exists()

        loaded_wps = nav.load_from_sqlite(str(db_file))
        assert len(loaded_wps) == len(nav.waypoints)
        assert loaded_wps[0].label == "BASE-DEPLOYMENT-DATUM"
        assert any(w.label == "SURVIVOR-SIGHTING-A" for w in loaded_wps)

    # Test Pause / Resume controls
    nav.pause()
    assert nav.is_paused is True
    pose_before = nav.current_pose
    nav.update_pose_step(dx_rel=0.0, dz_rel=2.0, dyaw_deg=10.0, frame_idx=4, tracking_ok=True)
    assert nav.current_pose == pose_before  # Motion rejected while paused

    nav.resume()
    assert nav.is_paused is False


def test_visual_slam_keyframe_and_pose_graph_optimization():
    """Verify keyframe selection, loop closure geometric verification, and SE(2) pose-graph optimization."""
    slam = VisualSLAM(min_keyframe_dist=0.8, min_loop_matches=10, min_loop_inliers=8)

    # Create synthetic textured scene A
    np.random.seed(123)
    scene_a = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(80):
        pt = (np.random.randint(20, 300), np.random.randint(20, 220))
        cv2.circle(scene_a, pt, 5, (255, 255, 255), -1)

    # Add keyframe 0 at origin
    added, kf0, _ = slam.consider_keyframe(scene_a, frame_idx=0, cur_pose=(0.0, 0.0, 0.0))
    assert added is True
    assert len(slam.keyframes) == 1

    # Add keyframe 1 moved forward
    added1, kf1, _ = slam.consider_keyframe(scene_a, frame_idx=10, cur_pose=(0.0, 1.2, 5.0))
    assert added1 is True

    # Add keyframes 2, 3, 4, 5, 6 with clear displacement > 0.8
    for i in range(2, 7):
        slam.consider_keyframe(scene_a, frame_idx=i * 10, cur_pose=(i * 1.2, 1.2 + i * 1.2, 25.0 * i))

    assert len(slam.keyframes) >= 7

    # Now simulate revisiting Scene A with slight drift (e.g., pose = (0.3, 0.2, 3.0))
    # This should match with keyframe 0 or 1
    added_loop, kf_loop, loop_cand = slam.consider_keyframe(scene_a, frame_idx=90, cur_pose=(0.3, 0.2, 3.0))
    assert added_loop is True

    # Check that candidate loop closures were evaluated
    assert len(slam.candidates_history) > 0

    # Test Pose Graph Optimization
    opt_poses = slam.optimize_pose_graph()
    assert len(opt_poses) == len(slam.keyframes)
    assert opt_poses[0] == (0.0, 0.0, 0.0)  # Anchor fixed at origin
