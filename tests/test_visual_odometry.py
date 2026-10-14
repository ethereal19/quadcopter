"""Automated test suite for Monocular Visual Odometry module."""

import cv2
import numpy as np
import pytest
from resqnav.config import VOConfig
from resqnav.visual_odometry import MonocularVisualOdometry


def test_vo_initialization():
    """Verify VO initializes with origin pose and clean trajectory."""
    cfg = VOConfig(num_features=500)
    vo = MonocularVisualOdometry(cfg)

    assert len(vo.trajectory) == 1
    assert vo.trajectory[0] == (0.0, 0.0, True)
    assert np.allclose(vo.R_world, np.eye(3))
    assert np.allclose(vo.t_world, np.zeros((3, 1)))


def test_vo_handles_black_frame():
    """Verify VO handles zero-texture / solid black frames gracefully without crashing."""
    vo = MonocularVisualOdometry()

    # Create solid black image
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # First black frame (fails gracefully due to 0 keypoints)
    m1, _ = vo.process_frame(black_frame, frame_idx=0)
    assert m1.num_keypoints == 0
    assert m1.tracking_ok is False
    assert m1.status == "INSUFFICIENT_FEATURES"

    # Second black frame (should report insufficient features without throwing exception)
    m2, _ = vo.process_frame(black_frame, frame_idx=1)
    assert m2.tracking_ok is False
    assert m2.status == "INSUFFICIENT_FEATURES"
    # Position should not drift or invent motion
    assert m2.x == 0.0
    assert m2.z == 0.0


def test_vo_tracks_synthetic_textured_motion(tmp_path):
    """Verify VO tracks 3D perspective motion in synthetic rover corridor."""
    from resqnav.video_reader import VideoReader, generate_synthetic_pov_video

    video_file = tmp_path / "test_corridor.mp4"
    generate_synthetic_pov_video(video_file, num_frames=10, width=640, height=480, fps=25)

    reader = VideoReader(video_file)
    cfg = VOConfig(num_features=1000, min_features_tracked=10)
    vo = MonocularVisualOdometry(cfg)

    # Process first 5 frames
    metrics_list = []
    for idx, frame in reader.iterate_frames(step=1, max_frames=5):
        m, _ = vo.process_frame(frame, frame_idx=idx)
        metrics_list.append(m)

    reader.release()

    assert len(metrics_list) == 5
    assert metrics_list[0].status == "INITIALIZED"
    assert metrics_list[0].num_keypoints > 50

    # Frame 1 should find matches with Frame 0
    assert metrics_list[1].num_matches > 0
    assert len(vo.trajectory) == 5


def test_vo_reset():
    """Verify reset clears internal state and history."""
    vo = MonocularVisualOdometry()
    frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    vo.process_frame(frame, frame_idx=0)
    assert len(vo.all_metrics) == 1

    vo.reset()
    assert len(vo.all_metrics) == 0
    assert len(vo.trajectory) == 1
    assert vo.trajectory[0] == (0.0, 0.0, True)
