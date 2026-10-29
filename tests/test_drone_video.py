"""Tests for synthetic aerial drone video generation and Nadir VO tracking."""

from pathlib import Path
import pytest
import cv2
import numpy as np

from resqnav.video_reader import VideoReader, generate_synthetic_drone_video


def test_drone_video_generation(tmp_path):
    """Verify that generate_synthetic_drone_video produces a valid playable video file."""
    out_video = tmp_path / "test_drone.mp4"
    res_path = generate_synthetic_drone_video(
        output_path=out_video,
        num_frames=20,
        width=320,
        height=240,
        fps=25,
        include_hud=False,
        seed=42
    )

    assert res_path.exists(), "Drone video file should be created"
    assert res_path.stat().st_size > 1000, "Drone video file should have non-trivial size"

    reader = VideoReader(res_path)
    meta = reader.get_metadata()
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["frame_count"] == 20

    ok, frame, idx = reader.read_frame()
    assert ok is True
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    reader.release()


def test_drone_planar_feature_tracking():
    """Verify that aerial Nadir frames yield high-density feature inliers under affine planar motion."""
    drone_path = Path("data/sample_drone_recon.mp4")
    if not drone_path.exists():
        pytest.skip("data/sample_drone_recon.mp4 not pre-generated")

    reader = VideoReader(drone_path)
    orb = cv2.ORB_create(nfeatures=600)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    prev_kps, prev_desc = None, None
    inlier_counts = []

    for idx, frame in reader.iterate_frames(step=1, max_frames=10):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = orb.detectAndCompute(gray, None)
        if prev_kps is not None and desc is not None:
            matches = bf.knnMatch(desc, prev_desc, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= 15:
                pts_curr = np.float32([kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts_prev = np.float32([prev_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                _, mask = cv2.estimateAffinePartial2D(pts_prev, pts_curr)
                if mask is not None:
                    inlier_counts.append(int(np.sum(mask)))
        prev_kps, prev_desc = kps, desc

    reader.release()
    assert len(inlier_counts) >= 5, "Should track multiple consecutive aerial frames"
    avg_inliers = np.mean(inlier_counts)
    assert avg_inliers > 50, f"Expected strong aerial tracking, got {avg_inliers} avg inliers"
