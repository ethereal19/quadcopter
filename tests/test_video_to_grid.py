"""Integration tests verifying dynamic 2D occupancy grid generation and A* path planning directly from video."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from resqnav.occupancy_grid import OccupancyGrid
from resqnav.path_planning import astar_search, validate_path
from resqnav.perception import VideoPerceptionEngine
from resqnav.video_reader import VideoReader
from resqnav.visual_odometry import MonocularVisualOdometry, VOConfig


def test_video_to_grid_pipeline_execution():
    """Verify end-to-end processing: video -> VO -> perception -> grid mapping -> A* pathfinding."""
    video_path = Path("data/sample_recon.mp4")
    assert video_path.exists(), "Sample recon video must exist"

    reader = VideoReader(video_path)
    vo = MonocularVisualOdometry(VOConfig(num_features=800, relative_step_scale=1.0))
    perception = VideoPerceptionEngine()

    grid = OccupancyGrid(
        rows=32,
        cols=42,
        cell_size_m=0.5,
        default_state=OccupancyGrid.CELL_UNKNOWN,
        mode_label="Mode B: Video-Derived Occupancy Grid Map"
    )

    start_pt = (26, 21)
    grid.set_cell(start_pt[0], start_pt[1], OccupancyGrid.CELL_FREE, hit_count=5)
    origin_x, origin_y = grid.cell_to_metric(start_pt[0], start_pt[1])

    frames_processed = 0
    cur_r, cur_c = start_pt

    for frame_idx, frame in reader.iterate_frames(step=4, max_frames=12):
        # 1. VO motion tracking
        metrics, vo_annotated = vo.process_frame(frame, frame_idx=frame_idx)
        frames_processed += 1

        # 2. Perception extraction
        free_mask, obs_mask, perc_annotated = perception.analyze_frame(frame)
        assert free_mask.shape == frame.shape[:2]
        assert obs_mask.shape == frame.shape[:2]

        # 3. World projection
        rover_x = origin_x + metrics.x
        rover_y = origin_y + metrics.z
        cur_r, cur_c = grid.metric_to_cell(rover_x, rover_y)

        # 4. Raycasting into Occupancy Grid
        grid.update_from_frame_perception(
            rover_x_m=rover_x,
            rover_y_m=rover_y,
            rover_yaw_deg=metrics.yaw_deg,
            free_mask=free_mask,
            obstacle_mask=obs_mask
        )

    reader.release()
    assert frames_processed == 12

    # Verify grid contains both observed FREE space and OCCUPIED obstacles from video
    free_count = np.sum(grid.grid == OccupancyGrid.CELL_FREE)
    obs_count = np.sum(grid.grid == OccupancyGrid.CELL_OCCUPIED)
    unk_count = np.sum(grid.grid == OccupancyGrid.CELL_UNKNOWN)

    assert free_count > 50, f"Expected >50 free cells carved from video, got {free_count}"
    assert obs_count > 0, f"Expected obstacles detected from video, got {obs_count}"
    assert unk_count > 200, f"Expected unobserved cells to remain unknown, got {unk_count}"

    # Verify dynamic A* path planning across video-derived grid
    path_result = astar_search(
        grid=grid,
        start=start_pt,
        goal=(cur_r, cur_c),
        allow_diagonal=True,
        prevent_corner_cutting=True,
        allow_unknown=True
    )

    assert path_result.found is True, f"A* path search failed: {path_result.status_message}"
    assert path_result.path_length_cells > 0
    assert path_result.path[0] == start_pt
    assert path_result.path[-1] == (cur_r, cur_c)

    # Validate path does not violate obstacles
    valid, msg = validate_path(grid, path_result.path, allow_diagonal=True, allow_unknown=True)
    assert valid is True, f"Path validation error: {msg}"


def test_video_curved_pipeline_execution():
    """Verify processing on curved recon video."""
    video_path = Path("data/sample_recon_curved.mp4")
    assert video_path.exists(), "Curved sample recon video must exist"

    reader = VideoReader(video_path)
    vo = MonocularVisualOdometry(VOConfig(num_features=600))
    perception = VideoPerceptionEngine()

    grid = OccupancyGrid(rows=32, cols=42, cell_size_m=0.5, default_state=OccupancyGrid.CELL_UNKNOWN)
    start_pt = (26, 21)
    grid.set_cell(start_pt[0], start_pt[1], OccupancyGrid.CELL_FREE)
    origin_x, origin_y = grid.cell_to_metric(start_pt[0], start_pt[1])

    cur_r, cur_c = start_pt
    for frame_idx, frame in reader.iterate_frames(step=6, max_frames=8):
        m, _ = vo.process_frame(frame, frame_idx=frame_idx)
        free_mask, obs_mask, _ = perception.analyze_frame(frame)
        rx = origin_x + m.x
        ry = origin_y + m.z
        cur_r, cur_c = grid.metric_to_cell(rx, ry)
        grid.update_from_frame_perception(rx, ry, m.yaw_deg, free_mask, obs_mask)

    reader.release()

    res = astar_search(grid, start_pt, (cur_r, cur_c), allow_unknown=True)
    assert res.found is True
    assert res.path_length_cells > 0
