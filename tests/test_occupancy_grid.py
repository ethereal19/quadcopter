"""Automated test suite for 2D Occupancy Grid Mapping."""

import numpy as np
import pytest
from resqnav.occupancy_grid import OccupancyGrid


def test_grid_initialization_and_cell_states():
    """Verify default ternary cell states and bounds checking."""
    grid = OccupancyGrid(rows=20, cols=30, cell_size_m=0.5, default_state=OccupancyGrid.CELL_UNKNOWN)

    assert grid.rows == 20
    assert grid.cols == 30
    assert grid.cell_size_m == 0.5
    assert grid.get_cell(0, 0) == OccupancyGrid.CELL_UNKNOWN
    assert grid.is_unknown(0, 0) is True

    # Mutate cell state
    grid.set_cell(5, 5, OccupancyGrid.CELL_FREE)
    assert grid.is_free(5, 5) is True
    assert grid.get_cell(5, 5) == OccupancyGrid.CELL_FREE

    grid.set_cell(5, 6, OccupancyGrid.CELL_OCCUPIED)
    assert grid.is_occupied(5, 6) is True
    assert grid.get_cell(5, 6) == OccupancyGrid.CELL_OCCUPIED


def test_grid_boundaries():
    """Verify out-of-bounds queries return occupied safely."""
    grid = OccupancyGrid(rows=10, cols=10)

    assert grid.is_valid(0, 0) is True
    assert grid.is_valid(9, 9) is True
    assert grid.is_valid(-1, 0) is False
    assert grid.is_valid(0, 10) is False
    assert grid.is_valid(10, 10) is False

    # Out of bounds access defaults to occupied to prevent out-of-bounds routing
    assert grid.get_cell(-1, 5) == OccupancyGrid.CELL_OCCUPIED
    assert grid.is_occupied(-1, 5) is True


def test_metric_conversions():
    """Verify conversion between cell coordinates and continuous metric values."""
    cell_size = 0.5
    grid = OccupancyGrid(rows=10, cols=10, cell_size_m=cell_size)

    # Convert cell (9, 0) [bottom-left corner]
    x, y = grid.cell_to_metric(9, 0)
    assert abs(x - 0.25) < 1e-4
    assert abs(y - 0.25) < 1e-4

    # Convert metric back to cell
    r, c = grid.metric_to_cell(x, y)
    assert r == 9
    assert c == 0


def test_block_fill():
    """Verify rectangular block region setting."""
    grid = OccupancyGrid(rows=15, cols=15)
    grid.clear(default_state=OccupancyGrid.CELL_FREE)

    grid.set_block(2, 2, 6, 6, OccupancyGrid.CELL_OCCUPIED)

    # Check corners and interior
    assert grid.is_occupied(2, 2) is True
    assert grid.is_occupied(6, 6) is True
    assert grid.is_occupied(4, 4) is True
    # Outside should be free
    assert grid.is_free(1, 1) is True
    assert grid.is_free(7, 7) is True


def test_mode_a_presets():
    """Verify Mode A benchmark preset maps instantiate valid scenarios."""
    rubble = OccupancyGrid.create_disaster_rubble_map(rows=20, cols=20)
    assert "Disaster Rubble" in rubble.mode_label
    assert rubble.is_occupied(1, 1) is True  # Perimeter wall

    warehouse = OccupancyGrid.create_warehouse_rescue_map(rows=20, cols=20)
    assert "Warehouse Depot" in warehouse.mode_label

    cave = OccupancyGrid.create_cave_tunnel_map(rows=20, cols=20)
    assert "Cave" in cave.mode_label


def test_mode_b_experimental_video_grid():
    """Verify Mode B grid updates from visual odometry poses while leaving unobserved cells unknown."""
    # Synthetic camera trajectory
    traj = [
        (0.0, 0.0, True),
        (0.0, 1.0, True),
        (0.2, 2.0, True),
        (0.5, 3.0, True),
    ]

    grid = OccupancyGrid.create_video_derived_grid(traj, rows=20, cols=20)

    assert "Mode B" in grid.mode_label

    # Check that some cells along trajectory were observed as FREE
    free_cells_count = np.sum(grid.grid == OccupancyGrid.CELL_FREE)
    assert free_cells_count > 0

    # Check that unobserved peripheral cells remain UNKNOWN (-1)
    unknown_cells_count = np.sum(grid.grid == OccupancyGrid.CELL_UNKNOWN)
    assert unknown_cells_count > 0
    assert grid.is_unknown(0, 0) is True  # Far corner unobserved


def test_update_from_frame_perception():
    """Verify that update_from_frame_perception marks ground as FREE and detected obstacles as OCCUPIED."""
    grid = OccupancyGrid(rows=30, cols=30, cell_size_m=0.5, default_state=OccupancyGrid.CELL_UNKNOWN)
    
    # Rover is at metric (7.5, 3.0), heading 0 degrees (facing forward)
    h, w = 240, 320
    free_mask = np.zeros((h, w), dtype=np.uint8)
    obstacle_mask = np.zeros((h, w), dtype=np.uint8)

    # Lower half has free ground
    free_mask[120:, :] = 255
    # Obstacle cluster on the right side of the screen at row 160
    obstacle_mask[150:180, 240:280] = 255
    free_mask[150:180, 240:280] = 0

    grid.update_from_frame_perception(
        rover_x_m=7.5,
        rover_y_m=3.0,
        rover_yaw_deg=0.0,
        free_mask=free_mask,
        obstacle_mask=obstacle_mask,
        fov_deg=60.0,
        max_range_m=6.0
    )

    # Check that free cells were mapped in front of the rover
    free_count = np.sum(grid.grid == OccupancyGrid.CELL_FREE)
    assert free_count > 5

    # Check that obstacle cells were mapped
    obs_count = np.sum(grid.grid == OccupancyGrid.CELL_OCCUPIED)
    assert obs_count > 0

    # Check that distant unobserved cells remain Unknown (-1)
    assert grid.is_unknown(0, 0) is True
