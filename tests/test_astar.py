"""Automated test suite for A* Path Planning algorithm."""

import pytest
from resqnav.occupancy_grid import OccupancyGrid
from resqnav.path_planning import astar_search, compute_heuristic, validate_path


@pytest.fixture
def empty_grid():
    """Create a 10x10 free occupancy grid."""
    grid = OccupancyGrid(rows=10, cols=10, cell_size_m=0.5)
    grid.clear(default_state=OccupancyGrid.CELL_FREE)
    return grid


@pytest.fixture
def blocked_grid():
    """Create a 10x10 grid with a solid wall dividing left and right halves."""
    grid = OccupancyGrid(rows=10, cols=10, cell_size_m=0.5)
    grid.clear(default_state=OccupancyGrid.CELL_FREE)
    # Solid vertical wall from top to bottom at col 5
    grid.set_block(0, 5, 9, 5, OccupancyGrid.CELL_OCCUPIED)
    return grid


def test_path_in_empty_grid(empty_grid):
    """Test A* finds valid direct path in an obstacle-free grid."""
    start = (1, 1)
    goal = (8, 8)
    result = astar_search(empty_grid, start, goal, allow_diagonal=True)

    assert result.found is True
    assert len(result.path) > 0
    assert result.path[0] == start
    assert result.path[-1] == goal

    # Validate adjacency and traversability
    is_valid, msg = validate_path(empty_grid, result.path, allow_diagonal=True)
    assert is_valid is True, msg


def test_path_avoids_obstacles():
    """Test A* navigates around a central obstacle."""
    grid = OccupancyGrid(rows=15, cols=15, cell_size_m=0.5)
    grid.clear(default_state=OccupancyGrid.CELL_FREE)

    # Place a 5x5 wall in the center
    grid.set_block(5, 5, 9, 9, OccupancyGrid.CELL_OCCUPIED)

    start = (7, 2)
    goal = (7, 12)
    result = astar_search(grid, start, goal, allow_diagonal=True)

    assert result.found is True
    # Ensure no step in the path touches any cell of the obstacle
    for r, c in result.path:
        assert not (5 <= r <= 9 and 5 <= c <= 9), f"Path intersected obstacle at ({r}, {c})"

    is_valid, msg = validate_path(grid, result.path, allow_diagonal=True)
    assert is_valid is True, msg


def test_unreachable_goal_returns_no_path(blocked_grid):
    """Test A* cleanly reports failure when goal is walled off."""
    start = (5, 2)
    goal = (5, 8)  # Across the solid wall at col 5
    result = astar_search(blocked_grid, start, goal, allow_diagonal=True)

    assert result.found is False
    assert result.path == []
    assert result.path_length_cells == 0
    assert "No traversable path" in result.status_message


def test_invalid_start_or_goal(empty_grid):
    """Test A* handles out-of-bounds start and goal gracefully."""
    # Out of bounds start
    res_start_oob = astar_search(empty_grid, (-1, 5), (5, 5))
    assert res_start_oob.found is False
    assert "out of grid boundaries" in res_start_oob.status_message

    # Out of bounds goal
    res_goal_oob = astar_search(empty_grid, (2, 2), (20, 20))
    assert res_goal_oob.found is False
    assert "out of grid boundaries" in res_goal_oob.status_message


def test_occupied_start_or_goal():
    """Test A* fails cleanly if start or goal is placed directly on an obstacle."""
    grid = OccupancyGrid(rows=10, cols=10)
    grid.clear(default_state=OccupancyGrid.CELL_FREE)
    grid.set_cell(2, 2, OccupancyGrid.CELL_OCCUPIED)
    grid.set_cell(8, 8, OccupancyGrid.CELL_OCCUPIED)

    # Start on obstacle
    res_start_occ = astar_search(grid, (2, 2), (5, 5))
    assert res_start_occ.found is False
    assert "occupied" in res_start_occ.status_message

    # Goal on obstacle
    res_goal_occ = astar_search(grid, (5, 5), (8, 8))
    assert res_goal_occ.found is False
    assert "occupied" in res_goal_occ.status_message


def test_unknown_cells_policy():
    """Test handling of CELL_UNKNOWN (-1) cells under allow_unknown flag."""
    grid = OccupancyGrid(rows=10, cols=10)
    grid.clear(default_state=OccupancyGrid.CELL_FREE)

    # Place a strip of unknown cells at col 5
    grid.set_block(0, 5, 9, 5, OccupancyGrid.CELL_UNKNOWN)

    start = (5, 2)
    goal = (5, 8)

    # Case 1: allow_unknown=False (default safe SAR policy) -> No path
    res_safe = astar_search(grid, start, goal, allow_unknown=False)
    assert res_safe.found is False

    # Case 2: allow_unknown=True (exploratory policy) -> Path found through unknown strip
    res_explore = astar_search(grid, start, goal, allow_unknown=True)
    assert res_explore.found is True
    assert len(res_explore.path) > 0


def test_diagonal_corner_cutting_prevention():
    """Test that diagonal moves cannot cut through adjacent obstacle corners."""
    grid = OccupancyGrid(rows=10, cols=10)
    grid.clear(default_state=OccupancyGrid.CELL_FREE)

    # Setup a diagonal squeeze:
    # (3, 3) is Free, (4, 4) is Free
    # (3, 4) is Occupied, (4, 3) is Occupied
    grid.set_cell(3, 4, OccupancyGrid.CELL_OCCUPIED)
    grid.set_cell(4, 3, OccupancyGrid.CELL_OCCUPIED)

    start = (3, 3)
    goal = (4, 4)

    # With corner-cutting prevented, diagonal from (3,3) to (4,4) is forbidden
    neighbors = grid.get_neighbors(3, 3, allow_diagonal=True, prevent_corner_cutting=True)
    neighbor_coords = [(nr, nc) for nr, nc, _ in neighbors]
    assert (4, 4) not in neighbor_coords, "Diagonal step clipped adjacent obstacle corners!"

    # With corner-cutting allowed, (4, 4) would be present
    neighbors_lax = grid.get_neighbors(3, 3, allow_diagonal=True, prevent_corner_cutting=False)
    neighbor_coords_lax = [(nr, nc) for nr, nc, _ in neighbors_lax]
    assert (4, 4) in neighbor_coords_lax


def test_path_adjacency_and_validity(empty_grid):
    """Test that every returned cell is traversable and consecutive cells are strictly adjacent."""
    start = (2, 3)
    goal = (7, 6)
    result = astar_search(empty_grid, start, goal, allow_diagonal=True)

    assert result.found is True
    is_valid, msg = validate_path(empty_grid, result.path, allow_diagonal=True)
    assert is_valid is True, msg


def test_same_start_and_goal(empty_grid):
    """Test trivial search when start equals goal."""
    pos = (4, 4)
    result = astar_search(empty_grid, pos, pos)

    assert result.found is True
    assert result.path == [pos]
    assert result.path_cost == 0.0
    assert result.path_length_cells == 1


def test_heuristic_admissibility():
    """Test that octile and euclidean heuristics are consistent and never overestimate."""
    h_oct = compute_heuristic(0, 0, 3, 4, "octile")
    h_euc = compute_heuristic(0, 0, 3, 4, "euclidean")
    h_man = compute_heuristic(0, 0, 3, 4, "manhattan")

    # For 3, 4: euclidean = 5.0, manhattan = 7.0
    assert abs(h_euc - 5.0) < 1e-6
    assert abs(h_man - 7.0) < 1e-6
    assert h_oct <= h_man
