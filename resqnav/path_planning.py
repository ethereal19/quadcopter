"""Heap-based A* Path Planning for 2D Occupancy Grids.

Features:
- Pure Python priority-queue A* algorithm (heapq)
- Admissible Manhattan, Euclidean, and Octile heuristics
- Strict prevention of diagonal obstacle corner-cutting
- Configurable unknown cell clearance policy
- Verification of path adjacency and traversability
"""

from dataclasses import dataclass, field
import heapq
import math
import time
from typing import Dict, List, Optional, Set, Tuple

from resqnav.occupancy_grid import OccupancyGrid


@dataclass
class PathResult:
    """Detailed search outcome and telemetry from A* solver."""
    found: bool
    path: List[Tuple[int, int]] = field(default_factory=list)
    path_length_cells: int = 0
    path_cost: float = 0.0
    explored_nodes_count: int = 0
    explored_nodes: Set[Tuple[int, int]] = field(default_factory=set)
    execution_time_ms: float = 0.0
    status_message: str = ""


def compute_heuristic(
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    heuristic_type: str = "octile"
) -> float:
    """Compute admissible heuristic distance between two cells."""
    dr = abs(r1 - r2)
    dc = abs(c1 - c2)

    if heuristic_type == "manhattan":
        return float(dr + dc)
    elif heuristic_type == "euclidean":
        return math.hypot(dr, dc)
    else:  # Octile distance for 8-connected grids with sqrt(2) diagonal cost
        d_min = min(dr, dc)
        d_max = max(dr, dc)
        return d_max + (math.sqrt(2.0) - 1.0) * d_min


def astar_search(
    grid: OccupancyGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diagonal: bool = True,
    prevent_corner_cutting: bool = True,
    allow_unknown: bool = False,
    heuristic_type: str = "octile"
) -> PathResult:
    """Execute A* pathfinding from start to goal on the provided occupancy grid.
    
    Args:
        grid: OccupancyGrid instance
        start: (row, col) start coordinates
        goal: (row, col) destination coordinates
        allow_diagonal: Enable 8-way movement
        prevent_corner_cutting: Reject diagonal steps that clip adjacent blocked corners
        allow_unknown: Whether cells with state CELL_UNKNOWN (-1) can be traversed
        heuristic_type: 'octile', 'manhattan', or 'euclidean'
        
    Returns:
        PathResult containing the reconstructed path and solver telemetry
    """
    start_time = time.perf_counter()

    # Pre-flight validation
    if not grid.is_valid(start[0], start[1]):
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return PathResult(
            found=False,
            execution_time_ms=round(elapsed, 3),
            status_message=f"Start cell {start} is out of grid boundaries (0..{grid.rows-1}, 0..{grid.cols-1})."
        )

    if not grid.is_valid(goal[0], goal[1]):
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return PathResult(
            found=False,
            execution_time_ms=round(elapsed, 3),
            status_message=f"Goal cell {goal} is out of grid boundaries (0..{grid.rows-1}, 0..{grid.cols-1})."
        )

    # Check occupancy status of start and goal
    start_state = grid.get_cell(start[0], start[1])
    goal_state = grid.get_cell(goal[0], goal[1])

    if start_state == OccupancyGrid.CELL_OCCUPIED:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return PathResult(
            found=False,
            execution_time_ms=round(elapsed, 3),
            status_message=f"Start position {start} is occupied by an obstacle."
        )

    if goal_state == OccupancyGrid.CELL_OCCUPIED:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return PathResult(
            found=False,
            execution_time_ms=round(elapsed, 3),
            status_message=f"Goal position {goal} is inside an occupied obstacle."
        )

    if not allow_unknown:
        if start_state == OccupancyGrid.CELL_UNKNOWN:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return PathResult(
                found=False,
                execution_time_ms=round(elapsed, 3),
                status_message=f"Start position {start} is in an UNKNOWN cell (allow_unknown=False)."
            )
        if goal_state == OccupancyGrid.CELL_UNKNOWN:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return PathResult(
                found=False,
                execution_time_ms=round(elapsed, 3),
                status_message=f"Goal position {goal} is in an UNKNOWN cell (allow_unknown=False)."
            )

    # Trivial case: already at destination
    if start == goal:
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return PathResult(
            found=True,
            path=[start],
            path_length_cells=1,
            path_cost=0.0,
            explored_nodes_count=1,
            explored_nodes={start},
            execution_time_ms=round(elapsed, 3),
            status_message="Start and goal are identical. Trivial zero-cost path."
        )

    # Priority queue: entries are (f_score, counter, current_node)
    counter = 0
    h_start = compute_heuristic(start[0], start[1], goal[0], goal[1], heuristic_type)
    open_heap: List[Tuple[float, int, Tuple[int, int]]] = [(h_start, counter, start)]

    # g_score maps node -> cheapest known cost from start
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}

    # came_from maps node -> best predecessor for path reconstruction
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # Set of visited/popped nodes
    closed_set: Set[Tuple[int, int]] = set()

    found_goal = False

    while open_heap:
        f_cur, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue

        closed_set.add(current)

        if current == goal:
            found_goal = True
            break

        cur_g = g_score[current]

        # Explore traversable neighbors
        neighbors = grid.get_neighbors(
            current[0],
            current[1],
            allow_diagonal=allow_diagonal,
            prevent_corner_cutting=prevent_corner_cutting,
            allow_unknown=allow_unknown
        )

        for nr, nc, move_cost in neighbors:
            neighbor = (nr, nc)
            if neighbor in closed_set:
                continue

            tentative_g = cur_g + move_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = compute_heuristic(nr, nc, goal[0], goal[1], heuristic_type)
                f = tentative_g + h
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))

    elapsed = (time.perf_counter() - start_time) * 1000.0

    if not found_goal:
        return PathResult(
            found=False,
            path=[],
            path_length_cells=0,
            path_cost=0.0,
            explored_nodes_count=len(closed_set),
            explored_nodes=closed_set,
            execution_time_ms=round(elapsed, 3),
            status_message=f"No traversable path exists between {start} and {goal}. Goal is completely blocked."
        )

    # Reconstruct optimal path from goal back to start
    path = [goal]
    curr = goal
    while curr in came_from:
        curr = came_from[curr]
        path.append(curr)
    path.reverse()

    total_cost = g_score[goal]

    return PathResult(
        found=True,
        path=path,
        path_length_cells=len(path),
        path_cost=round(total_cost, 3),
        explored_nodes_count=len(closed_set),
        explored_nodes=closed_set,
        execution_time_ms=round(elapsed, 3),
        status_message=f"Optimal path discovered: {len(path)} cells, cost {total_cost:.2f} (explored {len(closed_set)} nodes in {elapsed:.2f} ms)."
    )


def validate_path(
    grid: OccupancyGrid,
    path: List[Tuple[int, int]],
    allow_diagonal: bool = True,
    prevent_corner_cutting: bool = True,
    allow_unknown: bool = False
) -> Tuple[bool, str]:
    """Automated sanity checker verifying path continuity and traversability."""
    if not path:
        return False, "Path is empty."

    for idx, (r, c) in enumerate(path):
        if not grid.is_valid(r, c):
            return False, f"Step {idx} at {r, c} is outside grid boundaries."

        val = grid.get_cell(r, c)
        if val == OccupancyGrid.CELL_OCCUPIED:
            return False, f"Step {idx} at {r, c} crosses an OCCUPIED obstacle cell."
        if val == OccupancyGrid.CELL_UNKNOWN and not allow_unknown:
            return False, f"Step {idx} at {r, c} crosses an UNKNOWN cell with allow_unknown=False."

        if idx > 0:
            pr, pc = path[idx - 1]
            dr = abs(r - pr)
            dc = abs(c - pc)
            if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
                return False, f"Discontinuity between step {idx-1} {pr, pc} and step {idx} {r, c}."

            if dr == 1 and dc == 1:
                if not allow_diagonal:
                    return False, f"Diagonal step between {pr, pc} and {r, c} not allowed."
                if prevent_corner_cutting:
                    flank1 = grid._is_traversable(pr, c, allow_unknown)
                    flank2 = grid._is_traversable(r, pc, allow_unknown)
                    if not (flank1 and flank2):
                        return False, f"Corner-cutting collision between {pr, pc} and {r, c}."

    return True, f"Path is valid: {len(path)} contiguous traversable cells."
