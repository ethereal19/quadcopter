"""Ternary 2D Occupancy Grid Mapping for Autonomous Search & Rescue.

Supports:
- Three cell states: Free (0), Occupied (1), Unknown (-1)
- Mode A: Demonstration Maps (Simulated benchmark environments, clearly labelled)
- Mode B: Experimental Video-Derived Maps (Observation count logging with unknown zones)
"""

from typing import Dict, List, Optional, Set, Tuple
import math
import numpy as np

from resqnav.config import GridConfig


class OccupancyGrid:
    """2D Grid Map with ternary occupancy states and metric coordinate projections."""

    CELL_UNKNOWN: int = -1
    CELL_FREE: int = 0
    CELL_OCCUPIED: int = 1

    def __init__(
        self,
        rows: int = 30,
        cols: int = 40,
        cell_size_m: float = 0.5,
        default_state: int = CELL_UNKNOWN,
        mode_label: str = "Simulated Demonstration Map (Not from video)"
    ):
        self.rows = max(5, int(rows))
        self.cols = max(5, int(cols))
        self.cell_size_m = float(cell_size_m)
        self.mode_label = mode_label

        # Grid matrix: -1=Unknown, 0=Free, 1=Occupied
        self.grid = np.full((self.rows, self.cols), default_state, dtype=np.int8)

        # Observation confidence/hit counter matrix
        self.observation_counts = np.zeros((self.rows, self.cols), dtype=np.int32)

    def is_valid(self, r: int, c: int) -> bool:
        """Check if cell coordinate (r, c) is within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_cell(self, r: int, c: int) -> int:
        """Get cell state. Returns CELL_OCCUPIED if out of bounds."""
        if not self.is_valid(r, c):
            return self.CELL_OCCUPIED
        return int(self.grid[r, c])

    def set_cell(self, r: int, c: int, state: int, hit_count: int = 1):
        """Set cell state and increment observation count."""
        if self.is_valid(r, c):
            self.grid[r, c] = state
            self.observation_counts[r, c] += hit_count

    def set_block(self, r_start: int, c_start: int, r_end: int, c_end: int, state: int):
        """Fill a rectangular bounding box with the given state."""
        r0 = max(0, min(r_start, r_end))
        r1 = min(self.rows - 1, max(r_start, r_end))
        c0 = max(0, min(c_start, c_end))
        c1 = min(self.cols - 1, max(c_start, c_end))
        self.grid[r0:r1 + 1, c0:c1 + 1] = state
        self.observation_counts[r0:r1 + 1, c0:c1 + 1] += 1

    def is_free(self, r: int, c: int) -> bool:
        return bool(self.is_valid(r, c) and self.grid[r, c] == self.CELL_FREE)

    def is_occupied(self, r: int, c: int) -> bool:
        return bool(not self.is_valid(r, c) or self.grid[r, c] == self.CELL_OCCUPIED)

    def is_unknown(self, r: int, c: int) -> bool:
        return bool(self.is_valid(r, c) and self.grid[r, c] == self.CELL_UNKNOWN)

    def cell_to_metric(self, r: int, c: int) -> Tuple[float, float]:
        """Convert cell row and column to metric (X, Y) world coordinates."""
        x = (c + 0.5) * self.cell_size_m
        y = (self.rows - 1 - r + 0.5) * self.cell_size_m
        return round(x, 3), round(y, 3)

    def metric_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert metric (X, Y) coordinates to nearest cell (row, col)."""
        c = int(np.floor(x / self.cell_size_m))
        r = int(self.rows - 1 - np.floor(y / self.cell_size_m))
        return r, c

    def get_neighbors(
        self,
        r: int,
        c: int,
        allow_diagonal: bool = True,
        prevent_corner_cutting: bool = True,
        allow_unknown: bool = False
    ) -> List[Tuple[int, int, float]]:
        """Return valid adjacent neighbor cells as (row, col, movement_cost).
        
        Diagonal corner-cutting prevention rule:
        If moving diagonally to (r + dr, c + dc), both adjacent orthogonal cells
        (r + dr, c) and (r, c + dc) MUST NOT be blocked.
        """
        neighbors = []

        # 4 Orthogonal directions (Cost = 1.0)
        orthogonals = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in orthogonals:
            nr, nc = r + dr, c + dc
            if self._is_traversable(nr, nc, allow_unknown):
                neighbors.append((nr, nc, 1.0))

        if not allow_diagonal:
            return neighbors

        # 4 Diagonal directions (Cost = sqrt(2) ≈ 1.414)
        diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in diagonals:
            nr, nc = r + dr, c + dc
            if not self._is_traversable(nr, nc, allow_unknown):
                continue

            if prevent_corner_cutting:
                # Both orthogonal flank cells must be traversable
                flank1 = self._is_traversable(r + dr, c, allow_unknown)
                flank2 = self._is_traversable(r, c + dc, allow_unknown)
                if not (flank1 and flank2):
                    continue

            neighbors.append((nr, nc, 1.41421356))

        return neighbors

    def _is_traversable(self, r: int, c: int, allow_unknown: bool) -> bool:
        if not self.is_valid(r, c):
            return False
        val = self.grid[r, c]
        if val == self.CELL_FREE:
            return True
        if val == self.CELL_UNKNOWN:
            return bool(allow_unknown)
        return False

    def clear(self, default_state: int = CELL_UNKNOWN):
        """Reset grid to empty state."""
        self.grid.fill(default_state)
        self.observation_counts.fill(0)

    # --------------------------------------------------------------------------
    # MODE A: DEMONSTRATION PRESET MAPS (Simulated Benchmark Environments)
    # --------------------------------------------------------------------------

    @classmethod
    def create_disaster_rubble_map(cls, rows: int = 30, cols: int = 40) -> "OccupancyGrid":
        """Mode A: Collapsed earthquake rubble benchmark scenario with navigation corridors."""
        grid = cls(rows, cols, mode_label="Mode A: Disaster Rubble Simulation Map (Pre-configured)")
        # Initialize interior as Free space with Unknown outer periphery
        grid.grid.fill(cls.CELL_UNKNOWN)
        grid.set_block(2, 2, rows - 3, cols - 3, cls.CELL_FREE)

        # Boundary perimeter collapse walls (Occupied)
        grid.set_block(1, 1, 1, cols - 2, cls.CELL_OCCUPIED)
        grid.set_block(rows - 2, 1, rows - 2, cols - 2, cls.CELL_OCCUPIED)
        grid.set_block(1, 1, rows - 2, 1, cls.CELL_OCCUPIED)
        grid.set_block(1, cols - 2, rows - 2, cols - 2, cls.CELL_OCCUPIED)

        # Collapsed concrete beam 1
        grid.set_block(5, 8, 18, 10, cls.CELL_OCCUPIED)
        # Fallen slab 2
        grid.set_block(14, 18, 25, 20, cls.CELL_OCCUPIED)
        # Central rubble pile
        grid.set_block(8, 22, 12, 32, cls.CELL_OCCUPIED)
        # Debris cluster
        grid.set_block(20, 26, 26, 30, cls.CELL_OCCUPIED)
        grid.set_block(4, 34, 10, 36, cls.CELL_OCCUPIED)

        return grid

    @classmethod
    def create_warehouse_rescue_map(cls, rows: int = 30, cols: int = 40) -> "OccupancyGrid":
        """Mode A: Industrial storage warehouse with fallen shelving aisles."""
        grid = cls(rows, cols, mode_label="Mode A: Warehouse Depot Simulation Map (Pre-configured)")
        grid.grid.fill(cls.CELL_FREE)

        # Boundary walls
        grid.set_block(0, 0, 0, cols - 1, cls.CELL_OCCUPIED)
        grid.set_block(rows - 1, 0, rows - 1, cols - 1, cls.CELL_OCCUPIED)
        grid.set_block(0, 0, rows - 1, 0, cls.CELL_OCCUPIED)
        grid.set_block(0, cols - 1, rows - 1, cols - 1, cls.CELL_OCCUPIED)

        # Shelving aisles with occasional collapsed choke points
        for c in range(6, cols - 6, 8):
            grid.set_block(4, c, rows - 10, c + 1, cls.CELL_OCCUPIED)
            # Gap in middle for forklift transit
            grid.set_block(13, c, 17, c + 1, cls.CELL_FREE)

        # Pallet stacks
        grid.set_block(22, 10, 25, 14, cls.CELL_OCCUPIED)
        grid.set_block(22, 24, 25, 28, cls.CELL_OCCUPIED)

        # Partially unmapped zone (Unknown) in top right corner
        grid.set_block(2, cols - 12, 9, cols - 3, cls.CELL_UNKNOWN)
        return grid

    @classmethod
    def create_cave_tunnel_map(cls, rows: int = 30, cols: int = 40) -> "OccupancyGrid":
        """Mode A: Subterranean cave passage with stalagmites and cavern chambers."""
        grid = cls(rows, cols, mode_label="Mode A: Subterranean Cave Simulation Map (Pre-configured)")
        # Entire map begins as solid rock (Occupied)
        grid.grid.fill(cls.CELL_OCCUPIED)

        # Carve out a winding corridor (Free)
        for c in range(2, cols - 3):
            # Sinusoidal winding path
            center_r = int(rows / 2 + 6 * np.sin(c * 0.25))
            grid.set_block(center_r - 2, c, center_r + 2, c, cls.CELL_FREE)

        # Carve a cavern chamber in the middle
        grid.set_block(8, 16, 22, 26, cls.CELL_FREE)
        # Stalagmite in center of cavern
        grid.set_block(13, 20, 17, 22, cls.CELL_OCCUPIED)

        # Side branch leading to dead end
        grid.set_block(18, 24, 26, 26, cls.CELL_FREE)
        grid.set_block(25, 24, 27, 34, cls.CELL_FREE)

        return grid

    # --------------------------------------------------------------------------
    # MODE B: EXPERIMENTAL VIDEO-DERIVED MAP (Observation-backed mapping)
    # --------------------------------------------------------------------------

    def update_from_frame_perception(
        self,
        rover_x_m: float,
        rover_y_m: float,
        rover_yaw_deg: float,
        free_mask: Optional[np.ndarray],
        obstacle_mask: Optional[np.ndarray],
        fov_deg: float = 65.0,
        max_range_m: float = 8.0,
        num_rays: int = 28
    ):
        """Update occupancy grid by projecting camera FOV ground and obstacles from video frames.
        
        Args:
            rover_x_m: Rover current world X metric coordinate.
            rover_y_m: Rover current world Y metric coordinate (forward axis).
            rover_yaw_deg: Heading angle in degrees (0 = forward, positive = clockwise/right).
            free_mask: Binary mask (H, W) where 255 represents traversable ground.
            obstacle_mask: Binary mask (H, W) where 255 represents obstacles / rubble.
            fov_deg: Camera horizontal field of view in degrees.
            max_range_m: Maximum sensing projection distance.
            num_rays: Number of angular ray sectors cast across the FOV.
        """
        # Always mark rover immediate footprint as Free
        r_cur, c_cur = self.metric_to_cell(rover_x_m, rover_y_m)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if self.is_valid(r_cur + dr, c_cur + dc):
                    self.set_cell(r_cur + dr, c_cur + dc, self.CELL_FREE, hit_count=2)

        if free_mask is None or obstacle_mask is None:
            return

        h, w = free_mask.shape[:2]
        horizon_y = int(h * 0.40)
        h_cam_fy = 220.0  # Virtual camera baseline height * focal length

        for i in range(num_rays):
            u = int((i + 0.5) / num_rays * w)
            bearing_deg = ((u - w / 2.0) / (w / 2.0)) * (fov_deg / 2.0)
            ray_angle_rad = math.radians(rover_yaw_deg + bearing_deg)

            # Analyze column u from horizon to bottom of frame
            obs_col = obstacle_mask[horizon_y:, u]
            free_col = free_mask[horizon_y:, u]

            obs_indices = np.where(obs_col > 0)[0]
            if len(obs_indices) > 0:
                # Closest obstacle to rover in this column (largest row index)
                closest_obs_rel_row = int(np.max(obs_indices))
                delta_v = max(2, closest_obs_rel_row)
                d_obs = min(max_range_m, max(0.8, h_cam_fy / delta_v))
                d_free = max(0.0, d_obs - self.cell_size_m * 0.8)
            else:
                d_obs = None
                free_indices = np.where(free_col > 0)[0]
                if len(free_indices) > 0:
                    far_free_rel_row = int(np.min(free_indices))
                    delta_v = max(2, far_free_rel_row)
                    d_free = min(max_range_m, max(0.8, h_cam_fy / delta_v))
                else:
                    d_free = 1.8  # Minimal forward clearance

            # Raycast along free space
            ds = self.cell_size_m * 0.5
            s = 0.3
            while s <= d_free:
                rx = rover_x_m + s * math.sin(ray_angle_rad)
                ry = rover_y_m + s * math.cos(ray_angle_rad)
                r, c = self.metric_to_cell(rx, ry)
                if self.is_valid(r, c):
                    # Only overwrite if not heavily confirmed occupied
                    if self.grid[r, c] != self.CELL_OCCUPIED or self.observation_counts[r, c] < 3:
                        self.set_cell(r, c, self.CELL_FREE, hit_count=1)
                s += ds

            # Mark obstacle cell if detected within range
            if d_obs is not None and d_obs < max_range_m:
                ox = rover_x_m + d_obs * math.sin(ray_angle_rad)
                oy = rover_y_m + d_obs * math.cos(ray_angle_rad)
                r_obs, c_obs = self.metric_to_cell(ox, oy)
                if self.is_valid(r_obs, c_obs):
                    self.set_cell(r_obs, c_obs, self.CELL_OCCUPIED, hit_count=2)

    @classmethod
    def create_video_derived_grid(
        cls,
        trajectory: Optional[List[Tuple[float, float, bool]]] = None,
        rows: int = 32,
        cols: int = 42,
        cell_size_m: float = 0.5
    ) -> "OccupancyGrid":
        """Mode B: Experimental mapping driven by visual odometry observation history.
        
        CRITICAL HONESTY NOTICE:
        Without dense stereo disparity, depth camera, or wheel encoders, true metric
        depth projection is unconstrained. This experimental interface projects
        the estimated camera corridor as 'Observed Free' space and marks unobserved
        sectors strictly as 'Unknown (-1)'. No metric map is fabricated.
        """
        grid = cls(
            rows=rows,
            cols=cols,
            cell_size_m=cell_size_m,
            default_state=cls.CELL_UNKNOWN,
            mode_label="Mode B: Video-Derived Occupancy Grid Map"
        )

        if not trajectory:
            return grid

        # Center camera trajectory in the grid
        start_r = int(rows * 0.82)
        start_c = int(cols * 0.5)

        for x_rel, z_rel, ok in trajectory:
            if not ok:
                continue

            dr = int(round(-z_rel * 2.0))
            dc = int(round(x_rel * 2.0))

            r = start_r + dr
            c = start_c + dc

            # Mark camera footprint and immediate visual cone as Free
            for or_offset in (-1, 0, 1):
                for oc_offset in (-1, 0, 1):
                    nr, nc = r + or_offset, c + oc_offset
                    if grid.is_valid(nr, nc):
                        grid.set_cell(nr, nc, cls.CELL_FREE, hit_count=1)

        return grid
