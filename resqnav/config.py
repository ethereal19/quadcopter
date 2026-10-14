"""Configuration schemas and default parameters for ResQNav."""

from dataclasses import dataclass
import numpy as np


@dataclass
class VOConfig:
    """Configuration for monocular visual odometry estimation."""
    num_features: int = 1500
    scale_factor: float = 1.2
    n_levels: int = 8
    match_ratio: float = 0.75
    ransac_threshold: float = 1.0
    ransac_prob: float = 0.999
    min_features_tracked: int = 15
    focal_length: float = 718.856
    cx: float = 320.0
    cy: float = 240.0
    # Educational scale: In monocular VO, translation vector norm is 1.0 without metric depth.
    relative_step_scale: float = 1.0

    def get_camera_matrix(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Construct a 3x3 pinhole intrinsic camera matrix K."""
        fx = self.focal_length
        fy = self.focal_length
        cx = width / 2.0 if self.cx is None else self.cx
        cy = height / 2.0 if self.cy is None else self.cy
        return np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


@dataclass
class GridConfig:
    """Configuration for 2D ternary occupancy grid mapping."""
    rows: int = 30
    cols: int = 40
    cell_size_m: float = 0.5
    
    # Ternary cell values
    CELL_UNKNOWN: int = -1
    CELL_FREE: int = 0
    CELL_OCCUPIED: int = 1


@dataclass
class PlannerConfig:
    """Configuration for A* graph search path planning."""
    allow_diagonal: bool = True
    prevent_corner_cutting: bool = True
    allow_unknown: bool = False
    heuristic: str = "octile"  # 'octile', 'manhattan', 'euclidean'


@dataclass
class AppConfig:
    """General application settings."""
    default_frame_step: int = 2
    max_frames_limit: int = 500
    sample_video_path: str = "data/sample_recon.mp4"
