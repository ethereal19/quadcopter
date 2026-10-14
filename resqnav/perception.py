"""Video perception module: extracts traversable ground and obstacles from video frames.

Supports two perception backends:
1. Neural Semantic Segmentation: Uses BiSeNetV2 checkpoint (model_final.pth) for 4-class terrain classification.
2. Fast Computer Vision Extractor: Pure OpenCV edge/contrast/color ground segmentation (zero external weight dependencies).
"""

from pathlib import Path
import math
from typing import Optional, Tuple
import cv2
import numpy as np


class VideoPerceptionEngine:
    """Extracts traversable ground and obstacle masks from RGB video frames."""

    def __init__(
        self,
        use_neural: bool = False,
        confidence_threshold: float = 0.60,
        floor_fraction: float = 0.40
    ):
        self.use_neural = use_neural
        self.confidence_threshold = confidence_threshold
        self.floor_fraction = floor_fraction
        self.neural_model = None
        self.device = None

        if self.use_neural:
            self._init_neural_engine()

    def _init_neural_engine(self):
        """Attempt to load BiSeNetV2 neural weights if available."""
        try:
            import torch
            from pathlib import Path
            import sys
            
            project_root = Path(__file__).resolve().parents[1]
            seg_dir = project_root / "semantic segmentation"
            model_file = seg_dir / "bisenetv2.py"
            weights_file = seg_dir / "model_final.pth"

            if model_file.is_file() and weights_file.is_file():
                if str(seg_dir) not in sys.path:
                    sys.path.insert(0, str(seg_dir))
                from segmentation import load_segmentation_model, SegmentationEngine
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                model = load_segmentation_model(model_file, weights_file, self.device)
                self.neural_model = SegmentationEngine(
                    model,
                    self.device,
                    confidence=self.confidence_threshold,
                    floor_fraction=self.floor_fraction
                )
        except Exception:
            self.neural_model = None

    def analyze_frame(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Analyze a frame and extract (traversable_mask, obstacle_mask, visual_overlay).
        
        Returns:
            traversable_mask: (H, W) uint8 binary mask (255 = Free, 0 = Not Free)
            obstacle_mask: (H, W) uint8 binary mask (255 = Obstacle, 0 = Clear)
            annotated_frame: (H, W, 3) BGR frame with green ground and red obstacle overlays
        """
        h, w = frame_bgr.shape[:2]

        if self.use_neural and self.neural_model is not None:
            return self._analyze_neural(frame_bgr)
        else:
            return self._analyze_opencv(frame_bgr)

    def _analyze_neural(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run BiSeNetV2 inference on the frame."""
        h, w = frame_bgr.shape[:2]
        color_map, free_mask = self.neural_model.predict(frame_bgr)

        # In BiSeNet: 0=Sky/Background, 1=Traversable, 2=Fixed Obstacle, 3=Vegetation
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)
        # Anything in bottom 65% of frame that is NOT free ground is an obstacle/boundary
        horizon_cutoff = int(h * 0.35)
        obstacle_mask[horizon_cutoff:] = (free_mask[horizon_cutoff:] == 0).astype(np.uint8) * 255

        # Visual overlay blending
        annotated = frame_bgr.copy()
        # Tint free ground green
        ground_idx = free_mask > 0
        annotated[ground_idx] = cv2.addWeighted(annotated[ground_idx], 0.65, np.full_like(annotated[ground_idx], (16, 185, 129)), 0.35, 0)
        # Tint obstacles red
        obs_idx = obstacle_mask > 0
        annotated[obs_idx] = cv2.addWeighted(annotated[obs_idx], 0.65, np.full_like(annotated[obs_idx], (239, 68, 68)), 0.35, 0)

        return free_mask, obstacle_mask, annotated

    def _analyze_opencv(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast, robust OpenCV vision-based ground and obstacle segmentation.
        
        Technique:
        - Analyzes ground corridor in bottom half of image.
        - Computes edge gradient energy and color variance.
        - Identifies high-contrast vertical rubble boundaries as obstacles.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Smooth to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection for rubble boundaries and vertical obstacles
        edges = cv2.Canny(blurred, 40, 120)

        # Dilate edges to form solid obstacle clusters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)

        # Ground plane horizon line (e.g., bottom 55% of image is reachable terrain)
        horizon_y = int(h * 0.42)

        free_mask = np.zeros((h, w), dtype=np.uint8)
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)

        # Below horizon, low-edge regions are free traversable ground
        ground_roi = dilated_edges[horizon_y:, :]
        free_roi = (ground_roi == 0).astype(np.uint8) * 255
        obs_roi = (ground_roi > 0).astype(np.uint8) * 255

        free_mask[horizon_y:, :] = free_roi
        obstacle_mask[horizon_y:, :] = obs_roi

        # Morphological cleanup on free space
        free_mask = cv2.morphologyEx(free_mask, cv2.MORPH_OPEN, kernel)

        # Create blended visual annotation
        annotated = frame_bgr.copy()
        # Tint traversable ground with emerald green
        green_tint = np.full_like(annotated, (16, 185, 129))
        annotated = np.where(free_mask[:, :, None] > 0, cv2.addWeighted(annotated, 0.70, green_tint, 0.30, 0), annotated)

        # Tint obstacle rubble with hazard crimson
        red_tint = np.full_like(annotated, (239, 68, 68))
        annotated = np.where(obstacle_mask[:, :, None] > 0, cv2.addWeighted(annotated, 0.65, red_tint, 0.35, 0), annotated)

        return free_mask, obstacle_mask, annotated
