"""Monocular Visual Odometry using ORB, RANSAC, and Essential Matrix decomposition.

EDUCATIONAL PROTOTYPE NOTE:
Monocular visual odometry suffers from fundamental scale ambiguity: the absolute magnitude
of translation cannot be recovered from visual correspondences alone without a metric reference
(such as stereo baseline, ground truth camera height, wheel odometry, or IMU integration).
All trajectory coordinates are reported in ARBITRARY RELATIVE UNITS, not physical metres.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

from resqnav.config import VOConfig


@dataclass
class VOMetrics:
    """Telemetry data generated per processed frame."""
    frame_idx: int
    status: str
    tracking_ok: bool
    num_keypoints: int
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    x: float
    z: float
    yaw_deg: float
    trajectory: List[Tuple[float, float, bool]] = field(default_factory=list)
    diagnostic_message: str = ""


class MonocularVisualOdometry:
    """Monocular Visual Odometry engine using ORB keypoint tracking."""

    def __init__(self, config: Optional[VOConfig] = None):
        self.config = config or VOConfig()
        self.orb = cv2.ORB_create(
            nfeatures=self.config.num_features,
            scaleFactor=self.config.scale_factor,
            nlevels=self.config.n_levels,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=18,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Camera intrinsics matrix K
        self.K: Optional[np.ndarray] = None

        # Previous frame state
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kps: Optional[List[cv2.KeyPoint]] = None
        self.prev_descs: Optional[np.ndarray] = None

        # Cumulative pose in world coordinate system (X: lateral right, Y: vertical down, Z: optical forward)
        self.R_world: np.ndarray = np.eye(3, dtype=np.float64)
        self.t_world: np.ndarray = np.zeros((3, 1), dtype=np.float64)

        # Trajectory history: list of (x, z, tracking_ok)
        self.trajectory: List[Tuple[float, float, bool]] = [(0.0, 0.0, True)]
        self.all_metrics: List[VOMetrics] = []

    def reset(self):
        """Reset trajectory and tracking state."""
        self.prev_gray = None
        self.prev_kps = None
        self.prev_descs = None
        self.R_world = np.eye(3, dtype=np.float64)
        self.t_world = np.zeros((3, 1), dtype=np.float64)
        self.trajectory = [(0.0, 0.0, True)]
        self.all_metrics.clear()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 0
    ) -> Tuple[VOMetrics, np.ndarray]:
        """Process a single video frame and estimate relative camera displacement.
        
        Returns:
            (metrics, annotated_frame_bgr)
        """
        h, w = frame_bgr.shape[:2]
        if self.K is None:
            self.K = self.config.get_camera_matrix(w, h)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        annotated = frame_bgr.copy()

        # Step 1: Detect ORB features & descriptors
        kps, descs = self.orb.detectAndCompute(gray, None)
        num_kps = len(kps) if kps is not None else 0

        # Check for low-texture / dark / blurred scenes with too few features
        if num_kps < self.config.min_features_tracked or descs is None:
            return self._record_failure(
                frame_idx,
                num_kps,
                status="INSUFFICIENT_FEATURES",
                msg=f"Only {num_kps} features detected (minimum required: {self.config.min_features_tracked}). Motion estimation skipped."
            ), annotated

        # Handle initial reference keyframe
        if self.prev_gray is None or self.prev_descs is None or self.prev_kps is None:
            self.prev_gray = gray
            self.prev_kps = kps
            self.prev_descs = descs

            # Draw detected initial keypoints
            if kps:
                cv2.drawKeypoints(
                    annotated, kps, annotated,
                    color=(56, 189, 248),
                    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
                )

            metrics = VOMetrics(
                frame_idx=frame_idx,
                status="INITIALIZED",
                tracking_ok=True,
                num_keypoints=num_kps,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                x=float(self.t_world[0, 0]),
                z=float(self.t_world[2, 0]),
                yaw_deg=0.0,
                trajectory=list(self.trajectory),
                diagnostic_message="Initial reference keyframe registered.",
            )
            self.all_metrics.append(metrics)
            return metrics, annotated

        # Step 2: KNN Descriptor matching with Lowe's ratio test
        try:
            matches_knn = self.matcher.knnMatch(descs, self.prev_descs, k=2)
        except cv2.error as err:
            return self._record_failure(frame_idx, num_kps, "MATCHER_ERROR", str(err)), annotated

        good_matches = []
        for pair in matches_knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.config.match_ratio * n.distance:
                    good_matches.append(m)

        num_matches = len(good_matches)
        if num_matches < self.config.min_features_tracked:
            metrics = self._record_failure(
                frame_idx,
                num_kps,
                status="DEGRADED_MATCHES",
                msg=f"Only {num_matches} good matches passed ratio test. Camera pose locked to prevent drift."
            )
            self.prev_gray, self.prev_kps, self.prev_descs = gray, kps, descs
            return metrics, annotated

        # Step 3: Extract matched pixel coordinates
        pts_curr = np.float32([kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_prev = np.float32([self.prev_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Step 4: Essential Matrix estimation with RANSAC
        E, inlier_mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            self.K,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold,
        )

        if E is None or inlier_mask is None or inlier_mask.shape[0] < 5:
            metrics = self._record_failure(
                frame_idx,
                num_kps,
                status="FAILED_ESSENTIAL_MATRIX",
                msg="Essential matrix estimation failed. Degenerate geometry or non-rigid motion.",
                num_matches=num_matches,
                num_inliers=0
            )
            self.prev_gray, self.prev_kps, self.prev_descs = gray, kps, descs
            return metrics, annotated

        # Handle stacked solutions returned by RANSAC
        if E.shape[0] > 3:
            E = E[:3, :]

        # Step 5: Decompose Essential matrix to recover relative R, t
        num_inliers, R_rel, t_rel, pose_mask = cv2.recoverPose(
            E, pts_curr, pts_prev, self.K, mask=inlier_mask
        )

        inlier_ratio = num_inliers / max(1, num_matches)
        if num_inliers < self.config.min_features_tracked:
            metrics = self._record_failure(
                frame_idx,
                num_kps,
                status="FAILED_LOW_INLIERS",
                msg=f"Too few pose inliers ({num_inliers}/{num_matches}). Pose discarded to prevent spurious jump.",
                num_matches=num_matches,
                num_inliers=num_inliers
            )
            self.prev_gray, self.prev_kps, self.prev_descs = gray, kps, descs
            return metrics, annotated

        # Step 6: Compose World Pose (X: right, Z: forward)
        # Note: In monocular camera coordinates, forward motion has positive t_z in front of camera
        # Normalized translation direction scaled by relative step factor
        step_scale = self.config.relative_step_scale

        # Update translation in current world orientation
        delta_t = step_scale * (self.R_world @ t_rel)
        self.t_world = self.t_world + delta_t
        self.R_world = R_rel @ self.R_world

        cur_x = float(self.t_world[0, 0])
        cur_z = float(self.t_world[2, 0])
        self.trajectory.append((cur_x, cur_z, True))

        # Calculate yaw heading angle from rotation matrix
        sy = np.sqrt(self.R_world[0, 0] ** 2 + self.R_world[1, 0] ** 2)
        yaw_rad = np.arctan2(self.R_world[1, 0], self.R_world[0, 0]) if sy > 1e-6 else 0.0
        yaw_deg = float(np.degrees(yaw_rad))

        # Step 7: Draw feature tracks and optical flow HUD
        inliers_bool = (pose_mask.ravel() > 0) if pose_mask is not None else np.ones(num_matches, dtype=bool)
        for i, match in enumerate(good_matches):
            p1 = (int(pts_prev[i, 0, 0]), int(pts_prev[i, 0, 1]))
            p2 = (int(pts_curr[i, 0, 0]), int(pts_curr[i, 0, 1]))
            if inliers_bool[i]:
                # Inliers: green vector line from previous to current position
                cv2.line(annotated, p1, p2, (16, 185, 129), 1, cv2.LINE_AA)
                cv2.circle(annotated, p2, 2, (56, 189, 248), -1)
            else:
                # Outlier: red dot
                cv2.circle(annotated, p2, 2, (239, 68, 68), -1)

        # Store previous state for next iteration
        self.prev_gray = gray
        self.prev_kps = kps
        self.prev_descs = descs

        metrics = VOMetrics(
            frame_idx=frame_idx,
            status="TRACKING_OK",
            tracking_ok=True,
            num_keypoints=num_kps,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=round(inlier_ratio, 3),
            x=round(cur_x, 3),
            z=round(cur_z, 3),
            yaw_deg=round(yaw_deg, 1),
            trajectory=list(self.trajectory),
            diagnostic_message=f"Tracking nominal: {num_inliers} inliers ({inlier_ratio:.0%})",
        )
        self.all_metrics.append(metrics)
        return metrics, annotated

    def _record_failure(
        self,
        frame_idx: int,
        num_kps: int,
        status: str,
        msg: str,
        num_matches: int = 0,
        num_inliers: int = 0
    ) -> VOMetrics:
        """Record tracking failure without fabricating position motion."""
        cur_x = float(self.t_world[0, 0])
        cur_z = float(self.t_world[2, 0])
        # Record stationary point marked as tracking_ok=False
        self.trajectory.append((cur_x, cur_z, False))

        inlier_ratio = (num_inliers / max(1, num_matches)) if num_matches > 0 else 0.0

        metrics = VOMetrics(
            frame_idx=frame_idx,
            status=status,
            tracking_ok=False,
            num_keypoints=num_kps,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=round(inlier_ratio, 3),
            x=round(cur_x, 3),
            z=round(cur_z, 3),
            yaw_deg=0.0,
            trajectory=list(self.trajectory),
            diagnostic_message=msg,
        )
        self.all_metrics.append(metrics)
        return metrics
