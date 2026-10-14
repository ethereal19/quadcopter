"""Visual SLAM and Loop Closure with Keyframe Management, Geometric Verification, and SE(2) Pose-Graph Optimization.

Features:
- Representative keyframe selection with visual descriptors and thumbnail caching.
- Appearance-based candidate retrieval with non-local temporal exclusion window.
- Strict geometric verification via RANSAC Essential Matrix and relative pose recovery (rejecting purely visual matches).
- Documented 2D Pose-Graph Optimization using SciPy non-linear least squares with robust Huber loss.
- Comparative trajectory visualization: Trajectory Before vs. After Optimization with loop closure edges.
- Explicit prototype labeling and scientific scale disclaimer.
"""

from dataclasses import dataclass, field
import datetime
import math
from typing import Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Keyframe:
    """Representative keyframe with extracted ORB descriptors and camera pose."""
    id: int
    frame_idx: int
    pose_x: float  # Relative units
    pose_z: float  # Relative units
    yaw_deg: float
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    thumbnail: np.ndarray  # (60, 80, 3) BGR thumbnail for dashboard display


@dataclass
class LoopClosureEdge:
    """Verified loop closure constraint edge between two distant keyframes."""
    query_kf_id: int
    match_kf_id: int
    rel_dx: float
    rel_dz: float
    rel_dyaw_deg: float
    inliers: int
    inlier_ratio: float
    weight: float = 12.0


@dataclass
class LoopCandidate:
    """Audit log of evaluated loop closure candidates and verification outcomes."""
    query_kf_id: int
    match_kf_id: int
    query_frame_idx: int
    match_frame_idx: int
    num_matches: int
    inliers: int
    inlier_ratio: float
    accepted: bool
    status_message: str
    timestamp_iso: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


class VisualSLAM:
    """Visual SLAM engine managing keyframes, loop closure detection, geometric verification, and pose-graph optimization."""

    def __init__(
        self,
        min_keyframe_dist: float = 1.0,
        min_keyframe_yaw_deg: float = 15.0,
        temporal_window: int = 5,
        min_loop_matches: int = 24,
        min_loop_inliers: int = 16,
        min_inlier_ratio: float = 0.35,
        focal_length: float = 450.0
    ):
        self.min_keyframe_dist = min_keyframe_dist
        self.min_keyframe_yaw_deg = min_keyframe_yaw_deg
        self.temporal_window = temporal_window
        self.min_loop_matches = min_loop_matches
        self.min_loop_inliers = min_loop_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.focal_length = focal_length

        self.keyframes: List[Keyframe] = []
        self.loop_edges: List[LoopClosureEdge] = []
        self.candidates_history: List[LoopCandidate] = []
        self.optimized_poses: Optional[List[Tuple[float, float, float]]] = None

        self.orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def reset(self):
        """Reset SLAM session."""
        self.keyframes.clear()
        self.loop_edges.clear()
        self.candidates_history.clear()
        self.optimized_poses = None

    def consider_keyframe(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        cur_pose: Tuple[float, float, float]
    ) -> Tuple[bool, Optional[Keyframe], Optional[LoopCandidate]]:
        """Evaluate whether current frame should become a keyframe and check for loop closure.
        
        Returns:
            (is_keyframe_added, keyframe_obj, loop_candidate_obj)
        """
        cur_x, cur_z, cur_yaw = cur_pose

        # Check eligibility against last keyframe
        if len(self.keyframes) > 0:
            last_kf = self.keyframes[-1]
            dist = math.hypot(cur_x - last_kf.pose_x, cur_z - last_kf.pose_z)
            yaw_diff = abs((cur_yaw - last_kf.yaw_deg + 180.0) % 360.0 - 180.0)

            if dist < self.min_keyframe_dist and yaw_diff < self.min_keyframe_yaw_deg:
                return False, None, None

        # Extract features for keyframe
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = self.orb.detectAndCompute(gray, None)

        if kps is None or descs is None or len(kps) < 30:
            return False, None, None

        # Create small thumbnail
        thumb = cv2.resize(frame_bgr, (120, 90), interpolation=cv2.INTER_AREA)

        kf = Keyframe(
            id=len(self.keyframes),
            frame_idx=frame_idx,
            pose_x=cur_x,
            pose_z=cur_z,
            yaw_deg=cur_yaw,
            keypoints=kps,
            descriptors=descs,
            thumbnail=thumb
        )
        self.keyframes.append(kf)

        # Check for loop closure against historical keyframes
        loop_cand = self._detect_and_verify_loop_closure(kf, frame_bgr.shape)
        if loop_cand is not None and loop_cand.accepted:
            # Auto-run pose graph optimization when a loop is verified
            self.optimize_pose_graph()

        return True, kf, loop_cand

    def _detect_and_verify_loop_closure(
        self,
        query_kf: Keyframe,
        frame_shape: Tuple[int, int, ...]
    ) -> Optional[LoopCandidate]:
        """Search for loop closure with temporal exclusion and strict geometric RANSAC verification."""
        if len(self.keyframes) <= self.temporal_window + 1:
            return None

        h, w = frame_shape[:2]
        K = np.array([
            [self.focal_length, 0.0, w / 2.0],
            [0.0, self.focal_length, h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Candidate search range: from 0 up to (query_kf.id - temporal_window)
        search_limit = query_kf.id - self.temporal_window
        best_candidate: Optional[LoopCandidate] = None
        best_edge: Optional[LoopClosureEdge] = None
        max_inliers = 0

        for candidate_kf in self.keyframes[:search_limit]:
            # Step 1: Visual feature matching with Lowe's ratio test
            try:
                matches_knn = self.matcher.knnMatch(query_kf.descriptors, candidate_kf.descriptors, k=2)
            except cv2.error:
                continue

            good_matches = []
            for pair in matches_knn:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            num_matches = len(good_matches)
            if num_matches < self.min_loop_matches:
                continue

            # Step 2: GEOMETRIC VERIFICATION via Essential Matrix RANSAC
            # (Do NOT accept matches based only on visual appearance similarity!)
            pts_query = np.float32([query_kf.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts_match = np.float32([candidate_kf.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            E, inlier_mask = cv2.findEssentialMat(
                pts_query,
                pts_match,
                K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.2
            )

            if E is None or inlier_mask is None or inlier_mask.shape[0] < 5:
                cand = LoopCandidate(
                    query_kf_id=query_kf.id,
                    match_kf_id=candidate_kf.id,
                    query_frame_idx=query_kf.frame_idx,
                    match_frame_idx=candidate_kf.frame_idx,
                    num_matches=num_matches,
                    inliers=0,
                    inlier_ratio=0.0,
                    accepted=False,
                    status_message="REJECTED: Essential Matrix RANSAC degenerate geometry"
                )
                self.candidates_history.append(cand)
                continue

            if E.shape[0] > 3:
                E = E[:3, :]

            num_inliers, R_rel, t_rel, _ = cv2.recoverPose(E, pts_query, pts_match, K, mask=inlier_mask)
            inlier_count = int(num_inliers)
            inlier_ratio = inlier_count / max(1, num_matches)

            # Step 3: Strict Acceptance Criteria
            if inlier_count >= self.min_loop_inliers and inlier_ratio >= self.min_inlier_ratio:
                t_vec = t_rel.ravel()
                rel_dx = float(t_vec[0])
                rel_dz = float(t_vec[2])
                rel_dyaw = math.degrees(math.atan2(R_rel[0, 2], R_rel[2, 2]))

                cand = LoopCandidate(
                    query_kf_id=query_kf.id,
                    match_kf_id=candidate_kf.id,
                    query_frame_idx=query_kf.frame_idx,
                    match_frame_idx=candidate_kf.frame_idx,
                    num_matches=num_matches,
                    inliers=inlier_count,
                    inlier_ratio=round(inlier_ratio, 3),
                    accepted=True,
                    status_message=f"ACCEPTED: Geometric RANSAC verified with {inlier_count} inliers ({inlier_ratio:.0%})"
                )

                edge = LoopClosureEdge(
                    query_kf_id=query_kf.id,
                    match_kf_id=candidate_kf.id,
                    rel_dx=rel_dx,
                    rel_dz=rel_dz,
                    rel_dyaw_deg=rel_dyaw,
                    inliers=inlier_count,
                    inlier_ratio=inlier_ratio
                )

                if inlier_count > max_inliers:
                    max_inliers = inlier_count
                    best_candidate = cand
                    best_edge = edge
            else:
                cand = LoopCandidate(
                    query_kf_id=query_kf.id,
                    match_kf_id=candidate_kf.id,
                    query_frame_idx=query_kf.frame_idx,
                    match_frame_idx=candidate_kf.frame_idx,
                    num_matches=num_matches,
                    inliers=inlier_count,
                    inlier_ratio=round(inlier_ratio, 3),
                    accepted=False,
                    status_message=f"REJECTED: Low inliers ({inlier_count}/{num_matches}) below threshold ({self.min_loop_inliers})"
                )
                self.candidates_history.append(cand)

        if best_candidate and best_edge:
            self.candidates_history.append(best_candidate)
            self.loop_edges.append(best_edge)
            return best_candidate

        return None

    def optimize_pose_graph(self) -> List[Tuple[float, float, float]]:
        """SE(2) Pose-Graph Optimization using SciPy Non-linear Least-Squares with Huber Loss.
        
        Mathematical Formulation:
        Minimizes total residual error across odometry constraints and loop closure edges:
            E(P) = sum_{k} w_odom || P_{k+1} - P_k - Delta P_{k,k+1} ||^2 
                 + sum_{(i,j)} w_loop || P_j - P_i - Delta P_{ij} ||^2
        """
        N = len(self.keyframes)
        if N < 2:
            return [(kf.pose_x, kf.pose_z, kf.yaw_deg) for kf in self.keyframes]

        # Initial parameter vector: [x_1, z_1, yaw_1, ..., x_{N-1}, z_{N-1}, yaw_{N-1}]
        # Anchor keyframe 0 fixed at (x_0, z_0, yaw_0)
        p0 = (self.keyframes[0].pose_x, self.keyframes[0].pose_z, self.keyframes[0].yaw_deg)
        x0_list = []
        for kf in self.keyframes[1:]:
            x0_list.extend([kf.pose_x, kf.pose_z, kf.yaw_deg])
        initial_params = np.array(x0_list, dtype=np.float64)

        # Precompute sequential odometry step measurements
        odom_deltas = []
        for k in range(N - 1):
            kf_a = self.keyframes[k]
            kf_b = self.keyframes[k + 1]
            dx = kf_b.pose_x - kf_a.pose_x
            dz = kf_b.pose_z - kf_a.pose_z
            dyaw = (kf_b.yaw_deg - kf_a.yaw_deg + 180.0) % 360.0 - 180.0
            odom_deltas.append((dx, dz, dyaw))

        def residuals(params):
            res = []
            # Reconstruct pose dictionary
            poses = {0: p0}
            for idx in range(1, N):
                base_idx = (idx - 1) * 3
                poses[idx] = (params[base_idx], params[base_idx + 1], params[base_idx + 2])

            # 1. Odometry constraints
            w_odom = 1.0
            for k in range(N - 1):
                p_a = poses[k]
                p_b = poses[k + 1]
                dx_obs, dz_obs, dyaw_obs = odom_deltas[k]

                res.append(w_odom * (p_b[0] - p_a[0] - dx_obs))
                res.append(w_odom * (p_b[1] - p_a[1] - dz_obs))
                dyaw_err = (p_b[2] - p_a[2] - dyaw_obs + 180.0) % 360.0 - 180.0
                res.append(w_odom * 0.05 * dyaw_err)

            # 2. Loop closure constraints
            for edge in self.loop_edges:
                p_q = poses.get(edge.query_kf_id)
                p_m = poses.get(edge.match_kf_id)
                if p_q is not None and p_m is not None:
                    # In a loop closure, query_kf should snap back toward match_kf
                    w_loop = edge.weight
                    res.append(w_loop * (p_q[0] - p_m[0]))
                    res.append(w_loop * (p_q[1] - p_m[1]))
                    dyaw_loop = (p_q[2] - p_m[2] + 180.0) % 360.0 - 180.0
                    res.append(w_loop * 0.05 * dyaw_loop)

            return np.array(res, dtype=np.float64)

        if len(self.loop_edges) > 0:
            sol = least_squares(residuals, initial_params, method="trf", loss="huber", max_nfev=150)
            opt_params = sol.x
        else:
            opt_params = initial_params

        # Format optimized output poses
        opt_poses = [p0]
        for idx in range(1, N):
            base_idx = (idx - 1) * 3
            opt_poses.append((
                float(round(opt_params[base_idx], 3)),
                float(round(opt_params[base_idx + 1], 3)),
                float(round(opt_params[base_idx + 2], 1))
            ))

        self.optimized_poses = opt_poses
        return opt_poses

    def plot_trajectory_comparison(
        self,
        title: str = "Visual SLAM: Trajectory Before vs After Pose-Graph Optimization"
    ) -> plt.Figure:
        """Render side-by-side comparative trajectory map before vs after loop closure optimization."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")

        if len(self.keyframes) == 0:
            ax.text(0.5, 0.5, "No Keyframes in SLAM Graph", color="#64748b", ha="center", va="center")
            return fig

        raw_xs = [kf.pose_x for kf in self.keyframes]
        raw_zs = [kf.pose_z for kf in self.keyframes]

        # 1. Trajectory Before Optimization (Drifted Odometry)
        ax.plot(
            raw_xs, raw_zs,
            color="#ea580c",
            linestyle="--",
            linewidth=2.2,
            alpha=0.90,
            label="Trajectory Before Optimization (Drifted)",
            zorder=3
        )
        ax.scatter(raw_xs, raw_zs, c="#ea580c", s=25, alpha=0.8, zorder=4)

        # 2. Trajectory After Optimization (if computed)
        if self.optimized_poses and len(self.optimized_poses) == len(self.keyframes):
            opt_xs = [p[0] for p in self.optimized_poses]
            opt_zs = [p[1] for p in self.optimized_poses]

            ax.plot(
                opt_xs, opt_zs,
                color="#0284c7",
                linestyle="-",
                linewidth=2.6,
                label="Trajectory After Pose-Graph Optimization",
                zorder=5
            )
            ax.scatter(opt_xs, opt_zs, c="#0284c7", s=35, edgecolors="#ffffff", linewidth=0.8, zorder=6)

            # Draw loop closure constraint edges connecting matched keyframes
            for edge in self.loop_edges:
                if edge.query_kf_id < len(opt_xs) and edge.match_kf_id < len(opt_xs):
                    qx, qz = opt_xs[edge.query_kf_id], opt_zs[edge.query_kf_id]
                    mx, mz = opt_xs[edge.match_kf_id], opt_zs[edge.match_kf_id]
                    ax.plot(
                        [qx, mx], [qz, mz],
                        color="#16a34a",
                        linestyle=":",
                        linewidth=2.5,
                        label=f"Loop Closure Edge (KF#{edge.match_kf_id} <-> #{edge.query_kf_id})",
                        zorder=7
                    )
        else:
            # Draw loop closure edges across raw poses
            for edge in self.loop_edges:
                if edge.query_kf_id < len(raw_xs) and edge.match_kf_id < len(raw_xs):
                    qx, qz = raw_xs[edge.query_kf_id], raw_zs[edge.query_kf_id]
                    mx, mz = raw_xs[edge.match_kf_id], raw_zs[edge.match_kf_id]
                    ax.plot(
                        [qx, mx], [qz, mz],
                        color="#16a34a",
                        linestyle=":",
                        linewidth=2.5,
                        label="Verified Loop Closure Edge",
                        zorder=7
                    )

        # Mark Origin
        ax.scatter(0.0, 0.0, c="#16a34a", marker="*", s=160, edgecolors="#0f172a", linewidth=1.0, label="Start / Origin (0, 0)", zorder=8)

        ax.grid(color="#e2e8f0", linestyle="--", linewidth=0.8, alpha=0.9)
        ax.tick_params(colors="#475569", labelsize=8)
        ax.set_xlabel("Lateral X Axis (Relative / Arbitrary Units)", color="#334155", fontsize=9, fontweight="bold")
        ax.set_ylabel("Forward Z Axis (Relative / Arbitrary Units)", color="#334155", fontsize=9, fontweight="bold")
        ax.set_title(title, color="#0f172a", fontsize=11, fontweight="bold", pad=10)

        # Prototype status notice
        fig.text(
            0.5, 0.01,
            "NOTICE: SE(2) Pose-Graph Prototype. Corrections optimized via SciPy Huber least-squares. Monocular scale is relative.",
            color="#b45309",
            fontsize=7.2,
            ha="center",
            fontfamily="monospace"
        )

        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            labelcolor="#1e293b",
            fontsize=7.5
        )

        ax.axis("equal")
        fig.tight_layout()
        return fig
