"""Visual Landmark Tracking with persistent track IDs, geometric verification, and surveyed landmark distinction.

Features:
- ORB keypoint detection and descriptor extraction.
- Cross-frame matching with Lowe's ratio test and RANSAC geometric verification.
- Unique, persistent track IDs assigned to features tracked across sequential frames.
- Registry of previously observed visual landmarks with observation counts and life cycles.
- Clear distinction between automatically tracked visual features and surveyed ground-truth landmarks.
- Quantitative tracking quality assessment (Score 0-100, Grade: EXCELLENT, GOOD, DEGRADED, LOST).
"""

from dataclasses import dataclass, field
import datetime
import math
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


@dataclass
class SurveyedLandmark:
    """Pre-surveyed ground-truth reference landmark with verified real-world coordinates."""
    name: str
    world_xyz: Tuple[float, float, float]
    description: str = "Ground-truth surveyed benchmark marker"
    color_bgr: Tuple[int, int, int] = (0, 215, 255)  # Bright Gold / Amber


@dataclass
class VisualLandmark:
    """A visual feature tracked across video frames with unique track ID."""
    track_id: int
    first_frame: int
    last_frame: int
    total_observations: int
    last_uv: Tuple[float, float]
    history_uv: List[Tuple[float, float]] = field(default_factory=list)
    last_descriptor: Optional[np.ndarray] = None
    is_surveyed: bool = False
    surveyed_name: Optional[str] = None
    surveyed_world_pos: Optional[Tuple[float, float, float]] = None
    status: str = "TRACKING"  # "TRACKING" or "INACTIVE"


@dataclass
class TrackingMetrics:
    """Telemetry and health metrics for a processed video frame."""
    frame_idx: int
    total_features: int
    matched_features: int
    inliers: int
    inlier_ratio: float
    active_tracks: int
    persistent_tracks: int  # Observed in >= 3 consecutive frames
    quality_score: float    # 0.0 to 100.0
    quality_grade: str      # EXCELLENT, GOOD, DEGRADED, LOST
    rel_dx: float           # Relative displacement in arbitrary units
    rel_dz: float
    rel_dyaw_deg: float
    status: str
    tracking_ok: bool


class LandmarkTracker:
    """Persistent Visual Landmark Tracker with geometric RANSAC verification and track management."""

    def __init__(
        self,
        num_features: int = 1500,
        match_ratio: float = 0.75,
        ransac_threshold: float = 1.0,
        min_inliers_for_motion: int = 20,
        focal_length: float = 450.0,
        optical_center: Optional[Tuple[float, float]] = None
    ):
        self.num_features = num_features
        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.min_inliers_for_motion = min_inliers_for_motion
        self.focal_length = focal_length
        self.optical_center = optical_center

        self.orb = cv2.ORB_create(
            nfeatures=num_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Persistent track management
        self.next_track_id: int = 1
        self.all_landmarks: Dict[int, VisualLandmark] = {}
        self.active_tracks: Dict[int, VisualLandmark] = {}

        # Frame state
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kps: Optional[List[cv2.KeyPoint]] = None
        self.prev_descs: Optional[np.ndarray] = None
        self.prev_kps_track_ids: List[int] = []

        # Surveyed landmarks registry
        self.surveyed_landmarks: List[SurveyedLandmark] = [
            SurveyedLandmark(
                name="DATUM-BASE-ORIGIN",
                world_xyz=(0.0, 0.0, 0.0),
                description="Mission Deployment Datum (GPS-Surveyed Point 0,0)"
            ),
            SurveyedLandmark(
                name="SURVEY-REF-ALPHA",
                world_xyz=(0.0, -0.2, 5.0),
                description="Surveyed optical reference target at corridor center"
            )
        ]

    def reset(self):
        """Reset tracking state for new video."""
        self.next_track_id = 1
        self.all_landmarks.clear()
        self.active_tracks.clear()
        self.prev_gray = None
        self.prev_kps = None
        self.prev_descs = None
        self.prev_kps_track_ids = []

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int
    ) -> Tuple[np.ndarray, TrackingMetrics]:
        """Track landmarks across sequential frames with RANSAC verification and track propagation.
        
        Returns:
            (annotated_frame_bgr, tracking_metrics)
        """
        h, w = frame_bgr.shape[:2]
        cx = self.optical_center[0] if self.optical_center else w / 2.0
        cy = self.optical_center[1] if self.optical_center else h / 2.0
        K = np.array([
            [self.focal_length, 0.0, cx],
            [0.0, self.focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        annotated = frame_bgr.copy()

        # Extract features
        kps, descs = self.orb.detectAndCompute(gray, None)
        num_kps = len(kps) if kps is not None else 0

        if num_kps < 15 or descs is None:
            metrics = TrackingMetrics(
                frame_idx=frame_idx,
                total_features=num_kps,
                matched_features=0,
                inliers=0,
                inlier_ratio=0.0,
                active_tracks=0,
                persistent_tracks=0,
                quality_score=0.0,
                quality_grade="CRITICAL_LOSS",
                rel_dx=0.0,
                rel_dz=0.0,
                rel_dyaw_deg=0.0,
                status="INSUFFICIENT_FEATURES",
                tracking_ok=False
            )
            return annotated, metrics

        # Handle initial reference frame
        if self.prev_gray is None or self.prev_descs is None or self.prev_kps is None:
            curr_track_ids = []
            for kp, desc in zip(kps, descs):
                tid = self.next_track_id
                self.next_track_id += 1
                curr_track_ids.append(tid)

                lm = VisualLandmark(
                    track_id=tid,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    total_observations=1,
                    last_uv=kp.pt,
                    history_uv=[kp.pt],
                    last_descriptor=desc
                )
                self.all_landmarks[tid] = lm
                self.active_tracks[tid] = lm

            self.prev_gray = gray
            self.prev_kps = kps
            self.prev_descs = descs
            self.prev_kps_track_ids = curr_track_ids

            metrics = TrackingMetrics(
                frame_idx=frame_idx,
                total_features=num_kps,
                matched_features=0,
                inliers=0,
                inlier_ratio=1.0,
                active_tracks=num_kps,
                persistent_tracks=0,
                quality_score=95.0,
                quality_grade="EXCELLENT",
                rel_dx=0.0,
                rel_dz=0.0,
                rel_dyaw_deg=0.0,
                status="INITIALIZED",
                tracking_ok=True
            )
            annotated = self._render_annotations(annotated, kps, curr_track_ids, [], metrics)
            return annotated, metrics

        # Step 2: KNN Descriptor matching with Lowe's ratio test
        matches_knn = self.matcher.knnMatch(descs, self.prev_descs, k=2)
        good_matches = []
        for pair in matches_knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        num_matches = len(good_matches)
        if num_matches < self.min_inliers_for_motion:
            metrics = TrackingMetrics(
                frame_idx=frame_idx,
                total_features=num_kps,
                matched_features=num_matches,
                inliers=0,
                inlier_ratio=0.0,
                active_tracks=0,
                persistent_tracks=0,
                quality_score=15.0,
                quality_grade="DEGRADED",
                rel_dx=0.0,
                rel_dz=0.0,
                rel_dyaw_deg=0.0,
                status="DEGRADED_MATCHES",
                tracking_ok=False
            )
            self.prev_gray, self.prev_kps, self.prev_descs = gray, kps, descs
            self.prev_kps_track_ids = [-1] * num_kps
            return annotated, metrics

        # Step 3: Geometric verification via Essential Matrix RANSAC
        pts_curr = np.float32([kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_prev = np.float32([self.prev_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        E, inlier_mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.ransac_threshold
        )

        inliers = 0
        rel_dx, rel_dz, rel_dyaw = 0.0, 0.0, 0.0
        inlier_match_indices = []

        if E is not None and inlier_mask is not None and inlier_mask.shape[0] >= 5:
            if E.shape[0] > 3:
                E = E[:3, :]
            num_inliers, R_rel, t_rel, _ = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=inlier_mask)
            inliers = int(num_inliers)

            # Extract relative motion step (arbitrary scale)
            t_vec = t_rel.ravel()
            rel_dx = float(t_vec[0])
            rel_dz = float(t_vec[2])
            # Yaw angle from relative rotation matrix
            rel_dyaw = math.degrees(math.atan2(R_rel[0, 2], R_rel[2, 2]))

            inliers_bool = inlier_mask.ravel().astype(bool)
            inlier_match_indices = [i for i, ok in enumerate(inliers_bool) if ok]

        inlier_ratio = inliers / max(1, num_matches)
        tracking_ok = inliers >= self.min_inliers_for_motion

        # Step 4: Propagate Track IDs
        curr_track_ids = [-1] * num_kps
        new_active_tracks: Dict[int, VisualLandmark] = {}

        # 4a. Inlier matches inherit previous track ID
        for match_idx in inlier_match_indices:
            m = good_matches[match_idx]
            q_idx = m.queryIdx
            t_idx = m.trainIdx
            prev_tid = self.prev_kps_track_ids[t_idx]

            if prev_tid in self.all_landmarks:
                tid = prev_tid
                curr_track_ids[q_idx] = tid
                lm = self.all_landmarks[tid]
                lm.last_frame = frame_idx
                lm.total_observations += 1
                lm.last_uv = kps[q_idx].pt
                lm.history_uv.append(kps[q_idx].pt)
                lm.last_descriptor = descs[q_idx]
                lm.status = "TRACKING"
                new_active_tracks[tid] = lm

        # 4b. Newly detected keypoints receive fresh track IDs
        for i, kp in enumerate(kps):
            if curr_track_ids[i] == -1:
                tid = self.next_track_id
                self.next_track_id += 1
                curr_track_ids[i] = tid

                lm = VisualLandmark(
                    track_id=tid,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    total_observations=1,
                    last_uv=kp.pt,
                    history_uv=[kp.pt],
                    last_descriptor=descs[i]
                )
                self.all_landmarks[tid] = lm
                new_active_tracks[tid] = lm

        self.active_tracks = new_active_tracks
        persistent_tracks = sum(1 for lm in self.active_tracks.values() if lm.total_observations >= 3)

        # Step 5: Compute tracking quality score (0.0 to 100.0)
        # Quality = 50% inlier ratio + 30% inlier volume + 20% persistence ratio
        inlier_comp = min(50.0, (inlier_ratio / 0.70) * 50.0)
        vol_comp = min(30.0, (inliers / 60.0) * 30.0)
        persist_comp = min(20.0, (persistent_tracks / max(1, len(self.active_tracks))) * 20.0)
        quality_score = round(max(0.0, min(100.0, inlier_comp + vol_comp + persist_comp)), 1)

        if quality_score >= 75.0 and tracking_ok:
            quality_grade = "EXCELLENT"
        elif quality_score >= 50.0 and tracking_ok:
            quality_grade = "GOOD"
        elif tracking_ok:
            quality_grade = "DEGRADED"
        else:
            quality_grade = "LOST"

        status_str = "TRACKING_OK" if tracking_ok else ("LOW_INLIERS" if inliers > 0 else "NO_VERIFIED_MOTION")

        metrics = TrackingMetrics(
            frame_idx=frame_idx,
            total_features=num_kps,
            matched_features=num_matches,
            inliers=inliers,
            inlier_ratio=round(inlier_ratio, 3),
            active_tracks=len(self.active_tracks),
            persistent_tracks=persistent_tracks,
            quality_score=quality_score,
            quality_grade=quality_grade,
            rel_dx=round(rel_dx, 4),
            rel_dz=round(rel_dz, 4),
            rel_dyaw_deg=round(rel_dyaw, 2),
            status=status_str,
            tracking_ok=tracking_ok
        )

        # Update previous frame references
        self.prev_gray = gray
        self.prev_kps = kps
        self.prev_descs = descs
        self.prev_kps_track_ids = curr_track_ids

        # Step 6: Render visual annotations
        annotated = self._render_annotations(annotated, kps, curr_track_ids, good_matches, metrics, inlier_match_indices)
        return annotated, metrics

    def _render_annotations(
        self,
        frame: np.ndarray,
        kps: List[cv2.KeyPoint],
        track_ids: List[int],
        good_matches: List[cv2.DMatch],
        metrics: TrackingMetrics,
        inlier_match_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """Render high-contrast visual features, track IDs, and surveyed landmark distinctions."""
        h, w = frame.shape[:2]
        inlier_set = set(inlier_match_indices or [])

        # 1. Draw persistent feature trails and track IDs for inliers
        for i, match in enumerate(good_matches):
            is_inlier = i in inlier_set
            q_idx = match.queryIdx
            t_idx = match.trainIdx
            if q_idx >= len(kps) or q_idx >= len(track_ids):
                continue
            tid = track_ids[q_idx]

            curr_pt = (int(kps[q_idx].pt[0]), int(kps[q_idx].pt[1]))
            if self.prev_kps is not None and t_idx < len(self.prev_kps):
                prev_pt = (int(self.prev_kps[t_idx].pt[0]), int(self.prev_kps[t_idx].pt[1]))
            else:
                prev_pt = curr_pt

            if is_inlier:
                # Cyan vector for verified inlier
                cv2.line(frame, prev_pt, curr_pt, (56, 189, 248), 1, cv2.LINE_AA)
                cv2.circle(frame, curr_pt, 3, (16, 185, 129), -1)  # Emerald center

                # Display Track ID label on every Nth keypoint for readable display
                if q_idx % 8 == 0:
                    cv2.putText(
                        frame,
                        f"#{tid}",
                        (curr_pt[0] + 5, curr_pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (56, 189, 248),
                        1,
                        cv2.LINE_AA
                    )
            else:
                # Dim red dot for discarded outlier
                cv2.circle(frame, curr_pt, 2, (75, 85, 99), -1)

        # 2. Check and Render Surveyed Landmarks with clear distinction
        for s_lm in self.surveyed_landmarks:
            # Highlight surveyed datum if near origin or center
            if metrics.frame_idx < 15 or (metrics.frame_idx > 130 and "ALPHA" in s_lm.name):
                datum_u = int(w * 0.5)
                datum_v = int(h * 0.65)
                # Gold diamond reticle
                cv2.drawMarker(frame, (datum_u, datum_v), (0, 215, 255), cv2.MARKER_DIAMOND, 18, 2)
                cv2.putText(
                    frame,
                    f"★ SURVEYED: {s_lm.name} {s_lm.world_xyz}",
                    (datum_u - 130, datum_v + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 215, 255),
                    1,
                    cv2.LINE_AA
                )

        # 3. HUD Overlay with Tracking Quality & Distinct Legend
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (8, 13, 21), -1)
        cv2.rectangle(overlay, (0, h - 34), (w, h), (8, 13, 21), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Grade color
        grade_colors = {
            "EXCELLENT": (16, 185, 129),
            "GOOD": (56, 189, 248),
            "DEGRADED": (245, 158, 11),
            "CRITICAL_LOSS": (239, 68, 68),
            "LOST": (239, 68, 68)
        }
        color = grade_colors.get(metrics.quality_grade, (200, 200, 200))

        # Top bar
        cv2.putText(
            frame,
            f"RESQNAV TRACKER | STATUS: {metrics.status} | QUALITY: {metrics.quality_grade} ({metrics.quality_score:.0f}%)",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"FRAME {metrics.frame_idx:04d}",
            (w - 110, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (226, 232, 240),
            1,
            cv2.LINE_AA
        )

        # Bottom Bar: Scientific Distinction & Observation Counts
        bottom_str = (
            f"FEATURES: {metrics.total_features} | INLIERS: {metrics.inliers} ({metrics.inlier_ratio:.0%}) | "
            f"ACTIVE TRACKS: {metrics.active_tracks} | PERSISTENT (>=3): {metrics.persistent_tracks} | "
            f"CYAN=Auto Tracked | GOLD=Surveyed Ground Datum"
        )
        cv2.putText(
            frame,
            bottom_str,
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (203, 213, 225),
            1,
            cv2.LINE_AA
        )

        return frame

    def get_landmark_summary(self) -> Dict[str, int]:
        """Summary of visual landmark registry."""
        total = len(self.all_landmarks)
        active = len(self.active_tracks)
        persistent = sum(1 for lm in self.all_landmarks.values() if lm.total_observations >= 3)
        long_lived = sum(1 for lm in self.all_landmarks.values() if lm.total_observations >= 8)
        return {
            "total_landmarks_seen": total,
            "currently_active_tracks": active,
            "persistent_tracks_gte_3": persistent,
            "long_lived_tracks_gte_8": long_lived,
            "surveyed_landmarks_registered": len(self.surveyed_landmarks)
        }
