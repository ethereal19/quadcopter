"""Breadcrumb Navigation module with trajectory construction, SQLite/CSV waypoint persistence, and 2D mapping.

Features:
- Frame-to-frame camera motion integration into cumulative trajectory.
- Automated distance-based breadcrumbs and user-triggered manual waypoints.
- CSV and SQLite persistence (saving and loading waypoints).
- Pause, resume, save, and replay state controls.
- Current estimated pose vs last-known valid pose telemetry.
- 2D Matplotlib breadcrumb map with numbered waypoints and mandatory scientific scale ambiguity labeling.
"""

import csv
from dataclasses import dataclass, asdict
import datetime
import math
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Waypoint:
    """A marked navigation waypoint along the rover's trajectory."""
    id: int
    timestamp_iso: str
    frame_idx: int
    x_rel: float
    z_rel: float
    yaw_deg: float
    label: str
    is_manual: bool = False


class BreadcrumbNavigator:
    """Manages rover trajectory, waypoint persistence, playback controls, and 2D breadcrumb mapping."""

    def __init__(
        self,
        auto_drop_interval_dist: float = 1.0,
        relative_scale_label: str = "Relative / Arbitrary Units (Unscaled)"
    ):
        self.auto_drop_interval_dist = auto_drop_interval_dist
        self.scale_label = relative_scale_label

        # Trajectory history: (x, z, yaw_deg, frame_idx, tracking_ok)
        self.trajectory: List[Tuple[float, float, float, int, bool]] = [(0.0, 0.0, 0.0, 0, True)]
        self.waypoints: List[Waypoint] = []
        self.next_waypoint_id: int = 1

        # Current and last known pose
        self.current_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_known_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_frame_timestamp: str = datetime.datetime.now().isoformat()
        self.last_frame_idx: int = 0

        # Controls
        self.is_paused: bool = False
        self.last_dropped_dist_pos: Tuple[float, float] = (0.0, 0.0)

        # Drop initial base origin waypoint
        self.add_manual_waypoint(label="BASE-DEPLOYMENT-DATUM", frame_idx=0)

    def reset(self):
        """Reset trajectory and waypoints for replay or new mission."""
        self.trajectory = [(0.0, 0.0, 0.0, 0, True)]
        self.waypoints.clear()
        self.next_waypoint_id = 1
        self.current_pose = (0.0, 0.0, 0.0)
        self.last_known_pose = (0.0, 0.0, 0.0)
        self.last_dropped_dist_pos = (0.0, 0.0)
        self.is_paused = False
        self.add_manual_waypoint(label="BASE-DEPLOYMENT-DATUM", frame_idx=0)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def toggle_pause(self) -> bool:
        self.is_paused = not self.is_paused
        return self.is_paused

    def update_pose_step(
        self,
        dx_rel: float,
        dz_rel: float,
        dyaw_deg: float,
        frame_idx: int,
        tracking_ok: bool
    ) -> Tuple[float, float, float]:
        """Integrate relative camera displacement into cumulative trajectory if not paused."""
        if self.is_paused:
            return self.current_pose

        self.last_frame_idx = frame_idx
        self.last_frame_timestamp = datetime.datetime.now().isoformat()

        cur_x, cur_z, cur_yaw = self.current_pose

        if tracking_ok:
            # Integrate relative step in world frame rotated by current yaw
            yaw_rad = math.radians(cur_yaw)
            world_dx = dx_rel * math.cos(yaw_rad) + dz_rel * math.sin(yaw_rad)
            world_dz = -dx_rel * math.sin(yaw_rad) + dz_rel * math.cos(yaw_rad)

            new_x = cur_x + world_dx
            new_z = cur_z + world_dz
            new_yaw = (cur_yaw + dyaw_deg + 180.0) % 360.0 - 180.0

            self.current_pose = (round(new_x, 3), round(new_z, 3), round(new_yaw, 1))
            self.last_known_pose = self.current_pose
            self.trajectory.append((self.current_pose[0], self.current_pose[1], self.current_pose[2], frame_idx, True))

            # Check distance traveled since last breadcrumb
            dist_since_last = math.hypot(
                self.current_pose[0] - self.last_dropped_dist_pos[0],
                self.current_pose[1] - self.last_dropped_dist_pos[1]
            )
            if dist_since_last >= self.auto_drop_interval_dist:
                self._drop_auto_waypoint(frame_idx)
        else:
            # Degraded or lost tracking: lock position and record degraded point
            self.trajectory.append((cur_x, cur_z, cur_yaw, frame_idx, False))

        return self.current_pose

    def _drop_auto_waypoint(self, frame_idx: int):
        wp = Waypoint(
            id=self.next_waypoint_id,
            timestamp_iso=datetime.datetime.now().isoformat(),
            frame_idx=frame_idx,
            x_rel=self.current_pose[0],
            z_rel=self.current_pose[1],
            yaw_deg=self.current_pose[2],
            label=f"AUTO-WP-{self.next_waypoint_id:03d}",
            is_manual=False
        )
        self.next_waypoint_id += 1
        self.waypoints.append(wp)
        self.last_dropped_dist_pos = (self.current_pose[0], self.current_pose[1])

    def add_manual_waypoint(self, label: Optional[str] = None, frame_idx: Optional[int] = None) -> Waypoint:
        """Drop a user-marked manual waypoint at the current estimated rover pose."""
        f_idx = frame_idx if frame_idx is not None else self.last_frame_idx
        lbl = label if label else f"MANUAL-POI-{self.next_waypoint_id:03d}"
        wp = Waypoint(
            id=self.next_waypoint_id,
            timestamp_iso=datetime.datetime.now().isoformat(),
            frame_idx=f_idx,
            x_rel=self.current_pose[0],
            z_rel=self.current_pose[1],
            yaw_deg=self.current_pose[2],
            label=lbl,
            is_manual=True
        )
        self.next_waypoint_id += 1
        self.waypoints.append(wp)
        self.last_dropped_dist_pos = (self.current_pose[0], self.current_pose[1])
        return wp

    def save_to_csv(self, filepath: str) -> Path:
        """Export all breadcrumb waypoints to a standard CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp_iso", "frame_idx", "x_rel", "z_rel", "yaw_deg", "label", "is_manual"])
            for wp in self.waypoints:
                writer.writerow([
                    wp.id, wp.timestamp_iso, wp.frame_idx,
                    wp.x_rel, wp.z_rel, wp.yaw_deg,
                    wp.label, 1 if wp.is_manual else 0
                ])
        return path

    def save_to_sqlite(self, db_path: str) -> Path:
        """Persist breadcrumb waypoints into an SQLite database."""
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS waypoints (
                id INTEGER PRIMARY KEY,
                timestamp_iso TEXT,
                frame_idx INTEGER,
                x_rel REAL,
                z_rel REAL,
                yaw_deg REAL,
                label TEXT,
                is_manual INTEGER
            )
        """)
        cursor.execute("DELETE FROM waypoints")
        for wp in self.waypoints:
            cursor.execute("""
                INSERT INTO waypoints (id, timestamp_iso, frame_idx, x_rel, z_rel, yaw_deg, label, is_manual)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (wp.id, wp.timestamp_iso, wp.frame_idx, wp.x_rel, wp.z_rel, wp.yaw_deg, wp.label, 1 if wp.is_manual else 0))
        conn.commit()
        conn.close()
        return path

    def load_from_sqlite(self, db_path: str) -> List[Waypoint]:
        """Load stored waypoints from SQLite database."""
        path = Path(db_path)
        if not path.exists():
            return []
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp_iso, frame_idx, x_rel, z_rel, yaw_deg, label, is_manual FROM waypoints ORDER BY id ASC")
        rows = cursor.fetchall()
        loaded = []
        for r in rows:
            wp = Waypoint(
                id=r[0], timestamp_iso=r[1], frame_idx=r[2],
                x_rel=r[3], z_rel=r[4], yaw_deg=r[5],
                label=r[6], is_manual=bool(r[7])
            )
            loaded.append(wp)
        conn.close()
        return loaded

    def plot_breadcrumb_map(self, title: str = "2D Breadcrumb Trajectory Map") -> plt.Figure:
        """Generate high-contrast 2D Matplotlib breadcrumb map with numbered waypoints."""
        fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")

        xs = [pt[0] for pt in self.trajectory]
        zs = [pt[1] for pt in self.trajectory]

        # Draw continuous estimated trajectory
        ax.plot(
            xs, zs,
            color="#0284c7",
            linewidth=2.4,
            alpha=0.95,
            label="Monocular Trajectory (Relative)",
            zorder=3
        )

        # Highlight degraded segments
        deg_x = [pt[0] for pt in self.trajectory if not pt[4]]
        deg_z = [pt[1] for pt in self.trajectory if not pt[4]]
        if deg_x:
            ax.scatter(deg_x, deg_z, c="#dc2626", marker="x", s=25, label="Degraded VO Steps", zorder=4)

        # Plot numbered breadcrumb waypoints
        for wp in self.waypoints:
            marker_color = "#d97706" if wp.is_manual else "#16a34a"
            marker_shape = "D" if wp.is_manual else "o"
            ax.scatter(wp.x_rel, wp.z_rel, c=marker_color, marker=marker_shape, s=55, edgecolors="#ffffff", linewidth=1.0, zorder=6)
            # Label waypoint number
            ax.text(
                wp.x_rel + 0.12, wp.z_rel + 0.12,
                f"#{wp.id}",
                color="#0f172a",
                fontsize=7.5,
                fontweight="bold",
                zorder=7
            )

        # Origin Marker
        ax.scatter(0.0, 0.0, c="#16a34a", marker="*", s=160, edgecolors="#0f172a", linewidth=1.0, label="Origin (0, 0)", zorder=8)

        # Current Pose & Heading Arrow
        cur_x, cur_z, cur_yaw = self.current_pose
        ax.scatter(cur_x, cur_z, c="#d97706", marker="^", s=130, edgecolors="#0f172a", linewidth=1.0, label="Current Estimated Pose", zorder=9)

        yaw_rad = math.radians(cur_yaw)
        arrow_len = max(0.6, (max(xs, default=1) - min(xs, default=0) + max(zs, default=1) - min(zs, default=0)) * 0.08)
        dx = arrow_len * math.sin(yaw_rad)
        dz = arrow_len * math.cos(yaw_rad)
        ax.arrow(cur_x, cur_z, dx, dz, color="#d97706", width=0.03, head_width=0.18, head_length=0.18, zorder=10)

        # Last-known Pose marker if distinct
        last_x, last_z, _ = self.last_known_pose
        if math.hypot(cur_x - last_x, cur_z - last_z) > 0.01:
            ax.scatter(last_x, last_z, c="#0891b2", marker="s", s=70, edgecolors="#0f172a", label="Last Known Pose", zorder=8)

        ax.grid(color="#e2e8f0", linestyle="--", linewidth=0.8, alpha=0.9)
        ax.tick_params(colors="#475569", labelsize=8)

        ax.set_xlabel(f"Lateral X Axis [{self.scale_label}]", color="#334155", fontsize=9, fontweight="bold")
        ax.set_ylabel(f"Forward Z Axis [{self.scale_label}]", color="#334155", fontsize=9, fontweight="bold")
        ax.set_title(title, color="#0f172a", fontsize=11, fontweight="bold", pad=10)

        # Scientific Rigor Disclaimer Subtitle
        fig.text(
            0.5, 0.01,
            "SCIENTIFIC NOTICE: Monocular trajectory scale is relative/arbitrary. Video processing does NOT give exact real-world positions.",
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
