"""Visualization utilities for 2D Occupancy Grids, VO Trajectories, and Telemetry HUD."""

from typing import List, Optional, Set, Tuple
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np

from resqnav.occupancy_grid import OccupancyGrid
from resqnav.visual_odometry import VOMetrics


def plot_occupancy_grid(
    grid: OccupancyGrid,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[List[Tuple[int, int]]] = None,
    explored_nodes: Optional[Set[Tuple[int, int]]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Render high-contrast 2D occupancy grid map with A* path overlay.
    
    Color Convention:
    - Free (0): Crisp clean white/cream (#f8fafc)
    - Occupied (1): Dark charcoal/slate (#0f172a)
    - Unknown (-1): Textured medium slate (#475569)
    - Explored nodes: Soft cyber-cyan tint
    - Path: Bold high-visibility electric blue / emerald
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    # Map raw ternary values [-1, 0, 1] into display indices [0, 1, 2]
    # -1 (Unknown) -> 0
    #  0 (Free)    -> 1
    #  1 (Occupied)-> 2
    display_matrix = np.zeros((grid.rows, grid.cols), dtype=np.uint8)
    display_matrix[grid.grid == OccupancyGrid.CELL_UNKNOWN] = 0
    display_matrix[grid.grid == OccupancyGrid.CELL_FREE] = 1
    display_matrix[grid.grid == OccupancyGrid.CELL_OCCUPIED] = 2

    # Custom colormap for states: Unknown (#94a3b8), Free (#ffffff), Occupied (#1e293b)
    cmap = ListedColormap(["#94a3b8", "#ffffff", "#1e293b"])

    ax.imshow(
        display_matrix,
        cmap=cmap,
        origin="upper",
        extent=[0, grid.cols, grid.rows, 0],
        vmin=0,
        vmax=2,
        interpolation="nearest"
    )

    # Overlay explored nodes if provided
    if explored_nodes:
        exp_r = [r + 0.5 for r, c in explored_nodes]
        exp_c = [c + 0.5 for r, c in explored_nodes]
        ax.scatter(exp_c, exp_r, c="#38bdf8", s=18, alpha=0.45, marker="s", label="A* Explored Nodes")

    # Overlay optimal path
    if path and len(path) > 0:
        path_c = [c + 0.5 for r, c in path]
        path_r = [r + 0.5 for r, c in path]
        ax.plot(
            path_c, path_r,
            color="#2563eb",
            linewidth=3.2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=f"Planned Path ({len(path)} cells)",
            zorder=6
        )
        # Waypoint nodes along path
        ax.scatter(path_c, path_r, c="#0284c7", s=14, zorder=7)

    # Overlay Start Position
    if start and grid.is_valid(start[0], start[1]):
        ax.scatter(
            start[1] + 0.5, start[0] + 0.5,
            c="#16a34a",
            s=180,
            marker="*",
            edgecolors="#0f172a",
            linewidth=1.2,
            label=f"Start S: ({start[0]}, {start[1]})",
            zorder=10
        )

    # Overlay Goal Position
    if goal and grid.is_valid(goal[0], goal[1]):
        ax.scatter(
            goal[1] + 0.5, goal[0] + 0.5,
            c="#dc2626",
            s=140,
            marker="X",
            edgecolors="#0f172a",
            linewidth=1.2,
            label=f"Goal G: ({goal[0]}, {goal[1]})",
            zorder=10
        )

    # Gridlines aligned to cell boundaries
    ax.set_xticks(np.arange(0, grid.cols + 1, 5))
    ax.set_yticks(np.arange(0, grid.rows + 1, 5))
    ax.grid(color="#cbd5e1", linestyle=":", linewidth=0.6, alpha=0.6)

    ax.tick_params(colors="#475569", labelsize=8)
    ax.set_xlabel("Grid Column (X Cells)", color="#334155", fontsize=9, fontweight="bold", labelpad=4)
    ax.set_ylabel("Grid Row (Y Cells)", color="#334155", fontsize=9, fontweight="bold", labelpad=4)

    title_text = title or grid.mode_label
    ax.set_title(title_text, color="#0f172a", fontsize=11, fontweight="bold", pad=8)

    # Custom Legend
    legend_patches = [
        mpatches.Patch(color="#ffffff", label="Free (Traversable)"),
        mpatches.Patch(color="#1e293b", label="Occupied (Obstacle)"),
        mpatches.Patch(color="#94a3b8", label="Unknown Zone"),
    ]
    if path:
        legend_patches.append(mpatches.Patch(color="#2563eb", label="A* Planned Path"))

    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.12),
        ncol=4,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        labelcolor="#1e293b",
        fontsize=8
    )

    fig.tight_layout()
    return fig


def plot_trajectory(
    trajectory: List[Tuple[float, float, bool]],
    current_metrics: Optional[VOMetrics] = None,
    title: str = "Monocular Visual Odometry Trajectory"
) -> plt.Figure:
    """Plot top-down (X, Z) ground-plane camera trajectory with scale ambiguity disclaimer."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    if not trajectory:
        ax.text(0.5, 0.5, "No VO motion telemetry available", color="#64748b", ha="center", va="center")
        return fig

    # Separate trajectory by tracking validity
    xs = [pt[0] for pt in trajectory]
    zs = [pt[1] for pt in trajectory]
    valid_flags = [pt[2] for pt in trajectory]

    # Draw continuous trajectory line
    ax.plot(
        xs, zs,
        color="#0284c7",
        linewidth=2.4,
        alpha=0.95,
        label="Estimated Camera Path",
        zorder=3
    )

    # Highlight degraded/lost tracking segments with red crosses
    failed_x = [pt[0] for pt in trajectory if not pt[2]]
    failed_z = [pt[1] for pt in trajectory if not pt[2]]
    if failed_x:
        ax.scatter(failed_x, failed_z, c="#dc2626", marker="x", s=30, label="Tracking Degraded", zorder=4)

    # Mark Origin (0, 0)
    ax.scatter(0.0, 0.0, c="#16a34a", s=110, marker="o", edgecolors="#0f172a", linewidth=1.2, label="Origin (0, 0)", zorder=5)

    # Current Rover Pose marker and heading vector
    if len(xs) > 0:
        cur_x, cur_z = xs[-1], zs[-1]
        ax.scatter(cur_x, cur_z, c="#d97706", s=140, marker="^", edgecolors="#0f172a", linewidth=1.2, label="Current Rover Pose", zorder=6)

        # Draw heading direction arrow
        if current_metrics is not None:
            yaw_rad = np.radians(current_metrics.yaw_deg)
            arrow_len = max(0.5, (max(xs) - min(xs) + max(zs) - min(zs)) * 0.08)
            dx = arrow_len * np.sin(yaw_rad)
            dz = arrow_len * np.cos(yaw_rad)
            ax.arrow(cur_x, cur_z, dx, dz, color="#d97706", width=0.03, head_width=0.15, head_length=0.15, zorder=7)

    ax.grid(color="#e2e8f0", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.tick_params(colors="#475569", labelsize=8)

    ax.set_xlabel("Relative Lateral X Axis (Arbitrary Units)", color="#334155", fontsize=9, fontweight="bold")
    ax.set_ylabel("Relative Forward Z Axis (Arbitrary Units)", color="#334155", fontsize=9, fontweight="bold")

    ax.set_title(title, color="#0f172a", fontsize=11, fontweight="bold", pad=12)

    # Mandatory Scientific Scale Ambiguity Disclaimer Subtitle
    fig.text(
        0.5, 0.01,
        "CRITICAL NOTE: Monocular VO has unknown absolute scale. Units are arbitrary relative units, NOT metres.",
        color="#b45309",
        fontsize=7.5,
        ha="center",
        fontfamily="monospace"
    )

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        labelcolor="#1e293b",
        fontsize=8
    )

    ax.axis("equal")
    fig.tight_layout()
    return fig


def draw_hud_on_frame(
    frame_bgr: np.ndarray,
    metrics: VOMetrics,
    frame_number: int,
    fps: float
) -> np.ndarray:
    """Burn tactical telemetry HUD overlay into video frame."""
    hud = frame_bgr.copy()
    h, w = hud.shape[:2]

    # Semi-transparent top and bottom banner bars
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (8, 13, 21), -1)
    cv2.rectangle(overlay, (0, h - 30), (w, h), (8, 13, 21), -1)
    cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)

    # Topbar text
    status_color = (16, 185, 129) if metrics.tracking_ok else (239, 68, 68)
    cv2.putText(
        hud,
        f"RESQNAV VO | STATUS: {metrics.status}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        hud,
        f"FRAME: {frame_number:04d} | {fps:.1f} FPS",
        (w - 200, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (226, 232, 240),
        1,
        cv2.LINE_AA
    )

    # Bottom bar telemetry: Keypoints, Inliers, Relative Pose
    telemetry_str = (
        f"KPS: {metrics.num_keypoints} | MATCHES: {metrics.num_matches} | "
        f"INLIERS: {metrics.num_inliers} ({metrics.inlier_ratio:.0%}) | "
        f"POSE: ({metrics.x:+.2f}, {metrics.z:+.2f}) rel | YAW: {metrics.yaw_deg:+.1f} deg"
    )
    cv2.putText(
        hud,
        telemetry_str,
        (12, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (56, 189, 248),
        1,
        cv2.LINE_AA
    )

    # Corner brackets [ ]
    b_len = 16
    color = (56, 189, 248)
    # Top-Left
    cv2.line(hud, (10, 44), (10 + b_len, 44), color, 2)
    cv2.line(hud, (10, 44), (10, 44 + b_len), color, 2)
    # Top-Right
    cv2.line(hud, (w - 10, 44), (w - 10 - b_len, 44), color, 2)
    cv2.line(hud, (w - 10, 44), (w - 10, 44 + b_len), color, 2)
    # Bottom-Left
    cv2.line(hud, (10, h - 38), (10 + b_len, h - 38), color, 2)
    cv2.line(hud, (10, h - 38), (10, h - 38 - b_len), color, 2)
    # Bottom-Right
    cv2.line(hud, (w - 10, h - 38), (w - 10 - b_len, h - 38), color, 2)
    cv2.line(hud, (w - 10, h - 38), (w - 10, h - 38 - b_len), color, 2)

    return hud
