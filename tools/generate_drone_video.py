"""Synthetic Aerial Drone Video Generator for ResQNav.

Generates a realistic top-down (Nadir / Bird's-Eye View) search-and-rescue (SAR)
drone reconnaissance video for testing Visual Odometry, Visual SLAM, 2D Occupancy
Grid Mapping, and Survivor Detection without physical drone hardware.

Simulates:
- High-contrast aerial terrain with dirt, roads, vegetation, and collapsed structures.
- Structural rubble, concrete foundations, and debris with rich ORB/FAST corner features.
- Human survivor figures on the ground with high-visibility markers for SAR detection.
- Realistic 6-DOF drone flight dynamics (forward translation, banking turns, altitude drift, wind turbulence).
- Configurable tactical UAV cockpit HUD overlay.
"""

import argparse
import math
from pathlib import Path
import sys
from typing import List, Tuple, Union

import cv2
import numpy as np


class DroneSceneWorld:
    """Simulated large-scale 2D aerial ground world with disaster terrain and objects."""

    def __init__(self, world_w: int = 2400, world_h: int = 3200, seed: int = 101):
        self.world_w = world_w
        self.world_h = world_h
        np.random.seed(seed)

        # Base ground canvas: weathered concrete / dirt terrain with multi-octave texture
        self.canvas = np.zeros((world_h, world_w, 3), dtype=np.uint8)
        self._build_terrain()
        self._build_roads_and_clearways()
        self._build_building_ruins()
        self._build_rubble_and_debris()
        self._build_landmarks_and_helipads()
        self._place_survivors()

    def _build_terrain(self):
        """Create multi-frequency ground texture simulating dusty rubble and patchy vegetation."""
        # Base soil/asphalt color: slate brown-grey
        base_color = np.array([55, 65, 75], dtype=np.float32)
        noise1 = np.random.normal(0, 15, (self.world_h // 4, self.world_w // 4)).astype(np.float32)
        noise1 = cv2.resize(noise1, (self.world_w, self.world_h), interpolation=cv2.INTER_LINEAR)

        noise2 = np.random.normal(0, 8, (self.world_h // 16, self.world_w // 16)).astype(np.float32)
        noise2 = cv2.resize(noise2, (self.world_w, self.world_h), interpolation=cv2.INTER_LINEAR)

        terrain = base_color + noise1[:, :, None] + noise2[:, :, None]
        np.clip(terrain, 25, 230, out=terrain)
        self.canvas[:] = terrain.astype(np.uint8)

        # Add scattered patches of dried grass / scorched earth
        for _ in range(60):
            cx = np.random.randint(100, self.world_w - 100)
            cy = np.random.randint(100, self.world_h - 100)
            axes = (np.random.randint(40, 180), np.random.randint(40, 180))
            angle = np.random.randint(0, 180)
            # Olive drab / dirt tint
            patch_color = (np.random.randint(35, 60), np.random.randint(55, 85), np.random.randint(50, 75))
            overlay = self.canvas.copy()
            cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, patch_color, -1)
            cv2.addWeighted(overlay, 0.45, self.canvas, 0.55, 0, self.canvas)

    def _build_roads_and_clearways(self):
        """Build main paved rescue access route cutting through the disaster sector."""
        # Winding asphalt road from bottom to top
        pts = []
        n_pts = 16
        for i in range(n_pts):
            t = i / (n_pts - 1)
            y = int(self.world_h * (1.0 - t))
            # S-curve drift
            x = int(self.world_w * 0.5 + 320 * math.sin(t * math.pi * 2.2) + 120 * math.cos(t * math.pi * 4.0))
            pts.append((x, y))

        road_curve = np.array(pts, dtype=np.int32)
        # Road asphalt base
        cv2.polylines(self.canvas, [road_curve], isClosed=False, color=(35, 42, 50), thickness=110)
        # Road shoulders / curb debris
        cv2.polylines(self.canvas, [road_curve], isClosed=False, color=(70, 80, 95), thickness=116)
        # Broken centerline markings (yellow)
        for i in range(0, len(pts) - 1, 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            cv2.line(self.canvas, p1, p2, (30, 180, 220), 4)

    def _build_building_ruins(self):
        """Construct ruined concrete structural foundations and collapsed walls."""
        # Pre-place 25 collapsed building complexes
        for i in range(35):
            bx = np.random.randint(150, self.world_w - 150)
            by = np.random.randint(150, self.world_h - 150)
            bw = np.random.randint(100, 260)
            bh = np.random.randint(90, 240)
            angle = np.random.uniform(0, 360)

            rect = ((bx, by), (bw, bh), angle)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Foundation floor (dark concrete)
            cv2.fillPoly(self.canvas, [box], (45, 52, 60))
            cv2.polylines(self.canvas, [box], isClosed=True, color=(120, 130, 145), thickness=6)

            # Internal partition walls (some collapsed, some standing)
            for _ in range(4):
                px1 = bx + np.random.randint(-bw // 3, bw // 3)
                py1 = by + np.random.randint(-bh // 3, bh // 3)
                px2 = px1 + np.random.randint(-60, 60)
                py2 = py1 + np.random.randint(-60, 60)
                cv2.line(self.canvas, (px1, py1), (px2, py2), (160, 170, 185), np.random.randint(3, 7))

    def _build_rubble_and_debris(self):
        """Add dense scattered rubble piles, shattered masonry, and high-contrast rocks."""
        # 1200 high-frequency debris elements for sharp ORB / FAST corner extraction
        for _ in range(1200):
            rx = np.random.randint(50, self.world_w - 50)
            ry = np.random.randint(50, self.world_h - 50)
            size = np.random.randint(3, 14)
            val = np.random.randint(130, 245)
            # Rubble piece: small bright/dark jagged polygon
            pts = np.array([
                [rx - size, ry - size // 2],
                [rx, ry - size],
                [rx + size, ry - size // 3],
                [rx + size // 2, ry + size],
                [rx - size // 2, ry + size // 2]
            ], dtype=np.int32)
            cv2.fillPoly(self.canvas, [pts], (val, val, val))
            # Cast small shadow next to rubble
            cv2.ellipse(self.canvas, (rx + 4, ry + 4), (size, size // 2), 45, 0, 360, (20, 25, 30), -1)

    def _build_landmarks_and_helipads(self):
        """Construct distinctive ground survey datums (e.g., Helipad H, target crosses)."""
        # Survey Helipad 1
        h1 = (int(self.world_w * 0.45), int(self.world_h * 0.82))
        cv2.circle(self.canvas, h1, 75, (220, 220, 225), 8)
        cv2.circle(self.canvas, h1, 55, (20, 180, 230), 4)
        # Letter H
        cv2.line(self.canvas, (h1[0] - 25, h1[1] - 35), (h1[0] - 25, h1[1] + 35), (240, 240, 245), 10)
        cv2.line(self.canvas, (h1[0] + 25, h1[1] - 35), (h1[0] + 25, h1[1] + 35), (240, 240, 245), 10)
        cv2.line(self.canvas, (h1[0] - 25, h1[1]), (h1[0] + 25, h1[1]), (240, 240, 245), 10)

        # Tactical datum crosses
        datums = [
            (int(self.world_w * 0.35), int(self.world_h * 0.55)),
            (int(self.world_w * 0.65), int(self.world_h * 0.35)),
            (int(self.world_w * 0.50), int(self.world_h * 0.15)),
        ]
        for idx, (dx, dy) in enumerate(datums, start=1):
            cv2.drawMarker(self.canvas, (dx, dy), (0, 230, 255), markerType=cv2.MARKER_CROSS, markerSize=50, thickness=6)
            cv2.circle(self.canvas, (dx, dy), 40, (0, 230, 255), 3)
            cv2.putText(self.canvas, f"DATUM-0{idx}", (dx + 48, dy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)

    def _place_survivors(self):
        """Place realistic human survivor silhouettes and high-visibility distress markers."""
        # 4 distinct survivor locations along the flight route
        self.survivors = [
            {"pos": (int(self.world_w * 0.48), int(self.world_h * 0.68)), "status": "WAVING", "color": (30, 120, 240)},     # Hi-vis orange
            {"pos": (int(self.world_w * 0.58), int(self.world_h * 0.48)), "status": "PRONE", "color": (50, 210, 80)},       # Green jacket
            {"pos": (int(self.world_w * 0.38), int(self.world_h * 0.32)), "status": "TRAPPED", "color": (220, 50, 50)},      # Red vest
            {"pos": (int(self.world_w * 0.52), int(self.world_h * 0.18)), "status": "SITTING", "color": (230, 200, 30)},     # Yellow beacon
        ]

        for s in self.survivors:
            sx, sy = s["pos"]
            col = s["color"]

            # Ground distress tarpaulin / signal square (makes aerial detection distinct)
            cv2.rectangle(self.canvas, (sx - 24, sy - 24), (sx + 24, sy + 24), col, -1)
            cv2.rectangle(self.canvas, (sx - 24, sy - 24), (sx + 24, sy + 24), (255, 255, 255), 2)

            # Aerial human silhouette: head + torso + outstretched arms
            # Torso
            cv2.ellipse(self.canvas, (sx, sy), (8, 14), 0, 0, 360, (20, 25, 30), -1)
            # Head
            cv2.circle(self.canvas, (sx, sy - 14), 6, (40, 45, 55), -1)
            # Outstretched arms (waving signal)
            cv2.line(self.canvas, (sx - 16, sy - 6), (sx + 16, sy - 6), (20, 25, 30), 4)
            # Legs
            cv2.line(self.canvas, (sx - 5, sy + 14), (sx - 8, sy + 26), (20, 25, 30), 4)
            cv2.line(self.canvas, (sx + 5, sy + 14), (sx + 8, sy + 26), (20, 25, 30), 4)


def generate_synthetic_drone_video(
    output_path: Union[str, Path] = "data/sample_drone_recon.mp4",
    num_frames: int = 220,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    include_hud: bool = True,
    seed: int = 101,
) -> Path:
    """Generate a realistic top-down aerial drone reconnaissance video.
    
    Args:
        output_path: Target video file path (.mp4).
        num_frames: Total number of frames to render (default 220 = 8.8s at 25 FPS).
        width: Video frame width (default 640).
        height: Video frame height (default 480).
        fps: Frame rate (default 25 FPS).
        include_hud: Whether to render authentic UAV cockpit HUD telemetry.
        seed: Random seed for world procedural generation.
        
    Returns:
        Path to the generated video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing Aerial Disaster Environment (Seed={seed})...")
    world = DroneSceneWorld(world_w=2400, world_h=3400, seed=seed)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise IOError(f"Failed to open OpenCV VideoWriter for {output_path}")

    # Drone flight trajectory planning
    # Starts at bottom (Y = 2800), flies northward (upward towards Y = 400)
    # With realistic gentle yaw turns and wind vibration
    start_x = world.world_w * 0.48
    start_y = world.world_h - 450.0
    end_y = 450.0
    total_dist_y = start_y - end_y

    base_altitude_m = 20.0  # Cruise at 20 metres
    speed_mps = 3.5

    print(f"[*] Rendering {num_frames} frames of Nadir Aerial Drone Recon to {output_path.name}...")

    for f in range(num_frames):
        t = f / max(1, num_frames - 1)

        # Smooth spline path: Northward advance + realistic S-curve banking
        cur_y = start_y - t * total_dist_y
        cur_x = start_x + 180.0 * math.sin(t * math.pi * 2.0) + 60.0 * math.cos(t * math.pi * 3.5)

        # Heading angle (tangent to path + slight flight controller yaw tracking)
        dy = - (total_dist_y / num_frames)
        dx = 180.0 * (2.0 * math.pi / num_frames) * math.cos(t * math.pi * 2.0)
        heading_rad = math.atan2(dx, -dy)  # 0 = North / Up
        heading_deg = (math.degrees(heading_rad) + 360) % 360

        # Altitude profile: climb slightly then cruise
        cur_alt_m = base_altitude_m + 2.5 * math.sin(t * math.pi)

        # High-frequency flight vibration / wind turbulence (typical for micro-UAVs)
        turb_x = 0.8 * math.sin(f * 0.9) + 0.4 * math.cos(f * 1.7)
        turb_y = 0.8 * math.cos(f * 0.8) + 0.4 * math.sin(f * 1.5)
        turb_yaw = 0.4 * math.sin(f * 1.2)

        cam_center_x = cur_x + turb_x
        cam_center_y = cur_y + turb_y
        cam_yaw_deg = heading_deg + turb_yaw

        # Scale factor from altitude: higher altitude = more ground visible
        # Ground Sampling Distance (GSD): at 20m, 1 pixel = ~0.04m (4cm/pixel)
        zoom_factor = 20.0 / cur_alt_m

        # Extract rotated, scaled region of interest from world canvas
        # 2D Affine Transformation (Translation -> Rotation -> Scale)
        rot_mat = cv2.getRotationMatrix2D((cam_center_x, cam_center_y), cam_yaw_deg, zoom_factor)
        # Shift center to frame center
        rot_mat[0, 2] += (width / 2.0) - cam_center_x
        rot_mat[1, 2] += (height / 2.0) - cam_center_y

        frame = cv2.warpAffine(
            world.canvas,
            rot_mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101
        )

        # Render Drone Cockpit HUD Overlay if requested
        if include_hud:
            _draw_drone_hud(
                frame=frame,
                frame_idx=f,
                num_frames=num_frames,
                alt_m=cur_alt_m,
                heading_deg=cam_yaw_deg,
                speed_mps=speed_mps + 0.2 * math.sin(f * 0.2),
                metric_x=(cam_center_x - start_x) * 0.04,
                metric_y=(start_y - cam_center_y) * 0.04,
                width=width,
                height=height
            )

        writer.write(frame)

    writer.release()
    print(f"[+] Drone video generation complete: {output_path.resolve()} ({output_path.stat().st_size // 1024} KB)")
    return output_path


def _draw_drone_hud(
    frame: np.ndarray,
    frame_idx: int,
    num_frames: int,
    alt_m: float,
    heading_deg: float,
    speed_mps: float,
    metric_x: float,
    metric_y: float,
    width: int,
    height: int
):
    """Render authentic tactical UAV Nadir reconnaissance HUD overlay."""
    # Top Telemetry Banner
    cv2.rectangle(frame, (10, 10), (width - 10, 42), (18, 24, 34), -1)
    cv2.rectangle(frame, (10, 10), (width - 10, 42), (56, 189, 248), 1)

    top_text = (
        f"UAV-01 | RECON-SAR | NADIR CAM (90deg) | "
        f"ALT: {alt_m:4.1f}m | HDG: {heading_deg:05.1f}deg | SPD: {speed_mps:3.1f}m/s"
    )
    cv2.putText(frame, top_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 245, 250), 1, cv2.LINE_AA)

    # Bottom Telemetry Banner
    cv2.rectangle(frame, (10, height - 38), (width - 10, height - 10), (18, 24, 34), -1)
    cv2.rectangle(frame, (10, height - 38), (width - 10, height - 10), (56, 189, 248), 1)

    bottom_text = (
        f"FRAME: {frame_idx:04d}/{num_frames} | "
        f"POS: X={metric_x:+06.2f}m, Y={metric_y:+06.2f}m | MODE: AUTO_SURVEY | GPS-DENIED"
    )
    cv2.putText(frame, bottom_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (56, 189, 248), 1, cv2.LINE_AA)

    # Central Drone Gimbal Reticle / Crosshair
    cx, cy = width // 2, height // 2
    reticle_color = (0, 230, 255)  # Tactical amber
    # Center box
    cv2.rectangle(frame, (cx - 16, cy - 16), (cx + 16, cy + 16), reticle_color, 1)
    cv2.line(frame, (cx - 28, cy), (cx - 18, cy), reticle_color, 1)
    cv2.line(frame, (cx + 18, cy), (cx + 28, cy), reticle_color, 1)
    cv2.line(frame, (cx, cy - 28), (cx, cy - 18), reticle_color, 1)
    cv2.line(frame, (cx, cy + 18), (cx, cy + 28), reticle_color, 1)

    # Artificial Horizon Pitch/Roll tick marks on sides
    cv2.line(frame, (40, cy), (65, cy), (240, 245, 250), 1)
    cv2.line(frame, (width - 65, cy), (width - 40, cy), (240, 245, 250), 1)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic top-down aerial drone reconnaissance video.")
    parser.add_argument("--output", type=str, default="data/sample_drone_recon.mp4", help="Output .mp4 video path")
    parser.add_argument("--frames", type=int, default=220, help="Number of frames to generate")
    parser.add_argument("--width", type=int, default=640, help="Video frame width")
    parser.add_argument("--height", type=int, default=480, help="Video frame height")
    parser.add_argument("--fps", type=int, default=25, help="Frame rate")
    parser.add_argument("--no-hud", action="store_true", help="Disable cockpit HUD telemetry overlay")
    parser.add_argument("--seed", type=int, default=101, help="Random seed for disaster world generation")
    args = parser.parse_args()

    generate_synthetic_drone_video(
        output_path=args.output,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        include_hud=not args.no_hud,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
