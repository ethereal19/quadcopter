"""ResQNav Standalone Light-Themed Web Application Server.

Runs on dedicated localhost:5050 providing:
- 100% Light Colour UI
- Drag-and-drop / File Ingestion for any video (.mp4, .avi, .mov, .mkv, .webm)
- Visual Landmark Tracking with ORB, Track IDs, and Surveyed vs Auto distinction
- Breadcrumb Navigation with numbered waypoints, CSV download, SQLite persistence
- Visual SLAM & Loop Closure with SE(2) Pose-Graph Optimization
- 2D Occupancy Grid Mapping & A* Path Planning
"""

import base64
import datetime
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
from typing import Dict, Any, Optional, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory
import numpy as np

from resqnav.landmark_tracker import LandmarkTracker, SurveyedLandmark, TrackingMetrics
from resqnav.breadcrumb_nav import BreadcrumbNavigator, Waypoint
from resqnav.slam_loop_closure import VisualSLAM, Keyframe, LoopCandidate
from resqnav.occupancy_grid import OccupancyGrid
from resqnav.path_planning import astar_search
from resqnav.perception import VideoPerceptionEngine
from resqnav.video_reader import VideoReader

STATIC_DIR = BASE_DIR / "static"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


class PipelineManager:
    """Thread-safe manager for ResQNav perception and autonomous navigation."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tracker = LandmarkTracker(
            num_features=1200,
            match_ratio=0.75,
            ransac_threshold=1.0,
            min_inliers_for_motion=18,
            focal_length=450.0
        )
        self.navigator = BreadcrumbNavigator(auto_drop_interval_dist=1.5)
        self.slam = VisualSLAM(
            min_keyframe_dist=1.2,
            min_keyframe_yaw_deg=10.0,
            temporal_window=5,
            min_loop_inliers=16,
            min_inlier_ratio=0.35
        )
        self.grid = OccupancyGrid(rows=32, cols=42, cell_size_m=0.5, default_state=OccupancyGrid.CELL_UNKNOWN)
        self.perception = VideoPerceptionEngine(use_neural=False)

        # Video State
        self.reader: Optional[VideoReader] = None
        self.current_video_name = "sample_recon.mp4"
        self.current_video_path = str(DATA_DIR / "sample_recon.mp4")
        self.is_playing = False
        self.play_speed = 1.0
        self.step_delay = 0.04
        self.frame_idx = 0
        self.total_frames = 120
        self.fps = 25.0
        self.resolution = (640, 480)

        # Frames cache
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.last_annotated_bgr: Optional[np.ndarray] = None
        self.latest_metrics: Optional[TrackingMetrics] = None

        # Start / Goal on Grid
        self.grid_start = (26, 21)
        self.grid_goal = (5, 21)
        self.astar_path: List[Tuple[int, int]] = []
        self.astar_status = "INITIALIZING"

        # Worker control
        self.stop_worker = False

        # Load initial scenario
        self.load_video(self.current_video_path, "sample_recon.mp4")

    def load_video(self, video_path: str, video_name: str) -> bool:
        with self.lock:
            self.is_playing = False
            try:
                if self.reader is not None:
                    self.reader.release()
                self.reader = VideoReader(video_path)
                meta = self.reader.get_metadata()
                self.current_video_path = video_path
                self.current_video_name = video_name
                self.total_frames = meta["frame_count"]
                self.fps = meta["fps"] or 25.0
                self.resolution = (meta["width"], meta["height"])

                # Reset all modules
                self.tracker.reset()
                self.navigator.reset()
                self.slam.reset()
                self.grid = OccupancyGrid(rows=32, cols=42, cell_size_m=0.5, default_state=OccupancyGrid.CELL_UNKNOWN)
                self.grid.set_cell(self.grid_start[0], self.grid_start[1], OccupancyGrid.CELL_FREE, hit_count=5)

                # Add surveyed ground datum benchmark
                self.tracker.surveyed_landmarks.append(SurveyedLandmark(
                    name="BENCHMARK-ALPHA",
                    world_xyz=(0.0, 0.0, 45.0),
                    description="Forward surveyed reference pillar"
                ))

                # Read frame 0
                ok, frame, f_idx = self.reader.read_frame()
                if ok and frame is not None:
                    self.frame_idx = 1
                    self.last_frame_bgr = frame
                    self.last_annotated_bgr = self._process_frame_unlocked(frame, 1)
                else:
                    self.frame_idx = 0
                    self.last_frame_bgr = None
                    self.last_annotated_bgr = None

                self._update_astar_unlocked()
                return True
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                return False

    def _process_frame_unlocked(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        """Processes one video frame across the full perception and navigation pipeline."""
        # 1. Visual Landmark Tracking (Features, Track IDs, Surveyed Reticle)
        annotated_frame, metrics = self.tracker.process_frame(frame_bgr, frame_idx=frame_idx)
        self.latest_metrics = metrics

        # 2. Breadcrumb Navigation (Pose step integration & auto-drop)
        cur_pose = self.navigator.update_pose_step(
            dx_rel=metrics.rel_dx,
            dz_rel=metrics.rel_dz,
            dyaw_deg=metrics.rel_dyaw_deg,
            frame_idx=frame_idx,
            tracking_ok=metrics.tracking_ok
        )

        # 3. Visual SLAM & Loop Closure (Keyframe evaluation & pose graph)
        kf_added, kf, loop_cand = self.slam.consider_keyframe(
            frame_bgr=frame_bgr,
            frame_idx=frame_idx,
            cur_pose=cur_pose
        )

        # 4. Perception & Occupancy Grid Mapping
        try:
            free_mask, obs_mask, _ = self.perception.analyze_frame(frame_bgr)
            origin_x, origin_y = self.grid.cell_to_metric(self.grid_start[0], self.grid_start[1])
            rover_x = origin_x + cur_pose[0]
            rover_y = origin_y + cur_pose[1]
            self.grid.update_from_frame_perception(rover_x, rover_y, cur_pose[2], free_mask, obs_mask)
        except Exception as e:
            pass

        return annotated_frame

    def step(self) -> bool:
        """Advance one frame."""
        with self.lock:
            if self.reader is None:
                return False
            ok, frame, f_idx = self.reader.read_frame()
            if not ok or frame is None:
                self.is_playing = False
                return False
            self.frame_idx = f_idx
            self.last_frame_bgr = frame
            self.last_annotated_bgr = self._process_frame_unlocked(frame, self.frame_idx)
            if self.frame_idx % 8 == 0:
                self._update_astar_unlocked()
            return True

    def _update_astar_unlocked(self):
        """Run A* path planner on current grid."""
        try:
            res = astar_search(
                self.grid,
                self.grid_start,
                self.grid_goal,
                allow_diagonal=True,
                allow_unknown=True
            )
            self.astar_path = res.path or []
            self.astar_status = "PATH_FOUND" if res.found else ("NO_PATH: " + res.status_message)
        except Exception as e:
            self.astar_path = []
            self.astar_status = f"ERROR: {e}"

    def reset(self):
        with self.lock:
            self.is_playing = False
            if self.reader is not None:
                self.reader.seek(0)
            self.tracker.reset()
            self.navigator.reset()
            self.slam.reset()
            self.grid = OccupancyGrid(rows=32, cols=42, cell_size_m=0.5, default_state=OccupancyGrid.CELL_UNKNOWN)
            self.grid.set_cell(self.grid_start[0], self.grid_start[1], OccupancyGrid.CELL_FREE, hit_count=5)

            if self.reader is not None:
                ok, frame, f_idx = self.reader.read_frame()
                if ok and frame is not None:
                    self.frame_idx = 1
                    self.last_frame_bgr = frame
                    self.last_annotated_bgr = self._process_frame_unlocked(frame, 1)
            self._update_astar_unlocked()

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            # Base64 frame
            frame_b64 = ""
            if self.last_annotated_bgr is not None:
                h, w = self.last_annotated_bgr.shape[:2]
                display_frame = self.last_annotated_bgr
                if w > 720:
                    scale = 720.0 / w
                    display_frame = cv2.resize(self.last_annotated_bgr, (720, int(h * scale)))
                _, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

            m = self.latest_metrics
            cur_pose = self.navigator.current_pose

            # Trajectory
            traj_data = [
                {"x": round(float(pt[0]), 3), "z": round(float(pt[1]), 3), "yaw": round(float(pt[2]), 1), "valid": bool(pt[4])}
                for pt in self.navigator.trajectory
            ]

            # Waypoints
            wp_data = [
                {
                    "id": wp.label,
                    "x": round(float(wp.x_rel), 3),
                    "z": round(float(wp.z_rel), 3),
                    "yaw_deg": round(float(wp.yaw_deg), 1),
                    "timestamp": wp.timestamp_iso,
                    "is_manual": wp.is_manual
                }
                for wp in self.navigator.waypoints
            ]

            # Keyframes & Loop Closures
            kf_data = [
                {
                    "id": kf.id,
                    "frame_idx": kf.frame_idx,
                    "x": round(float(kf.pose_x), 3),
                    "z": round(float(kf.pose_z), 3),
                    "yaw_deg": round(float(kf.yaw_deg), 1)
                }
                for kf in self.slam.keyframes
            ]

            loop_edges_data = [
                {
                    "from_kf": f"KF-{edge.query_kf_id:03d}",
                    "to_kf": f"KF-{edge.match_kf_id:03d}",
                    "inliers": edge.inliers,
                    "confidence": round(float(edge.confidence), 3)
                }
                for edge in self.slam.loop_edges
            ]

            opt_poses_data = []
            if self.slam.optimized_poses:
                opt_poses_data = [
                    {"x": round(float(p[0]), 3), "z": round(float(p[1]), 3), "yaw": round(float(p[2]), 1)}
                    for p in self.slam.optimized_poses
                ]

            # Grid Matrix
            grid_matrix = self.grid.grid.tolist()

            return {
                "video": {
                    "name": self.current_video_name,
                    "frame_idx": self.frame_idx,
                    "total_frames": self.total_frames,
                    "fps": round(self.fps, 1),
                    "resolution": f"{self.resolution[0]} × {self.resolution[1]}",
                    "is_playing": self.is_playing,
                    "speed": self.play_speed
                },
                "telemetry": {
                    "quality_score": round(m.quality_score, 1) if m else 0.0,
                    "quality_grade": m.quality_grade if m else "INITIALIZING",
                    "is_tracking_ok": m.tracking_ok if m else True,
                    "detected_features": m.total_features if m else 0,
                    "matched_features": m.matched_features if m else 0,
                    "inlier_matches": m.inliers if m else 0,
                    "inlier_ratio": round(m.inlier_ratio * 100.0, 1) if m else 0.0,
                    "persistent_tracks": m.persistent_tracks if m else 0,
                    "surveyed_visible": len(self.tracker.surveyed_landmarks),
                    "pose_x": round(float(cur_pose[0]), 3),
                    "pose_y": 0.0,
                    "pose_z": round(float(cur_pose[1]), 3),
                    "yaw_deg": round(float(cur_pose[2]), 1)
                },
                "image": frame_b64,
                "breadcrumbs": {
                    "trajectory": traj_data,
                    "waypoints": wp_data
                },
                "slam": {
                    "keyframes": kf_data,
                    "loop_edges": loop_edges_data,
                    "optimized_poses": opt_poses_data,
                    "total_loops": len(self.slam.loop_edges)
                },
                "grid": {
                    "rows": self.grid.rows,
                    "cols": self.grid.cols,
                    "matrix": grid_matrix,
                    "start": self.grid_start,
                    "goal": self.grid_goal,
                    "path": self.astar_path,
                    "status": self.astar_status
                }
            }


# Initialize singleton pipeline
pipeline = PipelineManager()


def background_worker():
    """Background playback stepper."""
    while not pipeline.stop_worker:
        if pipeline.is_playing:
            success = pipeline.step()
            if not success:
                pipeline.is_playing = False
            sleep_time = max(0.01, pipeline.step_delay / max(0.2, pipeline.play_speed))
            time.sleep(sleep_time)
        else:
            time.sleep(0.04)


worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()


# ------------------------------------------------------------------------------
# REST API ROUTES
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(pipeline.get_state())


@app.route("/api/control", methods=["POST"])
def control():
    data = request.get_json() or {}
    action = data.get("action", "")

    if action == "play":
        pipeline.is_playing = True
    elif action == "pause":
        pipeline.is_playing = False
    elif action == "toggle":
        pipeline.is_playing = not pipeline.is_playing
    elif action == "step":
        pipeline.is_playing = False
        pipeline.step()
    elif action == "reset":
        pipeline.reset()
    elif action == "speed":
        speed = float(data.get("value", 1.0))
        pipeline.play_speed = max(0.2, min(8.0, speed))
    elif action == "add_waypoint":
        with pipeline.lock:
            cur = pipeline.navigator.current_pose
            wp = pipeline.navigator.add_manual_waypoint(
                frame_idx=pipeline.frame_idx,
                label=f"MANUAL-WP-{len(pipeline.navigator.waypoints)+1:03d}"
            )
            return jsonify({"success": True, "waypoint_id": wp.label})
    elif action == "seek":
        target = int(data.get("frame", 0))
        with pipeline.lock:
            if pipeline.reader is not None:
                pipeline.reader.seek(target)
                ok, frame, f_idx = pipeline.reader.read_frame()
                if ok and frame is not None:
                    pipeline.frame_idx = target
                    pipeline.last_frame_bgr = frame
                    pipeline.last_annotated_bgr = pipeline._process_frame_unlocked(frame, target)

    return jsonify({"success": True, "is_playing": pipeline.is_playing, "speed": pipeline.play_speed})


@app.route("/api/samples", methods=["GET"])
def list_samples():
    samples = []
    for fn in ["sample_recon.mp4", "sample_recon_curved.mp4", "sample_recon_loop.mp4", "sample_drone_recon.mp4"]:
        p = DATA_DIR / fn
        if p.exists():
            desc = "Straight Recon Corridor (120 frames)"
            if "curved" in fn:
                desc = "Obstacle S-Curve Avoidance (120 frames)"
            elif "loop" in fn:
                desc = "360° Loop Closure Circuit (160 frames)"
            elif "drone" in fn:
                desc = "Nadir Aerial SAR Drone Recon (160 frames)"
            samples.append({
                "filename": fn,
                "name": fn.replace(".mp4", "").replace("_", " ").title(),
                "description": desc,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2)
            })
    return jsonify({"samples": samples})


@app.route("/api/load_sample", methods=["POST"])
def load_sample():
    data = request.get_json() or {}
    filename = data.get("filename", "sample_recon.mp4")
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return jsonify({"success": False, "error": f"Sample {filename} not found"}), 404

    ok = pipeline.load_video(str(filepath), filename)
    return jsonify({"success": ok, "video_name": filename})


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return jsonify({"success": False, "error": f"Unsupported video extension: {ext}"}), 400

    save_path = UPLOAD_DIR / file.filename
    file.save(str(save_path))

    ok = pipeline.load_video(str(save_path), file.filename)
    if ok:
        return jsonify({"success": True, "video_name": file.filename})
    else:
        return jsonify({"success": False, "error": "Failed to decode uploaded video stream"}), 500


@app.route("/api/export_csv", methods=["GET"])
def export_csv():
    csv_path = DATA_DIR / "resqnav_web_waypoints.csv"
    pipeline.navigator.save_to_csv(str(csv_path))
    return send_file(
        str(csv_path),
        mimetype="text/csv",
        as_attachment=True,
        download_name="resqnav_waypoints.csv"
    )


@app.route("/api/save_sqlite", methods=["POST"])
def save_sqlite():
    db_path = DATA_DIR / "resqnav_web_waypoints.db"
    count = pipeline.navigator.save_to_sqlite(str(db_path))
    return jsonify({"success": True, "count": count, "db_path": str(db_path)})


@app.route("/api/plan_path", methods=["POST"])
def plan_path():
    data = request.get_json() or {}
    start_r = int(data.get("start_r", pipeline.grid_start[0]))
    start_c = int(data.get("start_c", pipeline.grid_start[1]))
    goal_r = int(data.get("goal_r", pipeline.grid_goal[0]))
    goal_c = int(data.get("goal_c", pipeline.grid_goal[1]))

    with pipeline.lock:
        pipeline.grid_start = (start_r, start_c)
        pipeline.grid_goal = (goal_r, goal_c)
        pipeline._update_astar_unlocked()

    return jsonify({
        "success": True,
        "status": pipeline.astar_status,
        "path_length": len(pipeline.astar_path),
        "path": pipeline.astar_path
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  ResQNav Light-Themed Mission Control Web App")
    print("  Listening on: http://localhost:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
