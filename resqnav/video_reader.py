"""Video ingestion, frame navigation, and synthetic rover POV generation."""

import os
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Union
import cv2
import numpy as np


class VideoError(Exception):
    """Custom exception raised for video I/O, format, or decoding issues."""
    pass


class VideoReader:
    """Robust video reader supporting frame stepping, seeking, and metadata extraction."""

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, source: Union[str, Path]):
        self.source_path = Path(source)
        if not self.source_path.exists():
            raise VideoError(f"Video file not found: {self.source_path.resolve()}")

        ext = self.source_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise VideoError(
                f"Unsupported format '{ext}'. Supported video containers: "
                f"{', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        self.cap = cv2.VideoCapture(str(self.source_path))
        if not self.cap.isOpened():
            raise VideoError(
                f"Failed to open video file '{self.source_path.name}'. "
                f"The file may be corrupt or encoded with an unsupported codec."
            )

        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self._fps <= 0 or np.isnan(self._fps):
            self._fps = 30.0  # Safe fallback for variable frame-rate streams

        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._current_idx = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_seconds(self) -> float:
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    def get_metadata(self) -> Dict[str, Union[int, float, str]]:
        """Return a structured summary of video metadata."""
        return {
            "filename": self.source_path.name,
            "width": self._width,
            "height": self._height,
            "fps": round(self._fps, 2),
            "frame_count": self._frame_count,
            "duration_sec": round(self.duration_seconds, 2),
            "aspect_ratio": f"{self._width}x{self._height}",
        }

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Read the next frame in sequence."""
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, self._current_idx
        idx = self._current_idx
        self._current_idx += 1
        return True, frame, idx

    def seek_frame(self, target_idx: int) -> bool:
        """Seek to a specific 0-indexed frame number."""
        target_idx = max(0, min(target_idx, max(0, self._frame_count - 1)))
        ok = self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        if ok:
            self._current_idx = target_idx
        return ok

    def iterate_frames(
        self,
        step: int = 1,
        max_frames: Optional[int] = None,
        start_frame: int = 0
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator yielding (frame_index, frame_bgr) with downsampling step."""
        if start_frame > 0:
            self.seek_frame(start_frame)

        yielded = 0
        while self.cap.isOpened():
            if max_frames is not None and yielded >= max_frames:
                break

            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break

            yield current_pos, frame
            yielded += 1

            # Skip (step - 1) frames
            for _ in range(step - 1):
                if not self.cap.grab():
                    return

    def reset(self):
        """Rewind to the beginning."""
        self.seek_frame(0)

    def release(self):
        """Release underlying OpenCV capture handle."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def generate_synthetic_pov_video(
    output_path: Union[str, Path],
    num_frames: int = 150,
    width: int = 640,
    height: int = 480,
    fps: int = 25
) -> Path:
    """Generate a realistic robot-POV disaster-corridor video for out-of-the-box simulation.
    
    Simulates:
    - Textured terrain floor with perspective optical flow
    - Collapsed rubble concrete pillars passing on sides
    - Forward motion and gradual sweeping yaw turns
    - Corner features detectable by ORB/FAST
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Pre-generate 3D landmark features in a virtual corridor
    np.random.seed(42)
    n_landmarks = 240
    # Landmarks placed along corridor: X in [-4, 4], Y in [-1.5, 2.0], Z in [2, 45]
    lx = np.random.uniform(-4.5, 4.5, n_landmarks)
    ly = np.random.uniform(-1.5, 1.8, n_landmarks)
    lz = np.random.uniform(2.0, 45.0, n_landmarks)
    colors = np.random.randint(140, 255, (n_landmarks, 3), dtype=np.uint8)
    sizes = np.random.randint(4, 10, n_landmarks)

    # Virtual pinhole camera parameters
    fx = fy = 450.0
    cx = width / 2.0
    cy = height / 2.0

    camera_z = 0.0
    camera_x = 0.0
    yaw = 0.0

    for f in range(num_frames):
        # Frame blank dark slate canvas
        frame = np.full((height, width, 3), (18, 24, 34), dtype=np.uint8)

        # Draw textured ground grid with perspective
        horizon_y = int(cy - 20)
        cv2.line(frame, (0, horizon_y), (width, horizon_y), (35, 48, 65), 1)

        # Forward motion speed with a gentle turn around mid-run
        speed = 0.16
        camera_z += speed
        if 40 <= f <= 85:
            yaw += 0.008
            camera_x += 0.03
        elif 85 < f <= 130:
            yaw -= 0.007
            camera_x -= 0.025

        # Render ground texture strips
        for gz in np.linspace(camera_z % 2.0 + 1.0, 25.0, 15):
            proj_y = int(cy + fy * (1.2 / gz))
            if horizon_y < proj_y < height:
                cv2.line(frame, (0, proj_y), (width, proj_y), (28, 38, 52), 1)

        # Perspective projection of virtual rubble pillars & landmarks
        rel_x = lx - camera_x
        rel_z = lz - camera_z
        rel_y = ly

        # Rotate by camera yaw
        rot_x = rel_x * np.cos(yaw) - rel_z * np.sin(yaw)
        rot_z = rel_x * np.sin(yaw) + rel_z * np.cos(yaw)

        # Render points that are in front of the camera
        valid = rot_z > 0.8
        proj_u = (fx * rot_x[valid] / rot_z[valid] + cx).astype(int)
        proj_v = (fy * rel_y[valid] / rot_z[valid] + cy).astype(int)
        valid_screen = (proj_u >= 0) & (proj_u < width) & (proj_v >= 0) & (proj_v < height)

        u_screen = proj_u[valid_screen]
        v_screen = proj_v[valid_screen]
        c_screen = colors[valid][valid_screen]
        s_screen = sizes[valid][valid_screen]

        for u, v, c, s in zip(u_screen, v_screen, c_screen, s_screen):
            # Draw textured rubble blocks with high ORB corner response
            color = (int(c[0]), int(c[1]), int(c[2]))
            cv2.rectangle(frame, (u - s, v - s), (u + s, v + s), color, -1)
            cv2.rectangle(frame, (u - s, v - s), (u + s, v + s), (50, 60, 75), 1)

        # Synthetic HUD overlay
        cv2.putText(
            frame,
            f"SYNTHETIC POV STREAM | FRAME {f:04d} | RES: {width}x{height}",
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (56, 189, 248),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"SIM DISASTER CORRIDOR | Z={camera_z:.2f}m",
            (15, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    return output_path
