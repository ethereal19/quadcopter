# ResQNav: Video-Based Autonomous Search & Rescue Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Tests: 20 Passed](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](tests/)

**ResQNav** is a video-driven simulation and testing environment for low-cost, GPS-denied autonomous ground vehicles (UGVs) operating in disaster reconnaissance and search-and-rescue (SAR) missions.

Designed for algorithm verification **before** physical rover hardware (ESP32, motors, sensors) is assembled, ResQNav processes prerecorded robot-POV video footage to perform:
1. **Monocular Visual Odometry (VO)**: Feature tracking, RANSAC outlier rejection, and relative camera trajectory estimation.
2. **2D Ternary Occupancy Grid Mapping**: Clear separation between Mode A (Simulated Benchmark Maps) and Mode B (Experimental Observation-Derived Maps).
3. **Heap-Based A\* Path Planning**: Guaranteed optimal, collision-free waypoint generation with strict diagonal corner-cutting prevention and configurable unknown-zone policies.
4. **Interactive Mission Dashboard**: Streamlit interface displaying synchronized video feeds, feature flow vectors, metric grids, and real-time telemetry.

---

## 1. System Architecture

```text
rover/
├── resqnav/
│   ├── __init__.py           # Package namespace definition
│   ├── config.py             # VO, Grid, Planner, and App dataclass schemas
│   ├── video_reader.py       # Robust OpenCV video ingestion & synthetic POV generator
│   ├── visual_odometry.py    # Monocular ORB + RANSAC + Essential Matrix pose estimator
│   ├── occupancy_grid.py     # 2D ternary grid (Free, Occupied, Unknown) & Mode A/B presets
│   ├── path_planning.py      # Heap-based A* search with corner-cutting prevention
│   ├── visualization.py      # Matplotlib grid/trajectory visualizers & OpenCV HUD burner
│   └── app.py                # Interactive Streamlit local mission control dashboard
├── tests/
│   ├── test_astar.py             # 10 automated tests for A* edge cases & constraints
│   ├── test_occupancy_grid.py    # 6 tests for ternary states, bounds & metric projections
│   └── test_visual_odometry.py   # 4 tests for VO initialization, black frames & tracking
├── data/
│   └── sample_recon.mp4      # Pre-generated synthetic POV search-and-rescue corridor video
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation & execution guide
```

---

## 2. Windows Setup & Installation

### Step 1: Open PowerShell or Command Prompt
Open Windows Terminal, PowerShell, or Command Prompt and navigate to the project directory:
```powershell
cd c:\Users\HP\Documents\Anveshak---a-system-for-offroad-autonomous-ground-vehicle-main\rover
```

### Step 2: Create and Activate a Python Virtual Environment
Use Python 3.10, 3.11, 3.12, or 3.13:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1

# (Or in CMD.exe)
# .\.venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
Install the verified dependencies from `requirements.txt`:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 3. Running the Interactive Dashboard

Launch the Streamlit testing application with a single command:
```powershell
streamlit run resqnav/app.py
```
*(Alternatively, you can run directly with: `python -m streamlit run resqnav/app.py`)*

Once started, open your web browser to:
```text
http://localhost:8501
```

---

## 4. Operational Workflow & Walkthrough

### Step 1: Ingest Video Footage
In the left sidebar under **Video Source**:
- **Built-in Disaster POV Simulation Video**: Uses `data/sample_recon.mp4`, which includes textured ground planes, collapsed rubble pillars, and simulated forward rover motion.
- **Upload Video File**: Upload your own `.mp4`, `.avi`, or `.mov` robot-POV footage.

### Step 2: Run Monocular Visual Odometry
1. Adjust the **Frame Downsampling Step** (default: `2`) and **Maximum Frames Limit** (default: `120`).
2. Click **▶ Run Visual Odometry Tracking**.
3. Observe:
   - **Video Feed**: Live ORB keypoints (green = inliers, red = outliers) with optical flow vectors and a tactical telemetry HUD.
   - **Estimated Trajectory**: Top-down (X, Z) relative ground-plane trajectory with heading vector.

### Step 3: Select an Occupancy Grid Environment
In the main panel, choose an environment preset:
- **Disaster Rubble Benchmark (Mode A)**: Collapsed concrete structures, debris clusters, and unknown perimeter zones.
- **Warehouse Depot Shelves (Mode A)**: Industrial storage aisles with narrow transit choke points.
- **Subterranean Tunnel (Mode A)**: Winding underground passage with rocky walls.
- **Experimental Video-Derived Grid (Mode B)**: Dynamic grid mapping driven strictly by actual VO visual observations.

### Step 4: Configure Start, Goal, and A\* Search Constraints
1. Select the **Start Cell [S]** `(Row, Col)` and **Goal Cell [G]** `(Row, Col)`.
2. Configure search rules:
   - **Allow Diagonal (8-way)**: Enables diagonal movement at $\sqrt{2} \approx 1.414$ relative cost.
   - **Prevent Corner-Cutting**: Strictly forbids diagonal steps that clip adjacent obstacle corners, guaranteeing rover safety.
   - **Allow Unknown Cells**: When unchecked (safe default), unexplored cells are treated as impassable walls.
3. Observe the optimal path, cell length, total cost, explored nodes count, and solver compute time (typically `< 2.0 ms`).

---

## 5. Automated Test Suite

Run the automated test suite using `pytest`:
```powershell
pytest tests/ -v
```

### Test Coverage (20 Tests Passed):
- `tests/test_astar.py`:
  - `test_path_in_empty_grid`: Direct routing without obstacles.
  - `test_path_avoids_obstacles`: Collision-free detour around central obstacles.
  - `test_unreachable_goal_returns_no_path`: Clean detection of walled-off goals.
  - `test_invalid_start_or_goal`: Out-of-bounds start/goal input rejection.
  - `test_occupied_start_or_goal`: Prevention of paths starting/ending on obstacles.
  - `test_unknown_cells_policy`: Correct enforcement of the `allow_unknown` safety flag.
  - `test_diagonal_corner_cutting_prevention`: Rejection of diagonal corner clipping.
  - `test_path_adjacency_and_validity`: Step-by-step continuity and traversability check.
  - `test_same_start_and_goal`: Zero-cost trivial path when start equals goal.
  - `test_heuristic_admissibility`: Mathematical consistency of Octile, Euclidean, and Manhattan heuristics.
- `tests/test_occupancy_grid.py`:
  - `test_grid_initialization_and_cell_states`: Ternary states (`-1`, `0`, `1`).
  - `test_grid_boundaries`: Safe handling of out-of-bounds coordinates.
  - `test_metric_conversions`: Continuous metric $(X, Y)$ to discrete grid cell mapping.
  - `test_block_fill`: Region filling with obstacle blocks.
  - `test_mode_a_presets`: Benchmark map generation.
  - `test_mode_b_experimental_video_grid`: Observation-backed mapping with unknown zone retention.
- `tests/test_visual_odometry.py`:
  - `test_vo_initialization`: Identity pose initialization.
  - `test_vo_handles_black_frame`: Zero-texture / low-contrast failure handling without crashing.
  - `test_vo_tracks_synthetic_textured_motion`: Feature tracking on 3D corridor video stream.
  - `test_vo_reset`: Clean reset of trajectory history.

---

## 6. Scientific Rigor: Monocular Scale Ambiguity

### Why Monocular Visual Odometry Has Unknown Metric Scale
In monocular pinhole vision:
$$u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$
Multiplying $(X, Y, Z)$ by any positive scalar $\lambda > 0$ produces the exact same pixel projection $(u, v)$. 

Consequently, decomposing the Essential Matrix $E = [t]_\times R$ yields a unit-norm direction vector $\|t\| = 1.0$. Without an external baseline (e.g., stereo separation, known camera height above flat ground, wheel odometry, or IMU accelerometer integration), absolute distance cannot be determined.

**ResQNav Standard**: All trajectory plots in this software are explicitly labeled in **arbitrary relative units**, not metres. We do not invent metric scale where none physically exists.

---

## 7. Roadmap to Physical Rover Hardware Integration

When hardware is ready:
1. **Metric Scale Observability**: Integrate wheel encoder ticks or IMU data into an Extended Kalman Filter (EKF) to fuse metric translation with VO orientation.
2. **Dense Obstacle Mapping**: Mount an RGB-D camera (e.g., Intel RealSense) or 2D LiDAR to project real 3D point clouds onto the 2D occupancy grid, replacing the Mode A benchmark maps.
3. **Live Camera Ingestion**: Update `resqnav/video_reader.py` to connect directly to the rover's live ESP32-CAM MJPEG stream (`http://192.168.4.1:81/stream`).
4. **Closed-Loop Actuation**: Route A\* trajectory waypoints to the ESP32 motor controller REST endpoint (`/action?go=forward&speed=55`).

---

## 8. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `VideoError: Failed to open video file` | Missing codec or corrupt file container | Convert video to standard H.264 MP4 using HandBrake or ffmpeg (`ffmpeg -i input.mov -c:v libx264 output.mp4`). |
| Streamlit processing feels slow | High resolution or 60 FPS video | Increase **Frame Downsampling Step** to `4` or reduce **Maximum Frames Limit** to `100` in the sidebar. |
| `ModuleNotFoundError: No module named 'streamlit'` | Virtual environment not activated | Activate `.venv` (`.\.venv\Scripts\Activate.ps1`) and run `python -m pip install -r requirements.txt`. |
| Tracking status reports `DEGRADED_MATCHES` | Motion blur or dark scene | Ensure the recorded POV camera has adequate illumination and avoid rapid angular jerks. |
