"""ResQNav: Autonomous Ground Vehicle Simulation Dashboard.

Integrated Navigation Dashboard providing:
1. Status Panel: Video frame #, feature count, match inliers, tracking quality, relative trajectory pose, loop-closure events, processing status.
2. Panel A: Live Video Playback + Persistent Visual Landmark Tracking (Track IDs, RANSAC verification, Surveyed vs Auto distinction).
3. Panel B: Breadcrumb Navigation + Waypoints (Trajectory, CSV/SQLite persistence, 2D breadcrumb map, Pause/Resume/Replay).
4. Panel C: Visual SLAM + Loop Closure (Keyframe database, appearance + geometric RANSAC verification, SE(2) pose-graph optimization, Before vs After trajectory comparison).
5. 2D Occupancy Grid Mapping & A* Path Planning Studio.
"""

from dataclasses import asdict
import datetime
import os
from pathlib import Path
import tempfile
import time
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from resqnav.config import AppConfig, GridConfig, PlannerConfig, VOConfig
from resqnav.landmark_tracker import LandmarkTracker, TrackingMetrics, SurveyedLandmark
from resqnav.breadcrumb_nav import BreadcrumbNavigator, Waypoint
from resqnav.slam_loop_closure import VisualSLAM, Keyframe, LoopCandidate
from resqnav.occupancy_grid import OccupancyGrid
from resqnav.path_planning import astar_search, validate_path
from resqnav.perception import VideoPerceptionEngine
from resqnav.video_reader import VideoError, VideoReader, generate_synthetic_pov_video
from resqnav.visualization import draw_hud_on_frame, plot_occupancy_grid, plot_trajectory
from resqnav.visual_odometry import MonocularVisualOdometry, VOMetrics

# Configure page layout and tactical styling
st.set_page_config(
    page_title="ResQNav // Visual Landmark Tracking, Breadcrumb Nav & SLAM",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Mission Control CSS injection (Bulletproof Light Theme)
st.markdown("""
<style>
    /* Force Light Theme across all Streamlit containers */
    html, body, .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stMain"], 
    [data-testid="stMainBlockContainer"],
    section.main, 
    .main, 
    .block-container, 
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    /* Top Header and Sidebar */
    header, [data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stSidebarNav"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #1e293b !important;
    }

    /* Force all text elements to high-contrast dark slate */
    .stApp p, .stApp span, .stApp label, .stApp div, 
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #0f172a !important;
    }
    
    .stApp h1, .stApp h2, .stApp h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Status Panel Banner Card */
    .status-panel-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-top: 3px solid #0284c7 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05) !important;
        color: #0f172a !important;
    }
    .panel-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        letter-spacing: 0.6px !important;
        color: #0369a1 !important;
        border-bottom: 2px solid #e0f2fe !important;
        padding-bottom: 6px !important;
        margin-bottom: 12px !important;
    }
    .status-ok { color: #16a34a !important; font-weight: bold !important; }
    .status-warn { color: #d97706 !important; font-weight: bold !important; }
    .status-err { color: #dc2626 !important; font-weight: bold !important; }
    
    .disclaimer-box {
        background: #fffbeb !important;
        border: 1px solid #fef3c7 !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        font-size: 12px !important;
        color: #92400e !important;
        margin: 8px 0 !important;
    }
    .pill-badge {
        display: inline-block !important;
        padding: 3px 10px !important;
        border-radius: 4px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        margin-right: 6px !important;
    }
    .pill-auto { 
        background: #f0f9ff !important; 
        color: #0369a1 !important; 
        border: 1px solid #bae6fd !important; 
    }
    .pill-survey { 
        background: #fefce8 !important; 
        color: #854d0e !important; 
        border: 1px solid #fde047 !important; 
    }
    .light-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        margin-top: 8px !important;
        color: #0f172a !important;
    }

    /* Tabs, Expanders & Metrics */
    [data-baseweb="tab-list"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    [data-baseweb="tab"] {
        color: #64748b !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        color: #0284c7 !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(2, 132, 199, 0.25) !important;
    }
    .stButton > button:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.35) !important;
    }
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background-color: #f1f5f9 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"], [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1px dashed #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
    }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span {
        color: #64748b !important;
    }

    /* Radio buttons & Selectboxes */
    [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] span {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    .stRadio label {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "landmark_tracker" not in st.session_state:
    st.session_state.landmark_tracker = LandmarkTracker()
if "breadcrumb_nav" not in st.session_state:
    st.session_state.breadcrumb_nav = BreadcrumbNavigator()
if "slam_engine" not in st.session_state:
    st.session_state.slam_engine = VisualSLAM()
if "perception_engine" not in st.session_state:
    st.session_state.perception_engine = VideoPerceptionEngine()
if "video_grid" not in st.session_state:
    st.session_state.video_grid = None
if "last_tracking_metrics" not in st.session_state:
    st.session_state.last_tracking_metrics = None
if "latest_annotated_frame" not in st.session_state:
    st.session_state.latest_annotated_frame = None
if "mission_completed" not in st.session_state:
    st.session_state.mission_completed = False
if "mission_paused" not in st.session_state:
    st.session_state.mission_paused = False


# ------------------------------------------------------------------------------
# TOP HEADER
# ------------------------------------------------------------------------------
st.title("📡 ResQNav // Visual Landmark Tracking, Breadcrumb Nav & SLAM")
st.caption(
    "Integrated robotics testing environment for GPS-denied autonomous navigation: "
    "Persistent Feature Track IDs • Breadcrumb Waypoints (CSV/SQLite) • Visual SLAM Loop Closure & SE(2) Pose-Graph Optimization."
)


# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS & CONFIGURATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙ Control Deck")

    app_mode = st.radio(
        "Navigation Dashboard Mode",
        [
            "Mission Control (Panels A, B, C)",
            "Occupancy Grid & A* Planning Studio",
            "Visual Odometry & Sensor Diagnostics"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("📹 Video Source Ingestion")
    video_source_type = st.radio(
        "Source Selection",
        [
            "Closed-Loop Recon Arena (Loop Closure Demo)",
            "Upload Custom Video (.mp4, .avi, .mov)",
            "Corridor Rubble Recon (Straight)",
            "Cavern Winding Recon (S-Curve)"
        ],
        index=0
    )

    video_path: Optional[Path] = None

    if video_source_type == "Closed-Loop Recon Arena (Loop Closure Demo)":
        sample_loop = Path("data/sample_recon_loop.mp4")
        if not sample_loop.exists():
            sample_loop = Path("data/sample_recon.mp4")
        video_path = sample_loop
        st.success("Loaded Closed-Loop Recon Video (Returns to origin with Surveyed Datum).")
    elif video_source_type == "Upload Custom Video (.mp4, .avi, .mov)":
        uploaded_file = st.file_uploader(
            "Upload Prerecorded Robot POV Video",
            type=["mp4", "avi", "mov", "mkv", "webm"]
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = Path(tfile.name)
            st.success(f"Loaded '{uploaded_file.name}'")
    elif video_source_type == "Corridor Rubble Recon (Straight)":
        sample_path = Path("data/sample_recon.mp4")
        if not sample_path.exists():
            sample_path = generate_synthetic_pov_video(sample_path, num_frames=120)
        video_path = sample_path
        st.success("Loaded Corridor Recon video.")
    else:
        sample_curved = Path("data/sample_recon_curved.mp4")
        if not sample_curved.exists():
            sample_curved = Path("data/sample_recon.mp4")
        video_path = sample_curved
        st.success("Loaded Cavern Winding Recon video.")

    st.markdown("---")
    st.subheader("⏱ Playback & Processing Parameters")
    frame_step = st.slider("Frame Downsampling Step", min_value=1, max_value=8, value=2, help="Process every Nth frame for real-time speed.")
    max_frames = st.slider("Maximum Frames Limit", min_value=20, max_value=400, value=140, step=10, help="Prevents processing lag.")
    min_inliers = st.slider("RANSAC Min Inliers", min_value=10, max_value=60, value=20, step=2)

    st.markdown("---")
    with st.expander("🛠 Advanced SLAM & Tracker Tuning"):
        orb_features = st.number_input("ORB Max Features", min_value=200, max_value=3000, value=1200, step=100)
        ratio_thresh = st.slider("Lowe's Ratio Threshold", min_value=0.5, max_value=0.9, value=0.75, step=0.05)
        kf_dist_thresh = st.slider("Keyframe Distance Threshold", min_value=0.5, max_value=3.0, value=1.0, step=0.2)

    st.markdown("---")
    st.caption("ResQNav Simulation Framework • GPS-Denied Robotics")


# ------------------------------------------------------------------------------
# STATUS PANEL (Top of Dashboard)
# ------------------------------------------------------------------------------
m = st.session_state.last_tracking_metrics
nav = st.session_state.breadcrumb_nav
slam = st.session_state.slam_engine
tracker = st.session_state.landmark_tracker

frame_no = m.frame_idx if m else 0
feat_cnt = m.total_features if m else 0
inliers_cnt = m.inliers if m else 0
inlier_pct = f"{m.inlier_ratio:.0%}" if m else "0%"
q_score = f"{m.quality_score:.0f}%" if m else "100%"
q_grade = m.quality_grade if m else "STANDBY"
cur_x, cur_z, cur_yaw = nav.current_pose
status_str = "PAUSED" if nav.is_paused else (m.status if m else "READY")
wp_count = len(nav.waypoints)
loop_events = len(slam.loop_edges)

st.markdown('<div class="status-panel-card">', unsafe_allow_html=True)
s1, s2, s3, s4, s5, s6, s7, s8 = st.columns(8)
s1.metric("Frame Number", f"#{frame_no:04d}")
s2.metric("Features Detected", feat_cnt)
s3.metric("Match Inliers", f"{inliers_cnt} ({inlier_pct})")
s4.metric("Tracking Quality", f"{q_grade} ({q_score})")
s5.metric("Relative Pose", f"({cur_x:+.2f}, {cur_z:+.2f}) rel")
s6.metric("Heading Yaw", f"{cur_yaw:+.1f}°")
s7.metric("Waypoints / Tracks", f"{wp_count} / {len(tracker.active_tracks)}")
s8.metric("Loop Closures", f"{loop_events} Verified")
st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MISSION CONTROL: 3-PANEL TACTICAL DASHBOARD
# ------------------------------------------------------------------------------
if app_mode == "Mission Control (Panels A, B, C)":
    # Video Ingestion & Upload Hub (Directly accessible on Main Screen)
    with st.expander("📂 Video Source & File Upload Hub (Click to Upload or Select Video)", expanded=True):
        up_c1, up_c2 = st.columns([3, 2])
        with up_c1:
            main_uploaded_file = st.file_uploader(
                "Upload Your Robot-POV Video File (.mp4, .avi, .mov, .mkv, .webm):",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Select any prerecorded video from your computer to run Landmark Tracking, Breadcrumb Nav & SLAM.",
                key="main_page_video_uploader"
            )
            if main_uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(main_uploaded_file.name).suffix)
                tfile.write(main_uploaded_file.read())
                tfile.close()
                video_path = Path(tfile.name)
                st.session_state.active_video_name = main_uploaded_file.name
                st.success(f"✔ Successfully loaded custom video: '{main_uploaded_file.name}'")

        with up_c2:
            st.markdown("**Or Quick-Select Built-in Test Scenarios:**")
            q1, q2 = st.columns(2)
            if q1.button("🔄 Closed-Loop Arena", use_container_width=True, help="Full 360 loop with loop closure"):
                video_path = Path("data/sample_recon_loop.mp4")
                st.session_state.active_video_name = "Closed-Loop Recon Arena"
                st.rerun()
            if q2.button("🏛 Corridor Rubble", use_container_width=True, help="Straight disaster corridor"):
                video_path = Path("data/sample_recon.mp4")
                st.session_state.active_video_name = "Corridor Rubble Recon"
                st.rerun()
            if st.button("⛰ Cavern Winding (S-Curve)", use_container_width=True, help="Winding avoidance path"):
                video_path = Path("data/sample_recon_curved.mp4")
                st.session_state.active_video_name = "Cavern Winding Recon"
                st.rerun()

        active_title = st.session_state.get("active_video_name", video_path.name if video_path else "No Video Selected")
        st.markdown(f'<div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:6px; padding:8px 14px; margin-top:8px; color:#0f172a;">📹 <b>Active Video Source:</b> <span style="color:#0284c7; font-weight:bold;">{active_title}</span></div>', unsafe_allow_html=True)

    # Primary Execution Strip
    col_run_btn, col_pause_btn, col_resume_btn, col_reset_btn = st.columns([3, 1, 1, 1])
    
    with col_run_btn:
        run_mission_btn = st.button("▶ Run Full Mission: Landmark Tracking, Breadcrumbs & SLAM", type="primary", use_container_width=True)
    with col_pause_btn:
        if st.button("⏸ Pause", use_container_width=True):
            nav.pause()
            st.session_state.mission_paused = True
    with col_resume_btn:
        if st.button("▶ Resume", use_container_width=True):
            nav.resume()
            st.session_state.mission_paused = False
    with col_reset_btn:
        if st.button("🔄 Reset / Replay", use_container_width=True):
            nav.reset()
            tracker.reset()
            slam.reset()
            st.session_state.last_tracking_metrics = None
            st.session_state.latest_annotated_frame = None
            st.session_state.mission_completed = False
            st.rerun()

    # Three-Panel Live Layout
    panel_a_col, panel_b_col, panel_c_col = st.columns(3)

    with panel_a_col:
        st.markdown('<div class="panel-header">PANEL A: LIVE VIDEO & LANDMARK TRACKING</div>', unsafe_allow_html=True)
        video_placeholder = st.empty()
        landmark_summary_placeholder = st.empty()

    with panel_b_col:
        st.markdown('<div class="panel-header">PANEL B: BREADCRUMB MAP & WAYPOINTS</div>', unsafe_allow_html=True)
        map_placeholder = st.empty()
        waypoint_ctrl_placeholder = st.empty()

    with panel_c_col:
        st.markdown('<div class="panel-header">PANEL C: VISUAL SLAM & LOOP CLOSURE</div>', unsafe_allow_html=True)
        slam_placeholder = st.empty()
        slam_info_placeholder = st.empty()

    # Execution Loop when user clicks Run
    if run_mission_btn:
        if video_path is None or not video_path.exists():
            st.error("Please select or upload a valid video file first.")
        else:
            try:
                reader = VideoReader(video_path)
                meta = reader.get_metadata()

                tracker.reset()
                nav.reset()
                slam.reset()

                # Configure tracker parameters
                tracker.num_features = int(orb_features)
                tracker.match_ratio = float(ratio_thresh)
                tracker.min_inliers_for_motion = int(min_inliers)

                # Initialize live occupancy grid
                grid = OccupancyGrid(rows=32, cols=42, cell_size_m=0.5, default_state=OccupancyGrid.CELL_UNKNOWN)
                start_pt = (26, 21)
                origin_x, origin_y = grid.cell_to_metric(start_pt[0], start_pt[1])
                grid.set_cell(start_pt[0], start_pt[1], OccupancyGrid.CELL_FREE, hit_count=5)

                frames_processed = 0
                total_to_process = min(max_frames, meta["frame_count"] // frame_step)
                progress_bar = st.progress(0.0)

                for frame_idx, frame in reader.iterate_frames(step=frame_step, max_frames=max_frames):
                    frames_processed += 1

                    # 1. Visual Landmark Tracking (Features, Track IDs, Surveyed check)
                    annotated_frame, metrics = tracker.process_frame(frame, frame_idx=frame_idx)
                    st.session_state.last_tracking_metrics = metrics
                    st.session_state.latest_annotated_frame = annotated_frame

                    # 2. Breadcrumb Navigation (Pose integration & auto-breadcrumbs)
                    cur_pose = nav.update_pose_step(
                        dx_rel=metrics.rel_dx,
                        dz_rel=metrics.rel_dz,
                        dyaw_deg=metrics.rel_dyaw_deg,
                        frame_idx=frame_idx,
                        tracking_ok=metrics.tracking_ok
                    )

                    # 3. Visual SLAM & Loop Closure
                    kf_added, kf, loop_cand = slam.consider_keyframe(
                        frame_bgr=frame,
                        frame_idx=frame_idx,
                        cur_pose=cur_pose
                    )

                    # 4. Perception & Occupancy Grid Mapping
                    free_mask, obs_mask, _ = st.session_state.perception_engine.analyze_frame(frame)
                    rover_x = origin_x + cur_pose[0]
                    rover_y = origin_y + cur_pose[1]
                    grid.update_from_frame_perception(rover_x, rover_y, cur_pose[2], free_mask, obs_mask)

                    # Update UI previews every 2 frames
                    if frames_processed % 2 == 0 or frames_processed == total_to_process:
                        # Update Panel A: Video Playback
                        video_placeholder.image(
                            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx:04d} | Cyan=Auto Track IDs, Gold=Surveyed Ground Datum",
                            use_container_width=True
                        )

                        # Update Panel B: 2D Breadcrumb Map
                        bmap_fig = nav.plot_breadcrumb_map(title="Breadcrumb Navigation Trajectory")
                        map_placeholder.pyplot(bmap_fig, clear_figure=True)
                        plt.close(bmap_fig)

                        # Update Panel C: Visual SLAM Before vs After
                        slam_fig = slam.plot_trajectory_comparison(title="Visual SLAM: Trajectory & Loop Closures")
                        slam_placeholder.pyplot(slam_fig, clear_figure=True)
                        plt.close(slam_fig)

                    progress_bar.progress(min(1.0, frames_processed / max(1, total_to_process)))

                reader.release()
                st.session_state.video_grid = grid
                st.session_state.mission_completed = True
                progress_bar.empty()
                st.success(f"✔ Full mission complete! Processed {frames_processed} frames. Waypoints saved and SLAM graph optimized.")

            except VideoError as e:
                st.error(f"Video Reader Error: {e}")

    # Render persisted panels when not actively streaming
    else:
        # Panel A Persisted View
        with panel_a_col:
            if st.session_state.latest_annotated_frame is not None:
                st.image(
                    cv2.cvtColor(st.session_state.latest_annotated_frame, cv2.COLOR_BGR2RGB),
                    caption="Latest Frame Tracking Overlay (Cyan: Auto Tracks, Gold: Surveyed Datum)",
                    use_container_width=True
                )
            else:
                st.info("Click '▶ Run Full Mission' above to process video frames and track persistent visual landmarks.")

            # Distinct Landmark Registry Summary
            l_sum = tracker.get_landmark_summary()
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:12px; margin-top:10px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <span style="font-weight:bold; color:#0369a1; font-size:12px;">VISUAL LANDMARK REGISTRY SUMMARY</span><br>
                <div style="margin: 6px 0;">
                    <span class="pill-badge pill-auto">Auto-Tracked Features: {l_sum['total_landmarks_seen']} Total ({l_sum['currently_active_tracks']} Active)</span>
                    <span class="pill-badge pill-survey">Surveyed Ground Benchmarks: {l_sum['surveyed_landmarks_registered']}</span>
                </div>
                <p style="color:#475569; font-size:11px; margin-top:6px; margin-bottom:0; line-height: 1.4;">
                    <b>Scientific Notice:</b> Automatically tracked features are unanchored pixel points with no global datum; 
                    Surveyed landmarks have known real-world physical/GPS coordinates.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Panel B Persisted View
        with panel_b_col:
            bmap_fig = nav.plot_breadcrumb_map(title="Estimated Breadcrumb Trajectory")
            st.pyplot(bmap_fig, clear_figure=True)
            plt.close(bmap_fig)

            st.markdown("""
            <div class="disclaimer-box">
                <b>Scale Disclaimer:</b> Monocular trajectory is plotted in <b>relative/arbitrary scale</b>. 
                Video processing does not provide exact real-world coordinates without external metric sensor fusion.
            </div>
            """, unsafe_allow_html=True)

            # Waypoint Persistence Controls
            wp_c1, wp_c2 = st.columns(2)
            with wp_c1:
                manual_label = st.text_input("Waypoint Label", value="POI-SURVIVOR-ZONE")
                if st.button("📍 Drop Manual Waypoint", use_container_width=True):
                    wp_new = nav.add_manual_waypoint(label=manual_label)
                    st.success(f"Dropped Waypoint #{wp_new.id}: {wp_new.label}")
                    st.rerun()

            with wp_c2:
                # Save to CSV
                csv_path = Path("data/waypoints.csv")
                nav.save_to_csv(str(csv_path))
                if csv_path.exists():
                    st.download_button(
                        label="💾 Download Waypoints (CSV)",
                        data=csv_path.read_text(encoding="utf-8"),
                        file_name="waypoints.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # Save to SQLite
                db_path = Path("data/waypoints.db")
                if st.button("🗄 Save to SQLite DB", use_container_width=True):
                    nav.save_to_sqlite(str(db_path))
                    st.success(f"Saved {len(nav.waypoints)} waypoints to '{db_path.name}'.")

        # Panel C Persisted View
        with panel_c_col:
            slam_fig = slam.plot_trajectory_comparison(title="Visual SLAM: Trajectory Before vs After Optimization")
            st.pyplot(slam_fig, clear_figure=True)
            plt.close(slam_fig)

            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:12px; margin-top:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <span style="font-weight:bold; color:#0369a1; font-size:12px;">POSE-GRAPH SLAM & LOOP CLOSURE STATUS</span><br>
                <div style="font-size:12px; color:#1e293b; margin: 6px 0;">
                    • <b>Keyframes in Graph:</b> {len(slam.keyframes)}<br>
                    • <b>Evaluated Candidates:</b> {len(slam.candidates_history)}<br>
                    • <b>Geometric Verified Loops:</b> {len(slam.loop_edges)} (Green dashed edges)
                </div>
                <p style="color:#475569; font-size:11px; margin-top:6px; margin-bottom:0; line-height: 1.4;">
                    <b>Algorithm:</b> SE(2) Pose-Graph solved via SciPy Huber least-squares optimization. 
                    Candidate loops are verified with RANSAC Essential Matrix (never accepted on appearance alone).
                </p>
            </div>
            """, unsafe_allow_html=True)

            if len(slam.loop_edges) > 0:
                if st.button("⚡ Re-Optimize Pose Graph", use_container_width=True):
                    slam.optimize_pose_graph()
                    st.success("Re-computed SE(2) Pose-Graph optimization.")
                    st.rerun()

    # Bottom Inspection Tables: Waypoints Table & Loop Closure Audit Log
    st.markdown("---")
    t_c1, t_c2 = st.columns(2)

    with t_c1:
        st.subheader("📋 Breadcrumb Waypoints Database")
        if nav.waypoints:
            wp_data = [
                {
                    "ID": wp.id,
                    "Timestamp": wp.timestamp_iso[-12:-4],
                    "Frame": wp.frame_idx,
                    "X (Rel)": f"{wp.x_rel:+.2f}",
                    "Z (Rel)": f"{wp.z_rel:+.2f}",
                    "Yaw": f"{wp.yaw_deg:+.1f}°",
                    "Type": "Manual" if wp.is_manual else "Auto-Drop",
                    "Label": wp.label
                }
                for wp in nav.waypoints
            ]
            st.dataframe(pd.DataFrame(wp_data), use_container_width=True, height=220)
        else:
            st.caption("No waypoints recorded yet.")

    with t_c2:
        st.subheader("🔍 Visual SLAM Loop Closure Audit Log")
        if slam.candidates_history:
            cand_data = [
                {
                    "Query KF": f"#{c.query_kf_id} (F{c.query_frame_idx})",
                    "Matched KF": f"#{c.match_kf_id} (F{c.match_frame_idx})",
                    "Matches": c.num_matches,
                    "RANSAC Inliers": f"{c.inliers} ({c.inlier_ratio:.0%})",
                    "Decision": "✔ ACCEPTED" if c.accepted else "✖ REJECTED",
                    "Reason": c.status_message
                }
                for c in slam.candidates_history[-10:]
            ]
            st.dataframe(pd.DataFrame(cand_data), use_container_width=True, height=220)
        else:
            st.caption("No loop closure candidates evaluated yet.")


# ------------------------------------------------------------------------------
# OCCUPANCY GRID MAPPING & A* PATH PLANNING STUDIO (Preserved Mode)
# ------------------------------------------------------------------------------
elif app_mode == "Occupancy Grid & A* Planning Studio":
    st.markdown('<div class="panel-header">OCCUPANCY GRID MAPPING & HEAP-BASED A* PATH PLANNING</div>', unsafe_allow_html=True)

    col_map_settings, col_plan_settings = st.columns(2)

    with col_map_settings:
        st.subheader("🗺 Occupancy Grid Selection")
        map_preset = st.selectbox(
            "Map Environment Preset",
            [
                "Video-Derived Dynamic Grid (From Loaded Video)",
                "Disaster Rubble Benchmark (Mode A)",
                "Warehouse Depot Shelves (Mode A)",
                "Subterranean Tunnel (Mode A)"
            ],
            index=0
        )

        if "Video-Derived" in map_preset and st.session_state.video_grid is not None:
            grid = st.session_state.video_grid
        elif "Rubble" in map_preset:
            grid = OccupancyGrid.create_disaster_rubble_map(rows=28, cols=36)
        elif "Warehouse" in map_preset:
            grid = OccupancyGrid.create_warehouse_rescue_map(rows=28, cols=36)
        elif "Tunnel" in map_preset:
            grid = OccupancyGrid.create_cave_tunnel_map(rows=28, cols=36)
        else:
            grid = OccupancyGrid(rows=32, cols=42, cell_size_m=0.5, default_state=OccupancyGrid.CELL_UNKNOWN, mode_label="Video-Derived Map")

        st.markdown(
            f'<div class="disclaimer-box"><b>{grid.mode_label}</b><br>'
            f'Grid Size: {grid.rows} rows × {grid.cols} cols ({grid.cell_size_m}m resolution). '
            f'Free cells: {np.sum(grid.grid == 0)}, Occupied cells: {np.sum(grid.grid == 1)}, Unknown cells: {np.sum(grid.grid == -1)}.</div>',
            unsafe_allow_html=True
        )

    with col_plan_settings:
        st.subheader("🎯 Start & Goal Waypoints")
        c_start, c_goal = st.columns(2)

        def_start = (26, 21) if "Video-Derived" in map_preset else ((25, 4) if "Rubble" in map_preset else (20, 3))
        def_goal = (8, 21) if "Video-Derived" in map_preset else ((4, 33) if "Rubble" in map_preset else (6, 32))

        with c_start:
            start_r = st.number_input("Start Row", min_value=0, max_value=grid.rows - 1, value=min(grid.rows - 1, def_start[0]))
            start_c = st.number_input("Start Col", min_value=0, max_value=grid.cols - 1, value=min(grid.cols - 1, def_start[1]))
            start_pt = (int(start_r), int(start_c))

        with c_goal:
            goal_r = st.number_input("Goal Row", min_value=0, max_value=grid.rows - 1, value=min(grid.rows - 1, def_goal[0]))
            goal_c = st.number_input("Goal Col", min_value=0, max_value=grid.cols - 1, value=min(grid.cols - 1, def_goal[1]))
            goal_pt = (int(goal_r), int(goal_c))

        allow_diag = st.checkbox("Allow Diagonal (8-way)", value=True)
        prevent_corner = st.checkbox("Prevent Corner-Cutting", value=True)
        allow_unknown = st.checkbox("Allow Unknown Cells", value=True)
        heuristic = st.selectbox("Heuristic Metric", ["octile", "euclidean", "manhattan"], index=0)

    # Execute A* Search
    path_result = astar_search(
        grid=grid,
        start=start_pt,
        goal=goal_pt,
        allow_diagonal=allow_diag,
        prevent_corner_cutting=prevent_corner,
        allow_unknown=allow_unknown,
        heuristic_type=heuristic
    )

    r1, r2, r3, r4, r5 = st.columns(5)
    status_label = "PATH DISCOVERED" if path_result.found else "NO PATH FOUND"
    r1.metric("Search Status", status_label)
    r2.metric("Path Length", f"{path_result.path_length_cells} cells")
    r3.metric("Path Cost", f"{path_result.path_cost:.2f}")
    r4.metric("Explored Nodes", f"{path_result.explored_nodes_count}")
    r5.metric("Solver Compute Time", f"{path_result.execution_time_ms:.2f} ms")

    grid_fig = plot_occupancy_grid(
        grid=grid,
        start=start_pt,
        goal=goal_pt,
        path=path_result.path if path_result.found else None,
        explored_nodes=path_result.explored_nodes,
        title=f"{grid.mode_label} • A* Route Overlay"
    )
    st.pyplot(grid_fig, clear_figure=True)
    plt.close(grid_fig)


# ------------------------------------------------------------------------------
# VISUAL ODOMETRY & SENSOR DIAGNOSTICS
# ------------------------------------------------------------------------------
else:
    st.markdown('<div class="panel-header">VISUAL ODOMETRY & SCIENTIFIC RIGOR DIAGNOSTICS</div>', unsafe_allow_html=True)

    with st.expander("🔬 Scientific Rigor: Monocular Visual Odometry & Scale Ambiguity", expanded=True):
        st.markdown(r"""
        #### Why Monocular Visual Odometry Has Scale Ambiguity
        1. **Projective Geometry**: When projecting a 3D scene onto a 2D image plane, the projection equations:
           $$u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y$$
           are invariant under arbitrary scaling by a scalar $\lambda > 0$: $\lambda (X, Y, Z)$ projects to the exact same image coordinates $(u, v)$.
        2. **Essential Matrix Decomposition**: Factoring the Essential Matrix $E = [t]_\times R$ yields a translation vector $t$ whose magnitude is normalized to $\|t\| = 1$. Without an external metric baseline, translation distance between two frames cannot be determined in metres.
        3. **Honest Reporting in ResQNav**: All trajectory plots in this software are deliberately labeled in **arbitrary relative units**. We do not present unscaled monocular translation as true physical metres.
        """)

    with st.expander("🚀 Physical Rover Hardware Integration Roadmap (ESP32 / Motors)", expanded=True):
        st.markdown("""
        #### Path to Physical Rover Integration
        - **Step 1: Metric Scale Observability**: Integrate wheel encoder feedback or IMU accelerometer data with an Extended Kalman Filter (EKF) to observe true metric scale and fuse with VO orientation.
        - **Step 2: Depth Sensing for Autonomous Mapping**: Mount a low-cost stereo camera or 2D LiDAR to directly observe obstacle ranges, replacing the simulated Mode A demonstration maps with metric point-cloud occupancy grids.
        - **Step 3: Real-Time Stream Ingestion**: Replace file-based VideoReader with the ESP32-CAM MJPEG HTTP stream endpoint (`http://192.168.4.1:81/stream`), which is natively supported by the OpenCV `VideoCapture` pipeline.
        - **Step 4: Path Execution & Closed-Loop Control**: Transmit A* waypoint heading errors to the ESP32 PWM motor bridge using the REST action commands (`/action?go=forward&speed=55`).
        """)
