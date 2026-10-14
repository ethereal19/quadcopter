/**
 * ResQNav // Standalone Light-Themed Mission Control Client Logic
 * Handles real-time polling, video playback, HTML5 Canvas rendering,
 * drag-and-drop file upload, and navigation telemetry.
 */

// Global state cache
let stateCache = null;
let pollTimer = null;
let isUploading = false;

// DOM Elements
const annotatedVideoImg = document.getElementById("annotatedVideoImg");
const dropZone = document.getElementById("dropZone");
const videoFileInput = document.getElementById("videoFileInput");
const browseLink = document.getElementById("browseLink");
const uploadStatusText = document.getElementById("uploadStatusText");

// Playback controls
const btnPlayPause = document.getElementById("btnPlayPause");
const playIcon = document.getElementById("playIcon");
const playText = document.getElementById("playText");
const btnStep = document.getElementById("btnStep");
const btnReset = document.getElementById("btnReset");
const speedSelect = document.getElementById("speedSelect");
const frameScrubber = document.getElementById("frameScrubber");

// Metadata Readouts
const metaVideoName = document.getElementById("metaVideoName");
const metaResolution = document.getElementById("metaResolution");
const metaFps = document.getElementById("metaFps");
const metaTotalFrames = document.getElementById("metaTotalFrames");

// Ticker Readouts
const tickerFrameNum = document.getElementById("tickerFrameNum");
const tickerQuality = document.getElementById("tickerQuality");
const tickerPose = document.getElementById("tickerPose");
const tickerYaw = document.getElementById("tickerYaw");
const systemStatusText = document.getElementById("systemStatusText");

// Metric Cards Readouts
const metricQuality = document.getElementById("metricQuality");
const qualityBar = document.getElementById("qualityBar");
const gradeBadge = document.getElementById("gradeBadge");
const subInliers = document.getElementById("subInliers");
const subInlierRatio = document.getElementById("subInlierRatio");

const metricPoseZ = document.getElementById("metricPoseZ");
const subPoseX = document.getElementById("subPoseX");
const subPoseY = document.getElementById("subPoseY");
const subPoseYaw = document.getElementById("subPoseYaw");

const metricPersistentTracks = document.getElementById("metricPersistentTracks");
const subDetectedFeatures = document.getElementById("subDetectedFeatures");
const subMatchedFeatures = document.getElementById("subMatchedFeatures");
const subSurveyVisible = document.getElementById("subSurveyVisible");

const metricLoops = document.getElementById("metricLoops");
const subKeyframes = document.getElementById("subKeyframes");
const subLoopEdges = document.getElementById("subLoopEdges");

// Video HUD Overlays
const hudStatusPill = document.getElementById("hudStatusPill");
const hudFeatureCount = document.getElementById("hudFeatureCount");
const hudFrameFps = document.getElementById("hudFrameFps");
const hudPoseOverlay = document.getElementById("hudPoseOverlay");

// Canvases
const breadcrumbCanvas = document.getElementById("breadcrumbCanvas");
const slamCanvas = document.getElementById("slamCanvas");
const gridCanvas = document.getElementById("gridCanvas");

// Waypoints & Tables
const waypointTableBody = document.getElementById("waypointTableBody");
const loopTableBody = document.getElementById("loopTableBody");
const loopCountNotice = document.getElementById("loopCountNotice");
const btnAddManualWp = document.getElementById("btnAddManualWp");
const btnDownloadCsv = document.getElementById("btnDownloadCsv");
const btnSaveSqlite = document.getElementById("btnSaveSqlite");

// A* Controls
const startRow = document.getElementById("startRow");
const startCol = document.getElementById("startCol");
const goalRow = document.getElementById("goalRow");
const goalCol = document.getElementById("goalCol");
const btnReplanPath = document.getElementById("btnReplanPath");
const astarStatusBadge = document.getElementById("astarStatusBadge");
const pathLengthVal = document.getElementById("pathLengthVal");

// Presets
const presetButtons = document.querySelectorAll(".preset-btn");


// ------------------------------------------------------------------------------
// INITIALIZATION & EVENT LISTENERS
// ------------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  fetchState();
  startPolling();
});

function setupEventListeners() {
  // Play / Pause
  btnPlayPause.addEventListener("click", () => sendControl("toggle"));
  btnStep.addEventListener("click", () => sendControl("step"));
  btnReset.addEventListener("click", () => sendControl("reset"));

  // Speed
  speedSelect.addEventListener("change", (e) => {
    sendControl("speed", parseFloat(e.target.value));
  });

  // Scrubber
  frameScrubber.addEventListener("change", (e) => {
    sendControl("seek", null, parseInt(e.target.value));
  });

  // Presets
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const filename = btn.getAttribute("data-file");
      loadPresetSample(filename);
    });
  });

  // Drag & Drop Upload
  browseLink.addEventListener("click", () => videoFileInput.click());
  videoFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  // Waypoints actions
  btnAddManualWp.addEventListener("click", () => sendControl("add_waypoint"));
  btnDownloadCsv.addEventListener("click", () => {
    window.location.href = "/api/export_csv";
  });
  btnSaveSqlite.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/save_sqlite", { method: "POST" });
      const data = await res.json();
      if (data.success) {
        alert(`Successfully saved ${data.count} waypoints to SQLite database:\n${data.db_path}`);
      }
    } catch (err) {
      alert("Failed to save to SQLite: " + err);
    }
  });

  // A* Re-plan
  btnReplanPath.addEventListener("click", replanPath);
}


// ------------------------------------------------------------------------------
// SERVER API CALLS & POLLING
// ------------------------------------------------------------------------------
async function fetchState() {
  try {
    const res = await fetch("/api/state");
    if (!res.ok) return;
    const state = await res.json();
    stateCache = state;
    updateUI(state);
  } catch (err) {
    console.warn("Polling error:", err);
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    await fetchState();
  }, 80); // ~12 FPS polling for smooth trajectory & video updates
}

async function sendControl(action, value = null, frame = null) {
  try {
    const payload = { action };
    if (value !== null) payload.value = value;
    if (frame !== null) payload.frame = frame;
    await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await fetchState();
  } catch (err) {
    console.error("Control error:", err);
  }
}

async function loadPresetSample(filename) {
  try {
    uploadStatusText.innerText = `Loading scenario: ${filename}...`;
    const res = await fetch("/api/load_sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
    const data = await res.json();
    if (data.success) {
      uploadStatusText.innerText = `Scenario loaded: ${filename}`;
      await fetchState();
    } else {
      uploadStatusText.innerText = `Error: ${data.error}`;
    }
  } catch (err) {
    uploadStatusText.innerText = `Failed to load sample: ${err}`;
  }
}

async function uploadFile(file) {
  if (!file) return;
  isUploading = true;
  uploadStatusText.innerText = `Uploading and ingesting ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)...`;

  const formData = new FormData();
  formData.append("video", file);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (data.success) {
      uploadStatusText.innerText = `Successfully loaded custom video: ${file.name}`;
      presetButtons.forEach((b) => b.classList.remove("active"));
      await fetchState();
    } else {
      uploadStatusText.innerText = `Upload failed: ${data.error}`;
    }
  } catch (err) {
    uploadStatusText.innerText = `Upload error: ${err}`;
  } finally {
    isUploading = false;
  }
}

async function replanPath() {
  try {
    const payload = {
      start_r: parseInt(startRow.value),
      start_c: parseInt(startCol.value),
      goal_r: parseInt(goalRow.value),
      goal_c: parseInt(goalCol.value),
    };
    const res = await fetch("/api/plan_path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.success) {
      astarStatusBadge.innerText = data.status;
      pathLengthVal.innerText = `${data.path_length} steps`;
      await fetchState();
    }
  } catch (err) {
    console.error("Replan error:", err);
  }
}


// ------------------------------------------------------------------------------
// UI UPDATE DISPATCHER
// ------------------------------------------------------------------------------
function updateUI(state) {
  if (!state) return;

  const v = state.video;
  const t = state.telemetry;
  const b = state.breadcrumbs;
  const s = state.slam;
  const g = state.grid;

  // 1. Playback Button & Scrubber
  if (v.is_playing) {
    playIcon.innerText = "⏸";
    playText.innerText = "Pause Playback";
    btnPlayPause.classList.remove("btn-primary");
    btnPlayPause.classList.add("btn-secondary");
  } else {
    playIcon.innerText = "▶";
    playText.innerText = "Play Trajectory";
    btnPlayPause.classList.add("btn-primary");
    btnPlayPause.classList.remove("btn-secondary");
  }

  frameScrubber.max = v.total_frames || 100;
  frameScrubber.value = v.frame_idx || 0;

  // Video metadata
  metaVideoName.innerText = v.name || "None";
  metaResolution.innerText = v.resolution || "--";
  metaFps.innerText = `${v.fps} FPS`;
  metaTotalFrames.innerText = v.total_frames || 0;

  // Top Telemetry Ticker
  tickerFrameNum.innerText = `${String(v.frame_idx).padStart(3, "0")} / ${v.total_frames}`;
  tickerQuality.innerText = `${t.quality_score}%`;
  tickerPose.innerText = `(${t.pose_x >= 0 ? "+" : ""}${t.pose_x.toFixed(2)}, ${t.pose_z >= 0 ? "+" : ""}${t.pose_z.toFixed(2)}) rel`;
  tickerYaw.innerText = `${t.yaw_deg.toFixed(1)}°`;

  // Metric Card 1: Quality
  metricQuality.innerText = `${t.quality_score.toFixed(1)}%`;
  qualityBar.style.width = `${Math.min(100, Math.max(0, t.quality_score))}%`;
  gradeBadge.innerText = t.quality_grade;
  if (t.quality_grade === "EXCELLENT" || t.quality_grade === "GOOD") {
    gradeBadge.className = "badge badge-emerald";
    qualityBar.style.background = "var(--color-emerald)";
  } else if (t.quality_grade === "DEGRADED") {
    gradeBadge.className = "badge badge-amber";
    qualityBar.style.background = "var(--color-amber)";
  } else {
    gradeBadge.className = "badge badge-crimson";
    qualityBar.style.background = "var(--color-crimson)";
  }
  subInliers.innerText = t.inlier_matches;
  subInlierRatio.innerText = `${t.inlier_ratio}%`;

  // Metric Card 2: Pose
  metricPoseZ.innerText = `${t.pose_z >= 0 ? "+" : ""}${t.pose_z.toFixed(2)} rel`;
  subPoseX.innerText = `${t.pose_x >= 0 ? "+" : ""}${t.pose_x.toFixed(2)}`;
  subPoseY.innerText = `${t.pose_y >= 0 ? "+" : ""}${t.pose_y.toFixed(2)}`;
  subPoseYaw.innerText = `${t.yaw_deg.toFixed(1)}°`;

  // Metric Card 3: Features & Survey
  metricPersistentTracks.innerText = t.persistent_tracks;
  subDetectedFeatures.innerText = t.detected_features;
  subMatchedFeatures.innerText = t.matched_features;
  subSurveyVisible.innerText = t.surveyed_visible;

  // Metric Card 4: SLAM
  metricLoops.innerText = s.total_loops;
  subKeyframes.innerText = s.keyframes.length;
  subLoopEdges.innerText = s.loop_edges.length;

  // Panel A: Annotated Image Feed
  if (state.image) {
    annotatedVideoImg.src = state.image;
  }
  hudStatusPill.innerText = `STATUS: ${t.is_tracking_ok ? "TRACKING_OK" : "DEGRADED"}`;
  hudStatusPill.style.color = t.is_tracking_ok ? "var(--color-emerald)" : "var(--color-crimson)";
  hudFeatureCount.innerText = `FEAT: ${t.detected_features} | INL: ${t.inlier_matches}`;
  hudFrameFps.innerText = `FRAME: ${String(v.frame_idx).padStart(3, "0")} | ${v.fps} FPS`;
  hudPoseOverlay.innerText = `X: ${t.pose_x.toFixed(2)} | Z: ${t.pose_z.toFixed(2)}`;

  // Panel B: Breadcrumb Trajectory Canvas
  drawBreadcrumbCanvas(b.trajectory, b.waypoints, t);

  // Update Waypoint Table
  updateWaypointTable(b.waypoints);

  // Panel C: Visual SLAM Trajectory Canvas
  drawSlamCanvas(b.trajectory, s.keyframes, s.loop_edges, s.optimized_poses);

  // Update Loop Closures Table
  updateLoopTable(s.loop_edges);

  // Panel D: Occupancy Grid & A* Path
  drawGridCanvas(g.matrix, g.start, g.goal, g.path);
  astarStatusBadge.innerText = g.status;
  pathLengthVal.innerText = `${g.path ? g.path.length : 0} steps`;
}


// ------------------------------------------------------------------------------
// CANVAS RENDERING ENGINES
// ------------------------------------------------------------------------------

/**
 * Panel B: 2D Breadcrumb Trajectory Canvas (HTML5 Canvas)
 */
function drawBreadcrumbCanvas(trajectory, waypoints, currentTelemetry) {
  const ctx = breadcrumbCanvas.getContext("2d");
  const w = breadcrumbCanvas.width;
  const h = breadcrumbCanvas.height;

  // Clear background (Clean Light Canvas)
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  // Draw Light Tactical Grid
  drawGridBackground(ctx, w, h);

  if (!trajectory || trajectory.length === 0) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Awaiting camera motion frames...", w / 2, h / 2);
    return;
  }

  // Calculate bounding box for auto-scaling
  let minX = -5, maxX = 5, minZ = -5, maxZ = 20;
  trajectory.forEach((pt) => {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.z < minZ) minZ = pt.z;
    if (pt.z > maxZ) maxZ = pt.z;
  });

  const spanX = Math.max(10, (maxX - minX) * 1.3);
  const spanZ = Math.max(10, (maxZ - minZ) * 1.3);
  const midX = (minX + maxX) / 2;
  const midZ = (minZ + maxZ) / 2;

  const scale = Math.min((w - 60) / spanX, (h - 60) / spanZ);

  function toScreen(x, z) {
    const sx = w / 2 + (x - midX) * scale;
    const sy = h - 30 - (z - minZ) * scale;
    return { x: sx, y: sy };
  }

  // Draw Trajectory Line
  ctx.beginPath();
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  trajectory.forEach((pt, i) => {
    const s = toScreen(pt.x, pt.z);
    if (i === 0) ctx.moveTo(s.x, s.y);
    else ctx.lineTo(s.x, s.y);
  });
  ctx.stroke();

  // Draw Start Origin (0, 0)
  const origin = toScreen(0, 0);
  ctx.beginPath();
  ctx.arc(origin.x, origin.y, 6, 0, Math.PI * 2);
  ctx.fillStyle = "#10b981";
  ctx.fill();
  ctx.strokeStyle = "#065f46";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = "#065f46";
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillText("Origin (0,0)", origin.x + 8, origin.y - 4);

  // Draw Numbered Waypoints
  if (waypoints && waypoints.length > 0) {
    waypoints.forEach((wp) => {
      const p = toScreen(wp.x, wp.z);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "#7c3aed";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = "#5b21b6";
      ctx.font = "8.5px 'JetBrains Mono', monospace";
      ctx.fillText(wp.id, p.x + 6, p.y + 3);
    });
  }

  // Draw Current Rover Pose & Heading Arrow
  const lastPt = trajectory[trajectory.length - 1];
  const cur = toScreen(lastPt.x, lastPt.z);

  ctx.save();
  ctx.translate(cur.x, cur.y);
  const yawRad = (currentTelemetry.yaw_deg * Math.PI) / 180.0;
  ctx.rotate(-yawRad);

  // Triangle marker
  ctx.beginPath();
  ctx.moveTo(0, -10);
  ctx.lineTo(7, 8);
  ctx.lineTo(-7, 8);
  ctx.closePath();
  ctx.fillStyle = "#f59e0b";
  ctx.fill();
  ctx.strokeStyle = "#78350f";
  ctx.lineWidth = 1.8;
  ctx.stroke();
  ctx.restore();
}

/**
 * Panel C: Visual SLAM Trajectory Comparison Canvas
 */
function drawSlamCanvas(driftedTrajectory, keyframes, loopEdges, optimizedPoses) {
  const ctx = slamCanvas.getContext("2d");
  const w = slamCanvas.width;
  const h = slamCanvas.height;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  drawGridBackground(ctx, w, h);

  if (!driftedTrajectory || driftedTrajectory.length === 0) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Visual SLAM graph awaiting keyframes...", w / 2, h / 2);
    return;
  }

  let minX = -5, maxX = 5, minZ = -5, maxZ = 20;
  driftedTrajectory.forEach((pt) => {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.z < minZ) minZ = pt.z;
    if (pt.z > maxZ) maxZ = pt.z;
  });

  const spanX = Math.max(10, (maxX - minX) * 1.3);
  const spanZ = Math.max(10, (maxZ - minZ) * 1.3);
  const midX = (minX + maxX) / 2;
  const scale = Math.min((w - 60) / spanX, (h - 60) / spanZ);

  function toScreen(x, z) {
    return {
      x: w / 2 + (x - midX) * scale,
      y: h - 30 - (z - minZ) * scale,
    };
  }

  // 1. Draw Drifted VO Trajectory (Orange Dashed)
  ctx.beginPath();
  ctx.strokeStyle = "#ea580c";
  ctx.lineWidth = 2.2;
  ctx.setLineDash([5, 4]);
  driftedTrajectory.forEach((pt, i) => {
    const s = toScreen(pt.x, pt.z);
    if (i === 0) ctx.moveTo(s.x, s.y);
    else ctx.lineTo(s.x, s.y);
  });
  ctx.stroke();
  ctx.setLineDash([]); // Reset dash

  // 2. Draw Optimized Pose Graph (Blue Solid)
  if (optimizedPoses && optimizedPoses.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = "#0284c7";
    ctx.lineWidth = 2.8;
    optimizedPoses.forEach((pt, i) => {
      const s = toScreen(pt.x, pt.z);
      if (i === 0) ctx.moveTo(s.x, s.y);
      else ctx.lineTo(s.x, s.y);
    });
    ctx.stroke();
  }

  // 3. Draw Loop Closure Edges (Green Lines)
  if (loopEdges && loopEdges.length > 0 && keyframes) {
    const kfMap = {};
    keyframes.forEach((k) => (kfMap[k.id] = k));

    loopEdges.forEach((edge) => {
      const kf1 = kfMap[edge.from_kf];
      const kf2 = kfMap[edge.to_kf];
      if (kf1 && kf2) {
        const s1 = toScreen(kf1.x, kf1.z);
        const s2 = toScreen(kf2.x, kf2.z);

        ctx.beginPath();
        ctx.strokeStyle = "#16a34a";
        ctx.lineWidth = 2.2;
        ctx.moveTo(s1.x, s1.y);
        ctx.lineTo(s2.x, s2.y);
        ctx.stroke();

        ctx.fillStyle = "#15803d";
        ctx.font = "8.5px 'JetBrains Mono', monospace";
        ctx.fillText(`LOOP (${edge.inliers} inl)`, (s1.x + s2.x) / 2, (s1.y + s2.y) / 2 - 4);
      }
    });
  }

  // Draw Keyframe Points
  if (keyframes) {
    keyframes.forEach((kf) => {
      const s = toScreen(kf.x, kf.z);
      ctx.beginPath();
      ctx.arc(s.x, s.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = "#0284c7";
      ctx.fill();
    });
  }
}

/**
 * Panel D: 2D Occupancy Grid Map & A* Path
 */
function drawGridCanvas(matrix, start, goal, astarPath) {
  const ctx = gridCanvas.getContext("2d");
  const w = gridCanvas.width;
  const h = gridCanvas.height;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  if (!matrix || matrix.length === 0) return;

  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellW = w / cols;
  const cellH = h / rows;

  // Render Grid Cells
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = matrix[r][c];
      if (val === 1) {
        ctx.fillStyle = "#1e293b"; // Occupied (Obstacle)
      } else if (val === 0) {
        ctx.fillStyle = "#f8fafc"; // Free
      } else {
        ctx.fillStyle = "#cbd5e1"; // Unknown
      }
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);

      ctx.strokeStyle = "rgba(226, 232, 240, 0.4)";
      ctx.strokeRect(c * cellW, r * cellH, cellW, cellH);
    }
  }

  // Draw A* Path
  if (astarPath && astarPath.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 3.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    astarPath.forEach((pt, i) => {
      const px = (pt[1] + 0.5) * cellW;
      const py = (pt[0] + 0.5) * cellH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Glow effect
    ctx.beginPath();
    ctx.strokeStyle = "rgba(37, 99, 235, 0.3)";
    ctx.lineWidth = 8;
    astarPath.forEach((pt, i) => {
      const px = (pt[1] + 0.5) * cellW;
      const py = (pt[0] + 0.5) * cellH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  // Draw Start Position (*)
  if (start) {
    const sx = (start[1] + 0.5) * cellW;
    const sy = (start[0] + 0.5) * cellH;
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#10b981";
    ctx.fill();
    ctx.strokeStyle = "#065f46";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#065f46";
    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.fillText("S", sx - 3, sy + 3);
  }

  // Draw Goal Position (X)
  if (goal) {
    const gx = (goal[1] + 0.5) * cellW;
    const gy = (goal[0] + 0.5) * cellH;
    ctx.beginPath();
    ctx.arc(gx, gy, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#ef4444";
    ctx.fill();
    ctx.strokeStyle = "#991b1b";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.fillText("G", gx - 3, gy + 3);
  }
}

/**
 * Helper to draw subtle grid lines on canvases
 */
function drawGridBackground(ctx, w, h) {
  ctx.strokeStyle = "#f1f5f9";
  ctx.lineWidth = 1;

  for (let x = 0; x < w; x += 30) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 30) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
}


// ------------------------------------------------------------------------------
// TABLE POPULATORS
// ------------------------------------------------------------------------------
function updateWaypointTable(waypoints) {
  if (!waypoints || waypoints.length === 0) {
    waypointTableBody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">No waypoints logged yet</td></tr>`;
    return;
  }

  // Show latest 15 waypoints in reverse order
  const display = [...waypoints].reverse().slice(0, 15);
  waypointTableBody.innerHTML = display
    .map(
      (wp) => `
    <tr>
      <td><b>${wp.id}</b></td>
      <td>${wp.x.toFixed(2)}</td>
      <td>${wp.z.toFixed(2)}</td>
      <td>${wp.yaw_deg.toFixed(1)}°</td>
      <td class="text-muted">${wp.timestamp.split("T")[1]?.slice(0, 8) || wp.timestamp}</td>
      <td><span class="badge ${wp.is_manual ? "badge-amber" : "badge-sky"}">${wp.is_manual ? "MANUAL" : "AUTO"}</span></td>
    </tr>
  `
    )
    .join("");
}

function updateLoopTable(loopEdges) {
  if (!loopEdges || loopEdges.length === 0) {
    loopCountNotice.innerText = "0 events verified";
    loopTableBody.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No loop closures detected yet. Run 360° Loop scenario.</td></tr>`;
    return;
  }

  loopCountNotice.innerText = `${loopEdges.length} verified loop events`;
  loopTableBody.innerHTML = loopEdges
    .map(
      (e) => `
    <tr>
      <td><b>${e.from_kf}</b></td>
      <td><b>${e.to_kf}</b></td>
      <td class="text-emerald"><b>${e.inliers} inliers</b></td>
      <td>${(e.confidence * 100).toFixed(1)}%</td>
      <td><span class="badge badge-emerald">GEOMETRIC RANSAC</span></td>
    </tr>
  `
    )
    .join("");
}
