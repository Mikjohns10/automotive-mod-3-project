/* ============================================================
   DriveGuard AI — Application Logic
   ============================================================ */
'use strict';

// ── Global State ──────────────────────────────────────────────
const state = {
  riskScore: 28,
  speed: 72,
  accel: 0.8,
  braking: 0.2,
  steering: 8,
  laneDev: 0.12,
  ear: 0.82,       // Eye Aspect Ratio
  headYaw: 2.1,
  headPitch: -1.3,
  phoneDetected: false,
  currentClass: 'normal',
  sessionStart: Date.now(),
  alertCount: 3,
  alerts: [],
  history: {
    risk:     Array.from({length: 60}, (_,i) => 15 + Math.sin(i*0.18)*10 + Math.random()*5),
    speed:    Array.from({length: 60}, (_,i) => 65 + Math.sin(i*0.12)*12 + Math.random()*6),
    accel:    Array.from({length: 60}, (_,i) => 0.5 + Math.sin(i*0.3)*0.4 + Math.random()*0.2),
    braking:  Array.from({length: 60}, (_,i) => 0.1 + Math.abs(Math.sin(i*0.5))*0.3),
    eye:      Array.from({length: 60}, () => 0.7 + Math.random()*0.25),
    steering: Array.from({length: 60}, (_,i) => 5 + Math.sin(i*0.2)*8),
    lane:     Array.from({length: 60}, (_,i) => 0.05 + Math.abs(Math.sin(i*0.15))*0.2),
    head:     Array.from({length: 60}, (_,i) => Math.sin(i*0.1)*5),
  },
  sparkData: {
    speed: [], accel: [], braking: [], steering: [],
    lane: [], eye: [], head: [], phone: [],
  },
  incidents: [
    { time: '00:02:14', event: 'Sudden Braking',       severity: 'high',   dur: '1.2s', riskDelta: '+18' },
    { time: '00:04:31', event: 'Lane Deviation',        severity: 'medium', dur: '3.4s', riskDelta: '+8'  },
    { time: '00:05:47', event: 'Phone Usage Detected',  severity: 'high',   dur: '6.1s', riskDelta: '+24' },
    { time: '00:07:02', event: 'Aggressive Accel.',     severity: 'medium', dur: '2.8s', riskDelta: '+12' },
    { time: '00:08:15', event: 'Drowsiness Detected',   severity: 'high',   dur: '4.5s', riskDelta: '+20' },
    { time: '00:09:33', event: 'Head Turned Away',      severity: 'medium', dur: '2.1s', riskDelta: '+9'  },
    { time: '00:10:48', event: 'Lane Deviation',        severity: 'low',    dur: '1.8s', riskDelta: '+5'  },
  ],
  alertFeed: [
    { type: 'critical', icon: '😴', title: 'Drowsiness Detected',    desc: 'EAR dropped below 0.25 for 4+ seconds',     time: '00:08:15' },
    { type: 'critical', icon: '📱', title: 'Phone Usage Detected',    desc: 'Device detected near driver face region',   time: '00:05:47' },
    { type: 'critical', icon: '🛑', title: 'Sudden Braking Event',    desc: 'Deceleration spike: 6.2 m/s²',             time: '00:02:14' },
    { type: 'warning',  icon: '🚗', title: 'Aggressive Acceleration', desc: 'Accel exceeded 5.0 m/s² threshold',        time: '00:07:02' },
    { type: 'warning',  icon: '🛣️', title: 'Lane Deviation Warning',  desc: 'Vehicle drifted 0.38m from center',        time: '00:04:31' },
    { type: 'info',     icon: '📡', title: 'Sensor Calibration Done', desc: 'All 8 parameters recalibrated successfully', time: '00:01:00' },
  ],
};

// ── Utility ───────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const clamp = (v, min, max) => Math.min(Math.max(v, min), max);
const lerp = (a, b, t) => a + (b - a) * t;
const rand = (min, max) => min + Math.random() * (max - min);

// ── Server Connection ─────────────────────────────────────────
let socket = null;
let serverConnected = false;

function initSocketIO() {
  // Try to connect to the Python backend
  const serverUrl = window.location.protocol === 'file:'
    ? 'http://localhost:5000'
    : window.location.origin;

  try {
    socket = io(serverUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 2000,
      timeout: 3000,
    });

    socket.on('connect', () => {
      serverConnected = true;
      console.log('✓ Connected to DriveGuard AI backend');
      updateServerStatus(true);
    });

    socket.on('disconnect', () => {
      serverConnected = false;
      console.log('⚠ Disconnected from backend — standalone mode');
      updateServerStatus(false);
    });

    socket.on('connect_error', () => {
      serverConnected = false;
      updateServerStatus(false);
    });

    // Receive predictions from the server's simulation thread
    socket.on('sensor_update', (data) => {
      if (!serverConnected) return;
      applyServerPrediction(data);
    });

    socket.on('status', (data) => {
      console.log('Server:', data.message, '| Model loaded:', data.model_loaded);
    });

  } catch (e) {
    console.log('Socket.IO not available — running standalone');
    serverConnected = false;
    updateServerStatus(false);
  }
}

function updateServerStatus(connected) {
  const indicator = $('server-indicator');
  const dot = $('server-dot');
  const label = $('server-label');

  if (connected) {
    indicator.classList.add('connected');
    dot.classList.remove('disconnected');
    dot.classList.add('connected');
    label.textContent = 'SERVER';
  } else {
    indicator.classList.remove('connected');
    dot.classList.add('disconnected');
    dot.classList.remove('connected');
    label.textContent = 'STANDALONE';
    
    // When disconnected, always fallback to SVG simulation
    const realCam = $('real-camera-feed');
    const svgOverlay = $('face-overlay');
    if (realCam) realCam.style.display = 'none';
    if (svgOverlay) svgOverlay.style.display = 'block';
  }
}

function applyServerPrediction(data) {
  // Update state from server prediction result
  const params = data.parameters || {};
  state.speed     = params.speed || state.speed;
  state.accel     = params.acceleration || state.accel;
  state.braking   = params.braking || state.braking;
  state.steering  = params.steering_angle || state.steering;
  state.laneDev   = params.lane_deviation || state.laneDev;
  state.ear       = params.eye_aspect_ratio || state.ear;
  state.headYaw   = params.head_yaw || state.headYaw;
  state.phoneDetected = (params.phone_usage || 0) > 0.5;

  // Render compatibility: only show MJPEG feed if backend confirms camera is active
  const realCam = $('real-camera-feed');
  const svgOverlay = $('face-overlay');
  if (data.camera_active) {
    if (realCam) realCam.style.display = 'block';
    if (svgOverlay) svgOverlay.style.display = 'none';
  } else {
    if (realCam) realCam.style.display = 'none';
    if (svgOverlay) svgOverlay.style.display = 'block';
  }

  state.riskScore = data.risk_score || state.riskScore;

  // Map server class names to frontend keys
  const classMap = {
    'Normal': 'normal',
    'Aggressive': 'aggressive',
    'Distracted': 'distracted',
    'Drowsy': 'drowsy',
    'Unsafe Braking': 'unsafe-braking',
    'Unsafe Acceleration': 'unsafe-accel',
  };

  const cls = classMap[data.predicted_class] || 'normal';
  const label = data.predicted_class || 'Normal';

  if (cls !== state.currentClass) {
    state.currentClass = cls;
    appendTimeline(cls, label);
    if (cls !== 'normal') addAlert('warning', '⚠️', `${label} Detected`, `Risk: ${Math.round(data.risk_score)}`);
  }

  // Update class probabilities from server
  if (data.all_probabilities) {
    const confMap = {
      'Normal': 'conf-normal',
      'Aggressive': 'conf-aggressive',
      'Distracted': 'conf-distracted',
      'Drowsy': 'conf-drowsy',
      'Unsafe Braking': 'conf-braking',
      'Unsafe Acceleration': 'conf-accel',
    };
    for (const [clsName, confId] of Object.entries(confMap)) {
      const prob = data.all_probabilities[clsName];
      if (prob !== undefined) {
        const el = $(confId);
        if (el) el.textContent = (prob * 100).toFixed(1) + '%';
        const bar = document.querySelector(`.behavior-class[data-class="${classMap[clsName]}"] .class-bar`);
        if (bar) bar.style.width = (prob * 100).toFixed(1) + '%';
      }
    }
    // Highlight active class
    const allClassEls = document.querySelectorAll('.behavior-class');
    allClassEls.forEach(el => {
      el.classList.toggle('active-class', el.dataset.class === cls);
    });
  }

  // Handle server alerts
  if (data.alerts && data.alerts.length > 0) {
    data.alerts.forEach(a => {
      showToast(a.title, a.message);
    });
  }

  // Push data to history/sparklines/charts
  pushDataToBuffers();
  updateUI(cls);
}

function pushDataToBuffers() {
  // Sparkline data
  Object.keys(state.sparkData).forEach(k => {
    const normMap = {
      speed: state.speed/140,
      accel: state.accel/8,
      braking: state.braking/6,
      steering: (state.steering + 40)/80,
      lane: state.laneDev/0.8,
      eye: state.ear,
      head: (state.headYaw + 40)/80,
      phone: state.phoneDetected ? 1 : 0,
    };
    state.sparkData[k].push(normMap[k] ?? 0.5);
    if (state.sparkData[k].length > 30) state.sparkData[k].shift();
  });

  // History data
  state.history.speed.push(state.speed);
  state.history.accel.push(state.accel);
  state.history.braking.push(state.braking);
  state.history.eye.push(state.ear);
  state.history.risk.push(state.riskScore);
  if (state.history.speed.length > 120) {
    Object.values(state.history).forEach(arr => arr.shift());
  }

  // Telemetry
  telData.speed.push(state.speed);
  telData.accel.push(state.accel * 20);
  telData.braking.push(state.braking * 25);
}

// ── Initialize ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initBackground();
  initNav();
  initClock();
  initSessionTimer();
  initRiskGauge();
  initTelemetryChart();
  initBehaviorTimeline();
  initSparklines();
  initSensorsSection();
  initHistorySection();
  initModelSection();
  initAlertsSection();
  initSliders();
  initToast();
  initHeadPoseCanvas();
  initEarGauge();
  initRadarChart();

  // Try to connect to server, then start simulation as fallback
  initSocketIO();
  startSimulation();
});

// ── Animated background particles ────────────────────────────
function initBackground() {
  const canvas = $('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles;

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = Array.from({length: 60}, () => ({
      x: rand(0, W), y: rand(0, H),
      vx: rand(-0.15, 0.15), vy: rand(-0.15, 0.15),
      r: rand(0.5, 2),
      color: Math.random() > 0.5 ? '#00d4ff' : '#a855f7',
      alpha: rand(0.1, 0.5),
    }));
  }

  resize();
  createParticles();
  window.addEventListener('resize', () => { resize(); createParticles(); });

  function draw() {
    ctx.clearRect(0, 0, W, H);
    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d = Math.sqrt(dx*dx + dy*dy);
        if (d < 120) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0,212,255,${0.04 * (1 - d/120)})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
    // Draw particles
    for (const p of particles) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    requestAnimationFrame(draw);
  }
  draw();
}

// ── Navigation ────────────────────────────────────────────────
function initNav() {
  const sections = ['dashboard','camera','sensors','history','model','alerts'];
  const labels = { dashboard:'Dashboard', camera:'Driver Camera', sensors:'Sensors', history:'History', model:'AI Model', alerts:'Alerts' };

  sections.forEach(s => {
    $(`nav-${s}`).addEventListener('click', e => {
      e.preventDefault();
      sections.forEach(x => {
        $(`nav-${x}`).classList.remove('active');
        $(`section-${x}`).classList.remove('active');
      });
      $(`nav-${s}`).classList.add('active');
      $(`section-${s}`).classList.add('active');
      $('breadcrumb-label').textContent = labels[s];
      // Trigger section-specific draws
      if (s === 'sensors') drawAllGauges();
      if (s === 'history') drawHistoryChart();
      if (s === 'model')   drawConfusionMatrix();
    });
  });

  // Sidebar toggle (mobile)
  $('sidebar-toggle').addEventListener('click', () => {
    document.querySelector('.sidebar').classList.toggle('open');
  });
}

// ── Clock & Session Timer ─────────────────────────────────────
function initClock() {
  function update() {
    const now = new Date();
    $('current-time').textContent = now.toLocaleTimeString('en-US', { hour12: false });
  }
  update();
  setInterval(update, 1000);
}

function initSessionTimer() {
  setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.sessionStart) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2,'0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2,'0');
    const s = String(elapsed % 60).padStart(2,'0');
    $('session-time').textContent = `${h}:${m}:${s}`;
  }, 1000);
}

// ── Risk Gauge ────────────────────────────────────────────────
let riskCanvas, riskCtx;
function initRiskGauge() {
  riskCanvas = $('risk-gauge');
  riskCtx = riskCanvas.getContext('2d');
  drawRiskGauge(state.riskScore);
}

function drawRiskGauge(score) {
  const ctx = riskCtx;
  const W = riskCanvas.width, H = riskCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 10;
  const r = 90;
  const startA = Math.PI, endA = 0;

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, startA, endA, false);
  ctx.lineWidth = 14;
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineCap = 'round';
  ctx.stroke();

  // Gradient arc
  const pct = score / 100;
  const sweepEnd = startA + pct * Math.PI;
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0, '#10b981');
  grad.addColorStop(0.5, '#f59e0b');
  grad.addColorStop(1, '#ef4444');

  ctx.beginPath();
  ctx.arc(cx, cy, r, startA, sweepEnd, false);
  ctx.lineWidth = 14;
  ctx.strokeStyle = grad;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Glow effect
  ctx.beginPath();
  ctx.arc(cx, cy, r, startA, sweepEnd, false);
  ctx.lineWidth = 20;
  ctx.strokeStyle = score < 30
    ? 'rgba(16,185,129,0.12)'
    : score < 65 ? 'rgba(245,158,11,0.12)' : 'rgba(239,68,68,0.12)';
  ctx.lineCap = 'round';
  ctx.stroke();

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const a = Math.PI + (i/10) * Math.PI;
    const x1 = cx + (r - 20) * Math.cos(a);
    const y1 = cy + (r - 20) * Math.sin(a);
    const x2 = cx + (r - 10) * Math.cos(a);
    const y2 = cy + (r - 10) * Math.sin(a);
    ctx.beginPath();
    ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = i % 5 === 0 ? 2 : 1;
    ctx.stroke();
  }

  // Needle
  const needleA = Math.PI + pct * Math.PI;
  const nx = cx + (r - 25) * Math.cos(needleA);
  const ny = cy + (r - 25) * Math.sin(needleA);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, Math.PI*2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();

  // Update DOM
  $('risk-number').textContent = Math.round(score);
  updateRiskStatus(score);
}

function updateRiskStatus(score) {
  const el = $('risk-status-badge');
  const num = $('risk-number');
  const text = $('risk-status-text');

  if (score < 30) {
    el.style.cssText = 'background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.3);color:#10b981';
    text.textContent = 'LOW RISK';
    num.style.color = '#10b981';
  } else if (score < 65) {
    el.style.cssText = 'background:rgba(245,158,11,0.12);border-color:rgba(245,158,11,0.3);color:#f59e0b';
    text.textContent = 'MODERATE RISK';
    num.style.color = '#f59e0b';
  } else {
    el.style.cssText = 'background:rgba(239,68,68,0.12);border-color:rgba(239,68,68,0.3);color:#ef4444';
    text.textContent = 'HIGH RISK';
    num.style.color = '#ef4444';
  }
}

// ── Telemetry Chart ───────────────────────────────────────────
let telCtx, telData = { speed:[], accel:[], braking:[] };
const TEL_POINTS = 60;

function initTelemetryChart() {
  const canvas = $('telemetry-chart');
  canvas.style.width = '100%';
  canvas.style.height = '200px';

  // Initialize with history
  telData.speed   = [...state.history.speed];
  telData.accel   = state.history.accel.map(v => v * 20); // normalize for display
  telData.braking = state.history.braking.map(v => v * 25);

  telCtx = canvas.getContext('2d');
  drawTelemetry();
}

function drawTelemetry() {
  const canvas = $('telemetry-chart');
  const W = canvas.offsetWidth || 600;
  canvas.width = W;
  const H = 200;
  const ctx = telCtx;

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }

  // Draw series
  const series = [
    { data: telData.speed, color: '#00d4ff', scale: 150 },
    { data: telData.accel, color: '#a855f7', scale: 50  },
    { data: telData.braking, color: '#f59e0b', scale: 50 },
  ];

  series.forEach(({ data, color, scale }) => {
    if (data.length < 2) return;
    const pts = data.slice(-TEL_POINTS);

    // Gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, color.replace(')', ',0.2)').replace('rgb', 'rgba'));
    grad.addColorStop(1, 'transparent');

    ctx.beginPath();
    pts.forEach((v, i) => {
      const x = (i / (pts.length - 1)) * W;
      const y = H - clamp(v / scale, 0, 1) * (H - 20) - 10;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Fill
    ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  });
}

// ── Behavior Timeline ─────────────────────────────────────────
const CLASS_COLORS = {
  normal:          '#10b981',
  aggressive:      '#ef4444',
  distracted:      '#f59e0b',
  drowsy:          '#a855f7',
  'unsafe-braking':'#f59e0b',
  'unsafe-accel':  '#f97316',
};

function initBehaviorTimeline() {
  const el = $('behavior-timeline');
  const entries = [
    { t: '–0:55', cls: 'normal',    label: 'Normal Driving' },
    { t: '–0:42', cls: 'aggressive',label: 'Aggressive Accel.' },
    { t: '–0:38', cls: 'normal',    label: 'Normal Driving' },
    { t: '–0:21', cls: 'distracted',label: 'Distracted' },
    { t: '–0:14', cls: 'normal',    label: 'Normal Driving' },
    { t: '–0:05', cls: 'normal',    label: 'Normal Driving' },
    { t: 'Now',   cls: 'normal',    label: 'Normal Driving' },
  ];

  el.innerHTML = entries.map(e => `
    <div class="timeline-entry">
      <span class="timeline-time">${e.t}</span>
      <span class="timeline-dot" style="background:${CLASS_COLORS[e.cls]}"></span>
      <span class="timeline-label">${e.label}</span>
    </div>
  `).join('');
}

function appendTimeline(cls, label) {
  const el = $('behavior-timeline');
  const now = new Date();
  const t = now.toLocaleTimeString('en-US', { hour12: false });
  const div = document.createElement('div');
  div.className = 'timeline-entry';
  div.innerHTML = `
    <span class="timeline-time">${t}</span>
    <span class="timeline-dot" style="background:${CLASS_COLORS[cls]||'#888'}"></span>
    <span class="timeline-label">${label}</span>
  `;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  // Keep max 20 entries
  while (el.children.length > 20) el.removeChild(el.firstChild);
}

// ── Sparklines ────────────────────────────────────────────────
const SPARK_COLORS = {
  speed:'#00d4ff', accel:'#a855f7', braking:'#ef4444',
  steering:'#06b6d4', lane:'#10b981', eye:'#f59e0b',
  head:'#a855f7', phone:'#ef4444',
};

function initSparklines() {
  Object.keys(state.sparkData).forEach(key => {
    state.sparkData[key] = Array.from({length: 30}, () => Math.random());
  });
}

function drawSparkline(canvasId, data, color) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (data.length < 2) return;
  const min = Math.min(...data);
  const max = Math.max(...data) || 1;
  const range = max - min || 1;

  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - ((v - min) / range) * (H - 2) - 1;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.stroke();
}

function updateAllSparklines() {
  const pairs = [
    ['spark-speed',   state.sparkData.speed,    SPARK_COLORS.speed],
    ['spark-accel',   state.sparkData.accel,    SPARK_COLORS.accel],
    ['spark-braking', state.sparkData.braking,  SPARK_COLORS.braking],
    ['spark-steering',state.sparkData.steering, SPARK_COLORS.steering],
    ['spark-lane',    state.sparkData.lane,      SPARK_COLORS.lane],
    ['spark-eye',     state.sparkData.eye,       SPARK_COLORS.eye],
    ['spark-head',    state.sparkData.head,      SPARK_COLORS.head],
    ['spark-phone',   state.sparkData.phone,     SPARK_COLORS.phone],
  ];
  pairs.forEach(([id, data, color]) => drawSparkline(id, data, color));
}

// ── Sensor Gauges (Sensors page) ──────────────────────────────
function drawArcGauge(canvasId, value, min, max, color) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W/2, cy = H - 10;
  const r = 52;
  const pct = clamp((value - min) / (max - min), 0, 1);

  // BG arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.lineWidth = 10;
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineCap = 'round';
  ctx.stroke();

  // Value arc
  const sweep = Math.PI + pct * Math.PI;
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0, color);
  grad.addColorStop(1, color.replace('ff', 'cc'));
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, sweep, false);
  ctx.lineWidth = 10;
  ctx.strokeStyle = grad;
  ctx.lineCap = 'round';
  ctx.stroke();
}

function drawAllGauges() {
  drawArcGauge('g-speed', state.speed, 0, 150, '#00d4ff');
  drawArcGauge('g-accel', state.accel, 0, 10,  '#a855f7');
  drawArcGauge('g-brake', state.braking, 0, 10, '#ef4444');
  drawArcGauge('g-steer', Math.abs(state.steering), 0, 45, '#06b6d4');
  drawArcGauge('g-lane',  state.laneDev, 0, 1,  '#10b981');
  drawArcGauge('g-eye',   state.ear, 0, 1,      '#f59e0b');
}

// ── EAR Gauge (Camera page) ───────────────────────────────────
function initEarGauge() { drawEarGauge(state.ear); }

function drawEarGauge(ear) {
  const canvas = $('ear-gauge');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const cx = W/2, cy = H - 5;
  const r = 72;
  const pct = clamp(ear, 0, 1);

  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.lineWidth = 10; ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineCap = 'round'; ctx.stroke();

  const color = ear < 0.25 ? '#ef4444' : ear < 0.35 ? '#f59e0b' : '#10b981';
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, Math.PI + pct*Math.PI, false);
  ctx.lineWidth = 10; ctx.strokeStyle = color; ctx.lineCap = 'round'; ctx.stroke();

  $('ear-display').textContent = ear.toFixed(2);
  $('ear-display').style.color = color;
  $('cam-ear').textContent = ear.toFixed(2);
}

// ── Head Pose Canvas ──────────────────────────────────────────
function initHeadPoseCanvas() { drawHeadPose(state.headYaw, state.headPitch); }

function drawHeadPose(yaw, pitch) {
  const canvas = $('head-pose-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W/2, cy = H/2;

  // Face circle (front view)
  ctx.beginPath();
  ctx.arc(cx, cy, 40, 0, Math.PI*2);
  ctx.strokeStyle = 'rgba(0,212,255,0.3)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Reference cross
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx - 55, cy); ctx.lineTo(cx + 55, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - 55); ctx.lineTo(cx, cy + 55); ctx.stroke();

  // Yaw indicator
  const yawX = cx + (yaw / 45) * 40;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(yawX, cy);
  ctx.strokeStyle = '#ff6b6b'; ctx.lineWidth = 2.5; ctx.lineCap = 'round'; ctx.stroke();

  // Pitch indicator
  const pitchY = cy + (pitch / 45) * 40;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx, pitchY);
  ctx.strokeStyle = '#00d4ff'; ctx.lineWidth = 2.5; ctx.lineCap = 'round'; ctx.stroke();

  // Dot
  ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI*2);
  ctx.fillStyle = '#ffffff'; ctx.fill();

  // Labels
  ctx.font = '9px JetBrains Mono';
  ctx.fillStyle = '#4a5578';
  ctx.fillText(`YAW: ${yaw > 0 ? '+' : ''}${yaw.toFixed(1)}°`, 4, H - 20);
  ctx.fillText(`PITCH: ${pitch > 0 ? '+' : ''}${pitch.toFixed(1)}°`, 4, H - 8);
}

// ── Radar Chart (Sensors page) ────────────────────────────────
let radarCtx;
function initRadarChart() {
  const canvas = $('radar-chart');
  if (!canvas) return;
  radarCtx = canvas.getContext('2d');
  drawRadar();
}

function drawRadar() {
  const canvas = $('radar-chart');
  if (!canvas || !radarCtx) return;
  const ctx = radarCtx;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const params = ['Speed','Accel','Braking','Steering','Lane Dev','Eye (EAR)'];
  const safeVals = [0.48, 0.08, 0.04, 0.18, 0.12, 0.82]; // normalized safe reference
  const currentVals = [
    state.speed / 150,
    state.accel / 10,
    state.braking / 10,
    Math.abs(state.steering) / 45,
    state.laneDev / 1,
    state.ear,
  ];

  const n = params.length;
  const cx = W/2, cy = H/2 - 10;
  const R = 130;

  const angleSlice = (Math.PI * 2) / n;

  // Grid rings
  for (let r = 1; r <= 5; r++) {
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const a = angleSlice * i - Math.PI/2;
      const x = cx + (R * r/5) * Math.cos(a);
      const y = cy + (R * r/5) * Math.sin(a);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Axes
  for (let i = 0; i < n; i++) {
    const a = angleSlice * i - Math.PI/2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a));
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Labels
    const lx = cx + (R + 18) * Math.cos(a);
    const ly = cy + (R + 18) * Math.sin(a);
    ctx.font = '11px Outfit';
    ctx.fillStyle = '#8b9cc8';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(params[i], lx, ly);
  }

  // Safe polygon
  ctx.beginPath();
  safeVals.forEach((v, i) => {
    const a = angleSlice * i - Math.PI/2;
    const x = cx + R * v * Math.cos(a);
    const y = cy + R * v * Math.sin(a);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.strokeStyle = 'rgba(16,185,129,0.4)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(16,185,129,0.06)';
  ctx.fill();

  // Current polygon
  ctx.beginPath();
  currentVals.forEach((v, i) => {
    const a = angleSlice * i - Math.PI/2;
    const x = cx + R * clamp(v, 0, 1) * Math.cos(a);
    const y = cy + R * clamp(v, 0, 1) * Math.sin(a);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = 'rgba(0,212,255,0.12)';
  ctx.fill();

  // Dots on current
  currentVals.forEach((v, i) => {
    const a = angleSlice * i - Math.PI/2;
    const x = cx + R * clamp(v, 0, 1) * Math.cos(a);
    const y = cy + R * clamp(v, 0, 1) * Math.sin(a);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2);
    ctx.fillStyle = '#00d4ff'; ctx.fill();
  });
}

// ── History Chart ─────────────────────────────────────────────
function drawHistoryChart() {
  const canvas = $('history-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const data = state.history.risk;
  const n = data.length;

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i/4) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(100 - i*25, 4, y + 3);
  }

  // Threshold lines
  [[30, '#10b981'], [65, '#ef4444']].forEach(([val, color]) => {
    const y = H - (val/100) * H;
    ctx.beginPath();
    ctx.setLineDash([5, 4]);
    ctx.moveTo(40, y); ctx.lineTo(W, y);
    ctx.strokeStyle = color.replace(')', ',0.4)').replace('#', 'rgba(').replace('10b981', '16,185,129').replace('ef4444','239,68,68');
    ctx.strokeStyle = color + '66';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // Risk line
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = 40 + (i / (n-1)) * (W - 50);
    const y = H - clamp(v/100, 0, 1) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });

  const lineGrad = ctx.createLinearGradient(40, 0, W, 0);
  lineGrad.addColorStop(0, '#10b981');
  lineGrad.addColorStop(0.5, '#f59e0b');
  lineGrad.addColorStop(1, '#00d4ff');

  ctx.strokeStyle = lineGrad;
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Fill
  const fillGrad = ctx.createLinearGradient(0, 0, 0, H);
  fillGrad.addColorStop(0, 'rgba(0,212,255,0.15)');
  fillGrad.addColorStop(1, 'transparent');
  ctx.lineTo(W - 10, H); ctx.lineTo(40, H); ctx.closePath();
  ctx.fillStyle = fillGrad;
  ctx.fill();

  // Time labels
  ctx.font = '10px JetBrains Mono';
  ctx.fillStyle = 'rgba(255,255,255,0.2)';
  const times = ['-10:00','-8:00','-6:00','-4:00','-2:00','Now'];
  times.forEach((t, i) => {
    const x = 40 + (i / (times.length - 1)) * (W - 50);
    ctx.fillText(t, x - 15, H - 4);
  });
}

// ── Incidents Table ───────────────────────────────────────────
function initHistorySection() {
  const tbody = $('events-tbody');
  tbody.innerHTML = state.incidents.map(ev => `
    <tr>
      <td style="font-family:'JetBrains Mono',monospace;color:var(--text-dim)">${ev.time}</td>
      <td style="color:var(--text-primary);font-weight:500">${ev.event}</td>
      <td><span class="severity-badge sev-${ev.severity}">${ev.severity.toUpperCase()}</span></td>
      <td style="font-family:'JetBrains Mono',monospace">${ev.dur}</td>
      <td style="color:${ev.riskDelta.startsWith('+') ? 'var(--red)' : 'var(--green)'}; font-weight:600">${ev.riskDelta}</td>
    </tr>
  `).join('');
}

// ── Confusion Matrix ──────────────────────────────────────────
function initModelSection() { /* called via nav */ }

function drawConfusionMatrix() {
  const el = $('confusion-matrix');
  if (!el) return;
  const labels = ['N','Ag','Di','Dr','UB','UA'];
  const matrix = [
    [47, 1, 0, 1, 0, 0],
    [ 0,48, 1, 0, 1, 0],
    [ 1, 0,46, 0, 0, 1],
    [ 0, 0, 0,44, 0, 0],
    [ 0, 1, 0, 0,47, 1],
    [ 0, 0, 1, 0, 0,48],
  ];

  const headers = ['', ...labels];
  let html = headers.map(h => `<div class="cm-cell cm-header">${h}</div>`).join('');

  matrix.forEach((row, ri) => {
    html += `<div class="cm-cell cm-header">${labels[ri]}</div>`;
    row.forEach((val, ci) => {
      const cls = ri === ci ? 'cm-diag' : 'cm-val';
      html += `<div class="cm-cell ${cls}">${val}</div>`;
    });
  });
  el.innerHTML = html;
}

// ── Alerts Section ────────────────────────────────────────────
function initAlertsSection() {
  renderAlertFeed();

  $('clear-alerts-btn').addEventListener('click', () => {
    $('alerts-feed').innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-dim);font-size:0.8rem">No alerts. All systems normal.</div>';
    $('alert-badge').textContent = '0';
    state.alertCount = 0;
  });
}

function renderAlertFeed() {
  const el = $('alerts-feed');
  el.innerHTML = state.alertFeed.map(a => `
    <div class="alert-entry alert-${a.type}">
      <div class="alert-icon">${a.icon}</div>
      <div class="alert-body">
        <div class="alert-title">${a.title}</div>
        <div class="alert-desc">${a.desc}</div>
        <div class="alert-meta">Session time: ${a.time}</div>
      </div>
    </div>
  `).join('');
}

function addAlert(type, icon, title, desc) {
  const now = new Date();
  const elapsed = Math.floor((Date.now() - state.sessionStart) / 1000);
  const h = String(Math.floor(elapsed / 3600)).padStart(2,'0');
  const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2,'0');
  const s = String(elapsed % 60).padStart(2,'0');

  state.alertFeed.unshift({ type, icon, title, desc, time: `${h}:${m}:${s}` });
  state.alertCount++;
  $('alert-badge').textContent = state.alertCount;

  // Re-render feed if visible
  if ($('section-alerts').classList.contains('active')) {
    renderAlertFeed();
  }

  // Show toast
  showToast(title, desc);
}

// ── Slider Init ───────────────────────────────────────────────
function initSliders() {
  const sliders = [
    { id: 'thr-ear',   valId: 'thr-ear-val',   fmt: v => (v/100).toFixed(2) },
    { id: 'thr-speed', valId: 'thr-speed-val',  fmt: v => `${v} km/h` },
    { id: 'thr-accel', valId: 'thr-accel-val',  fmt: v => `${(v/10).toFixed(1)} m/s²` },
    { id: 'thr-lane',  valId: 'thr-lane-val',   fmt: v => `${(v/100).toFixed(2)} m` },
    { id: 'thr-risk',  valId: 'thr-risk-val',   fmt: v => v },
  ];

  sliders.forEach(({ id, valId, fmt }) => {
    const slider = $(id);
    const valEl = $(valId);
    if (!slider) return;
    valEl.textContent = fmt(slider.value);
    slider.addEventListener('input', () => {
      valEl.textContent = fmt(slider.value);
      const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
      slider.style.background = `linear-gradient(90deg, #00d4ff ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
    });
    // Initialize gradient
    const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
    slider.style.background = `linear-gradient(90deg, #00d4ff ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
  });
}

// ── Toast ─────────────────────────────────────────────────────
let toastTimer;
function initToast() {
  $('toast-close').addEventListener('click', () => {
    $('alert-toast').classList.remove('show');
    clearTimeout(toastTimer);
  });
}

function showToast(title, msg) {
  $('toast-title').textContent = title;
  $('toast-msg').textContent = msg;
  $('alert-toast').classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('alert-toast').classList.remove('show'), 5000);
}

// ── Main Simulation Loop ──────────────────────────────────────
let tick = 0;
let drowsyPhase = 0, aggrPhase = 0, distPhase = 0;

function startSimulation() {
  // Show initial toast
  setTimeout(() => showToast('Session Started', 'DriveGuard AI is monitoring your drive'), 1200);

  setInterval(simulateTick, 300);
  setInterval(drawTelemetry, 600);
  setInterval(updateAllSparklines, 500);

  // Periodic random events
  setInterval(triggerRandomEvent, 12000);
}

function simulateTick() {
  // Skip standalone simulation when backend is sending data
  if (serverConnected) return;
  tick++;

  // Smooth random walk for all params
  state.speed    = clamp(state.speed    + rand(-2, 2),    0, 140);
  state.accel    = clamp(state.accel    + rand(-0.2, 0.2), 0, 8);
  state.braking  = clamp(state.braking  + rand(-0.1, 0.1), 0, 6);
  state.steering = clamp(state.steering + rand(-1, 1),    -40, 40);
  state.laneDev  = clamp(state.laneDev  + rand(-0.03, 0.03), 0, 0.8);
  state.ear      = clamp(state.ear      + rand(-0.02, 0.02), 0.15, 1);
  state.headYaw  = clamp(state.headYaw  + rand(-0.8, 0.8),  -40, 40);
  state.headPitch= clamp(state.headPitch+ rand(-0.5, 0.5),  -30, 30);

  // Push to history
  Object.keys(state.sparkData).forEach(k => {
    const normMap = {
      speed: state.speed/140,
      accel: state.accel/8,
      braking: state.braking/6,
      steering: (state.steering + 40)/80,
      lane: state.laneDev/0.8,
      eye: state.ear,
      head: (state.headYaw + 40)/80,
      phone: state.phoneDetected ? 1 : 0,
    };
    state.sparkData[k].push(normMap[k] ?? 0.5);
    if (state.sparkData[k].length > 30) state.sparkData[k].shift();
  });

  state.history.speed.push(state.speed);
  state.history.accel.push(state.accel);
  state.history.braking.push(state.braking);
  state.history.eye.push(state.ear);
  state.history.risk.push(state.riskScore);
  if (state.history.speed.length > 120) {
    Object.values(state.history).forEach(arr => arr.shift());
  }

  telData.speed.push(state.speed);
  telData.accel.push(state.accel * 20);
  telData.braking.push(state.braking * 25);

  // Compute risk score
  let risk = 10;
  if (state.speed > 100)        risk += (state.speed - 100) * 0.3;
  if (state.accel > 3)          risk += (state.accel - 3) * 5;
  if (state.braking > 3)        risk += (state.braking - 3) * 6;
  if (state.ear < 0.30)         risk += (0.30 - state.ear) * 150;
  if (state.laneDev > 0.25)     risk += (state.laneDev - 0.25) * 40;
  if (Math.abs(state.headYaw) > 20) risk += (Math.abs(state.headYaw) - 20) * 1.5;
  if (state.phoneDetected)      risk += 35;
  state.riskScore = clamp(lerp(state.riskScore, risk, 0.15), 0, 100);

  // Classify behavior
  let cls = 'normal', label = 'Normal Driving';
  if (state.ear < 0.28)                               { cls = 'drowsy';          label = 'Drowsiness'; }
  else if (state.phoneDetected)                        { cls = 'distracted';      label = 'Phone Usage'; }
  else if (state.braking > 4)                          { cls = 'unsafe-braking';  label = 'Unsafe Braking'; }
  else if (state.accel > 4.5)                          { cls = 'unsafe-accel';    label = 'Unsafe Accel.'; }
  else if (state.accel > 3 || state.speed > 100)       { cls = 'aggressive';      label = 'Aggressive'; }
  else if (Math.abs(state.headYaw) > 22 || state.laneDev > 0.3) { cls = 'distracted'; label = 'Distracted'; }

  if (cls !== state.currentClass) {
    state.currentClass = cls;
    appendTimeline(cls, label);
    if (cls !== 'normal') addAlert('warning', '⚠️', `${label} Detected`, `Risk score: ${Math.round(state.riskScore)}`);
  }

  updateUI(cls);
}

function updateUI(cls) {
  // Risk gauge
  drawRiskGauge(state.riskScore);

  // Quick stats
  $('qs-speed').textContent  = Math.round(state.speed);
  $('qs-accel').textContent  = state.accel.toFixed(1);
  $('qs-eye').textContent    = state.ear.toFixed(2);
  $('qs-lane').textContent   = state.laneDev.toFixed(2);

  // Param cards
  $('pv-speed').textContent   = `${Math.round(state.speed)} km/h`;
  $('pv-accel').textContent   = `${state.accel.toFixed(2)} m/s²`;
  $('pv-braking').textContent = `${state.braking.toFixed(2)} m/s²`;
  $('pv-steering').textContent= `${Math.round(state.steering)}°`;
  $('pv-lane').textContent    = `${state.laneDev.toFixed(2)} m`;
  $('pv-eye').textContent     = state.ear.toFixed(3);
  $('pv-head').textContent    = Math.abs(state.headYaw) < 10 && Math.abs(state.headPitch) < 10 ? 'Centered' : 'Deviated';
  $('pv-phone').textContent   = state.phoneDetected ? '⚠️ Detected' : 'Not Detected';

  // Param status
  setParamStatus('param-speed',   'stat-speed',   state.speed,   90, 110);
  setParamStatus('param-accel',   'stat-accel',   state.accel,   3,  5);
  setParamStatus('param-braking', 'param-braking',state.braking, 3,  5);
  setParamStatus('param-eye',     'stat-eye',     state.ear,     0.35, 0.25, true); // inverted

  // Behavior classification
  updateBehaviorClasses(cls);

  // Camera section
  drawEarGauge(state.ear);
  drawHeadPose(state.headYaw, state.headPitch);
  $('cam-yaw').textContent   = `${state.headYaw > 0 ? '+' : ''}${state.headYaw.toFixed(1)}°`;
  $('cam-pitch').textContent = `${state.headPitch > 0 ? '+' : ''}${state.headPitch.toFixed(1)}°`;

  // Drowsy indicator
  const di = $('drowsy-indicator');
  if (state.ear < 0.28) {
    di.innerHTML = '<span class="drowsy-dot"></span> DROWSY ALERT';
    di.classList.add('alert');
  } else {
    di.innerHTML = '<span class="drowsy-dot"></span> EYES OPEN';
    di.classList.remove('alert');
  }

  // Phone status
  const phoneEl = $('cam-phone-status');
  if (state.phoneDetected) {
    phoneEl.innerHTML = 'PHONE: <span class="hud-val phone-alert">DETECTED</span>';
  } else {
    phoneEl.innerHTML = 'PHONE: <span class="hud-val phone-ok">NONE</span>';
  }

  // Sensor gauges (if visible)
  if ($('section-sensors').classList.contains('active')) {
    drawAllGauges();
    $('gv-speed').textContent  = `${Math.round(state.speed)} km/h`;
    $('gv-accel').textContent  = `${state.accel.toFixed(2)} m/s²`;
    $('gv-brake').textContent  = `${state.braking.toFixed(2)} m/s²`;
    $('gv-steer').textContent  = `${Math.round(state.steering)}°`;
    $('gv-lane').textContent   = `${state.laneDev.toFixed(2)} m`;
    $('gv-eye').textContent    = state.ear.toFixed(3);

    $('gb-speed').style.width  = `${(state.speed/140)*100}%`;
    $('gb-accel').style.width  = `${(state.accel/8)*100}%`;
    $('gb-brake').style.width  = `${(state.braking/6)*100}%`;
    $('gb-steer').style.width  = `${(Math.abs(state.steering)/40)*100}%`;
    $('gb-lane').style.width   = `${(state.laneDev/0.8)*100}%`;
    $('gb-eye').style.width    = `${state.ear*100}%`;

    drawRadar();
  }

  // Animate face overlay
  animateFaceOverlay();
}

function setParamStatus(cardId, _statId, val, warnThresh, dangerThresh, inverted = false) {
  const card = $(cardId);
  const statusEl = card ? card.querySelector('.param-status') : null;
  if (!statusEl) return;

  const isWarn = inverted ? val < warnThresh : val > warnThresh;
  const isDanger = inverted ? val < dangerThresh : val > dangerThresh;

  if (isDanger) {
    statusEl.textContent = 'ALERT';
    statusEl.className = 'param-status danger';
    card.classList.add('danger'); card.classList.remove('warning');
  } else if (isWarn) {
    statusEl.textContent = 'WARN';
    statusEl.className = 'param-status warn';
    card.classList.add('warning'); card.classList.remove('danger');
  } else {
    statusEl.textContent = 'OK';
    statusEl.className = 'param-status ok';
    card.classList.remove('warning','danger');
  }
}

const CLASS_LABELS = {
  normal: 'Normal Driving', aggressive: 'Aggressive', distracted: 'Distracted',
  drowsy: 'Drowsy', 'unsafe-braking': 'Unsafe Braking', 'unsafe-accel': 'Unsafe Accel.',
};

function updateBehaviorClasses(activeClass) {
  const classes = ['normal','aggressive','distracted','drowsy','unsafe-braking','unsafe-accel'];
  const confIds = {
    normal: 'conf-normal', aggressive: 'conf-aggressive', distracted: 'conf-distracted',
    drowsy: 'conf-drowsy', 'unsafe-braking': 'conf-braking', 'unsafe-accel': 'conf-accel',
  };

  // Distribute confidence values
  let confs = {};
  confs[activeClass] = rand(88, 97);
  const rest = classes.filter(c => c !== activeClass);
  let remaining = 100 - confs[activeClass];
  rest.forEach((c, i) => {
    if (i === rest.length - 1) {
      confs[c] = Math.max(0, remaining);
    } else {
      const v = rand(0, remaining / (rest.length - i));
      confs[c] = v;
      remaining -= v;
    }
  });

  classes.forEach(c => {
    const el = document.querySelector(`.behavior-class[data-class="${c}"]`);
    if (!el) return;
    el.classList.toggle('active-class', c === activeClass);
    const bar = el.querySelector('.class-bar');
    const pct = confs[c].toFixed(1);
    if (bar) bar.style.width = `${pct}%`;
    const confEl = $(confIds[c]);
    if (confEl) confEl.textContent = `${pct}%`;
  });
}

function animateFaceOverlay() {
  const t = Date.now() / 1000;
  // Subtle eye blink animation
  const eyeH = state.ear < 0.28 ? 8 : 22;
  $('left-eye-box') .setAttribute('height', eyeH);
  $('right-eye-box').setAttribute('height', eyeH);

  // Head pose on SVG
  const yawDelta = (state.headYaw / 40) * 20;
  const pitchDelta = (state.headPitch / 30) * 15;

  const poseYaw  = $('pose-yaw');
  const posePitch= $('pose-pitch');
  if (poseYaw) {
    poseYaw.setAttribute('x2', 200 + yawDelta);
    poseYaw.setAttribute('y1', 140 + pitchDelta);
    poseYaw.setAttribute('y2', 140 + pitchDelta);
  }
  if (posePitch) {
    posePitch.setAttribute('x1', 200 + yawDelta);
    posePitch.setAttribute('x2', 200 + yawDelta);
    posePitch.setAttribute('y2', 110 + pitchDelta);
  }
}

// ── Random Event Trigger ──────────────────────────────────────
const EVENTS = [
  () => { state.braking = rand(4, 6); setTimeout(() => state.braking = rand(0.1, 0.5), 2000); addAlert('critical','🛑','Hard Braking','Deceleration spike detected'); },
  () => { state.accel = rand(5, 7);   setTimeout(() => state.accel   = rand(0.3, 1),   3000); addAlert('warning','⚡','Aggressive Acceleration','Acceleration threshold exceeded'); },
  () => { state.ear = rand(0.15, 0.22); setTimeout(() => state.ear   = rand(0.7, 0.85),4000); addAlert('critical','😴','Drowsiness Alert','Eye closure sustained — please rest'); },
  () => { state.phoneDetected = true;   setTimeout(() => state.phoneDetected = false,   5000); addAlert('critical','📱','Phone Detected','Driver appears to be using a device'); },
  () => { state.laneDev = rand(0.4, 0.6); setTimeout(() => state.laneDev = rand(0.05, 0.15), 3000); addAlert('warning','🛣️','Lane Deviation','Vehicle drifting from center'); },
];

function triggerRandomEvent() {
  const ev = EVENTS[Math.floor(Math.random() * EVENTS.length)];
  ev();
  // Update detection log
  const log = $('detection-log');
  if (log) {
    const entry = document.createElement('div');
    entry.className = 'log-entry log-warn';
    entry.textContent = `⚠ Anomaly detected — ${new Date().toLocaleTimeString()}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
    if (log.children.length > 8) log.removeChild(log.firstChild);
  }
}
