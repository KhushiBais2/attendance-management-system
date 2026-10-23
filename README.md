# Multimodal Attendance & Engagement Agent

An AI-powered, privacy-preserving classroom intelligence web application that integrates computer vision, audio signal processing, and agent-based pedagogical reasoning to track sustained attendance, analyze real-time engagement trends, and generate actionable teaching suggestions.

---

## 1. System Architecture

The application is structured into a clean 4-layer architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. FRONTEND LAYER                                                      │
│    • HTML5 Dashboard (templates/index.html)                           │
│    • CSS3 Glassmorphism Theme (static/css/style.css)                   │
│    • JavaScript Controller & Chart.js Engine (static/js/controller.js) │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP / MJPEG Stream / REST
┌────────────────────────────────────▼───────────────────────────────────┐
│ 2. BACKEND / API GATEWAY LAYER (app.py)                                │
│    • Flask API Gateway & Route Handlers                                │
│    • Video Streaming Endpoint (/video_feed)                            │
│    • Telemetry Aggregator (/api/metrics)                               │
│    • Safe Code Execution Sandbox (utils/code_executor.py)              │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
       ┌─────────────────────────────┴─────────────────────────────┐
       │                                                           │
┌──────▼──────────────────────────────────┐ ┌──────────────────────▼──────┐
│ 3. MULTIMODAL INTELLIGENCE LAYER        │ │ 4. STORAGE & UTILITY LAYER │
│    • Vision Module                      │ │    • storage/              │
│      - MediaPipe FaceLandmarker         │ │      - attendance_logs.json│
│      - OpenCV Haar Cascade Fallback     │ │      - session_analytics.j │
│      - 3D Head Pose (Yaw/Pitch)         │ │      - generated_audio/    │
│      - Sustained Presence Verification  │ │      - generated_code/     │
│      - Synthetic Classroom Simulator    │ │      - uploaded_files/     │
│    • Audio Module                       │ │    • localStorage Sync     │
│      - RMS Energy & Zero-Crossing Rate  │ └────────────────────────────┘
│      - Dynamic Acoustic Simulation      │
│    • Fusion & Analytics Engine          │
│      - Weighted Formula (65% V + 35% A) │
│      - Inactivity Penalty Subtraction   │
│    • Pedagogical AI Agent               │
│      - Google GenAI (gemini-2.0-flash)  │
│      - 100% Offline Rule Fallback Engine│
│      - Text-to-Speech (pyttsx3/gTTS)    │
│      - Python Educational Code Gen      │
└─────────────────────────────────────────┘
```

---

## 2. Core Functional Modules & Epics

### EPIC 1 & 2: Vision Module (`utils/vision_module.py`)
- **Automated Sustained Attendance**: Unlike single-frame detectors that trigger false positives from passersby or transient glimpses, the system tracks each learner across consecutive frames. Attendance is marked **"Present (Verified)"** only after uninterrupted presence exceeds a configurable threshold (default `4.0 seconds`). The Webcam control requests access to the instructor's browser camera and sends downscaled still frames to the vision API; camera access requires permission and HTTPS (or localhost).
- **3D Head Pose Estimation**: Computes Yaw (turning left/right) and Pitch (looking down/up) using MediaPipe 3D landmark geometry and `cv2.solvePnP`. Approximates gaze alignment to evaluate visual focus.
- **Landmark Movement & Inactivity Detection**: Tracks Euclidean displacement of facial centroids frame-to-frame. Prolonged stillness flags an `inactivity_detected` alert (e.g. drooping head, sleepiness, or static photo spoofing).
- **Dual-Engine Architecture**: Automatically loads `MediaPipe FaceLandmarker` with local weights (`models/face_landmarker.task`), with automatic fallback to `OpenCV Haar Cascades` (`models/haarcascade_frontalface_default.xml`).
- **Vercel Face Detection**: The Python function explicitly bundles both Haar Cascade XML models so OpenCV face detection remains available in Vercel's serverless runtime, where the MediaPipe task model is excluded. The `/api/status` endpoint reports the active detector and any cascade initialization error.
- **Synthetic Classroom Simulation Mode**: When no physical camera is connected or during demonstrations, generates a high-fidelity synthetic classroom feed with animated student avatars, natural eye blinks, head turns, and staggered arrival times to demonstrate the sustained attendance verification transition in real time.

### EPIC 3: Audio Module (`utils/audio_module.py`)
- **Acoustic Signal Processing**: Evaluates audio buffers using:
  - **Root Mean Square (RMS) Energy**: Measures sound power and vocal amplitude.
  - **Zero-Crossing Rate (ZCR)**: Identifies high-frequency fricatives and spectral shifts characteristic of human speech.
- **Conversational Dynamics**: Combines RMS and ZCR to determine `speech_active` flags and computes an `audio_interaction_score` (0–100%).
- **Acoustic Presets & Live Microphone**: Supports both real browser microphone streaming via the Web Audio API and simulation presets (`Balanced`, `Active Discussion`, `Quiet Lecture`, `Q&A Interaction`).

### EPIC 4: Pedagogical AI Agent (`utils/genai_utils.py`)
- **Context-Aware Suggestions**: Takes fused telemetry and outputs one of four session states:
  - `Attention Drop Detected`
  - `Passive Listening / High Inactivity`
  - `Interaction Peak`
  - `Steady Session Pacing`
- **Hybrid AI Engine**:
  - **Online**: Uses `google-genai` with `gemini-2.0-flash` when `GEMINI_API_KEY` is present.
  - **Offline**: Seamlessly transitions to a deterministic **100% offline rule-based fallback agent** with a comprehensive pedagogical rule bank.
- **Audio TTS Generation**: Synthesizes audible advice through `pyttsx3` (Windows SAPI offline), `gTTS`, or browser speech synthesis.
- **Classroom Code Sandbox**: Generates concise Python programming challenges suited to the current engagement state and executes them safely in an isolated subprocess (`utils/code_executor.py`).

### EPIC 5: Multimodal Fusion Engine (`utils/fusion_engine.py`)
- **Fusion Formula**:
  $$\text{Fused Score} = (0.65 \times \text{Visual Score}) + (0.35 \times \text{Audio Score})$$
- **Inactivity Penalty**: If `inactivity_detected` is triggered, the overall score applies a penalizing coefficient:
  $$\text{Overall Engagement} = \max\left(5.0, \text{Fused Score} \times 0.72\right)$$
- **Rolling Time Series**: Maintains a 120-second rolling history for real-time Chart.js visualization.
- **Local Persistence**: Automatically saves timestamped session records to `storage/attendance_logs.json` and `storage/session_analytics.json`.

---

## 3. Installation & Quickstart

### Prerequisites
- Python 3.8+ (Python 3.11 recommended)
- Modern web browser (Chrome, Edge, Firefox)

### Step 1: Clone or Navigate to Project
```bash
cd "attendance management system"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment (Optional)
Copy `.env.example` to `.env` (already done by default):
```env
# Optional Gemini API Key (leaves blank for offline deterministic mode)
GEMINI_API_KEY=

# Sustained presence threshold (seconds)
SUSTAINED_ATTENDANCE_SECONDS=4.0

HOST=127.0.0.1
PORT=5000
```

### Step 4: Run the Application
```bash
python app.py
```

Open your browser to:
```
http://127.0.0.1:5000
```

> **Simulation Mode works out of the box!** No webcam or microphone required to test all features immediately.

---

## 4. API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Dashboard web application |
| `/video_feed` | `GET` | Multipart/x-mixed-replace JPEG video stream with HUD overlays |
| `/api/status` | `GET` | System health, active video source, and AI engine status |
| `/api/video_source` | `POST` | Switch feed: `{"source": "simulation" \| "webcam" \| "upload"}` |
| `/api/upload_video` | `POST` | Upload video file for classroom replay analysis |
| `/api/metrics` | `GET` | Real-time multimodal metrics, attendance summary, and AI advice |
| `/api/metrics_history` | `GET` | Rolling time-series data points for Chart.js |
| `/api/attendance` | `GET` | Participant attendance records and sustained duration states |
| `/api/attendance/reset` | `POST` | Clear current session and reset attendance logs |
| `/api/export_attendance` | `GET` | Download attendance records (`?format=csv` or `?format=json`) |
| `/api/export_analytics` | `GET` | Download session analytics (`?format=csv` or `?format=json`) |
| `/api/ai/suggestion` | `POST` | Force evaluate a fresh AI pedagogical suggestion |
| `/api/ai/tts` | `POST` | Synthesize TTS audio file for the current teaching suggestion |
| `/api/ai/generate_code` | `POST` | Generate interactive Python classroom code challenge |
| `/api/ai/execute_code` | `POST` | Safely execute Python snippet in isolated sandbox |
| `/api/ai/concept_diagram` | `POST` | Retrieve SVG pedagogical session concept diagram |
| `/api/audio/process_buffer` | `POST` | Stream microphone PCM audio buffer from client |
| `/api/audio/simulation_mode` | `POST` | Set acoustic simulation profile (`balanced`, `active_discussion`, etc.) |
| `/api/settings/gemini_key` | `POST` | Configure or update Gemini API key at runtime |

---

## 5. Privacy & Offline Guarantee

1. **Zero External Frame Transmission**: All computer vision models (`MediaPipe FaceLandmarker`, `OpenCV Haar Cascades`) execute strictly on the local machine CPU. No video streams are uploaded to any external server.
2. **Deterministic Offline Operation**: If no Gemini API key is configured or the machine is air-gapped, the pedagogical reasoning agent runs 100% locally using the built-in rule engine.
3. **Local Storage**: All session logs and attendance files remain in the local `storage/` directory.
