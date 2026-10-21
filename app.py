"""
Backend & API Gateway Layer (app.py)
Flask API Gateway integrating Computer Vision, Audio Signal Processing,
Multimodal Fusion Engine, GenAI Pedagogical Agent, and Safe Code Execution.
"""

import os
import time
import json
from flask import Flask, render_template, Response, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils.vision_module import VisionModule
from utils.audio_module import AudioModule
from utils.genai_utils import TeachingAssistantAgent
from utils.fusion_engine import FusionEngine
from utils.code_executor import CodeExecutor

# Load environment configuration
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploaded_files")
AUDIO_DIR = os.path.join(STORAGE_DIR, "generated_audio")
CODE_DIR = os.path.join(STORAGE_DIR, "generated_code")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CODE_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Configuration
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024  # 150MB max upload
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Initialize core modules
sustained_threshold = float(os.getenv("SUSTAINED_ATTENDANCE_SECONDS", "4.0"))
vision_module = VisionModule(models_dir=MODELS_DIR, sustained_threshold=sustained_threshold)
audio_module = AudioModule(sample_rate=16000)
genai_agent = TeachingAssistantAgent(api_key=os.getenv("GEMINI_API_KEY", ""), storage_dir=STORAGE_DIR)
fusion_engine = FusionEngine(storage_dir=STORAGE_DIR, max_history_points=120)
code_executor = CodeExecutor(timeout_seconds=4.0)

# Background tracking timer
last_ai_analysis_time = 0.0
cached_ai_result = {
    "session_state": "Steady Session Pacing",
    "ai_suggestion": "Session pacing is balanced. System initialized and monitoring classroom engagement.",
    "agent_source": "Deterministic Rule-Based Fallback Engine (Offline)"
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serves the primary Single-Page Application dashboard."""
    return render_template("index.html")


@app.route("/storage/<path:filename>")
def serve_storage(filename):
    """Serves generated audio, code, and uploaded assets securely from storage/."""
    return send_from_directory(STORAGE_DIR, filename)


def generate_mjpeg_stream():
    """Generates continuous multipart JPEG stream for the HUD video display."""
    import cv2
    while True:
        try:
            frame = vision_module.get_frame()
            if frame is None:
                time.sleep(0.04)
                continue

            # Encode as JPEG
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                time.sleep(0.03)
                continue

            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.033)  # Approx 30 FPS target
        except Exception as e:
            time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    """Live video feed endpoint providing multipart/x-mixed-replace JPEG stream."""
    return Response(generate_mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status", methods=["GET"])
def get_system_status():
    """System health check and module operational status."""
    return jsonify({
        "status": "online",
        "video_source": vision_module.video_source_mode,
        "detector_in_use": vision_module.detector_type,
        "gemini_active": genai_agent.is_gemini_available,
        "active_ai_engine": genai_agent.last_used_source,
        "sustained_threshold_seconds": vision_module.sustained_threshold,
        "audio_sim_mode": audio_module.sim_mode,
        "timestamp": time.time()
    })


@app.route("/api/video_source", methods=["POST"])
def set_video_source():
    """Switches the video input source between 'simulation', 'webcam', and 'upload'."""
    data = request.get_json() or {}
    source_mode = data.get("source", "simulation").lower()

    if source_mode not in ["simulation", "webcam", "upload"]:
        return jsonify({"success": False, "error": "Invalid source mode specified"}), 400

    video_path = None
    if source_mode == "upload":
        video_path = data.get("video_path")
        if not video_path and vision_module.uploaded_video_path:
            video_path = vision_module.uploaded_video_path

    success = vision_module.set_source_mode(source_mode, video_path)
    return jsonify({
        "success": success,
        "active_source": vision_module.video_source_mode,
        "detector_in_use": vision_module.detector_type
    })


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Uploads a recorded classroom video file and attaches it as the active video feed."""
    if "video_file" not in request.files:
        return jsonify({"success": False, "error": "No video file found in request"}), 400

    file = request.files["video_file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        save_name = f"upload_{timestamp}_{filename}"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        file.save(save_path)

        # Switch vision module to uploaded feed
        success = vision_module.set_source_mode("upload", save_path)
        return jsonify({
            "success": success,
            "filename": save_name,
            "file_path": save_path,
            "message": "Video successfully uploaded and set as active feed."
        })

    return jsonify({"success": False, "error": "Unsupported file format. Please upload MP4, AVI, or MKV."}), 400


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Primary real-time telemetry endpoint.
    Aggregates vision, audio, multimodal fusion, and cached AI suggestion.
    """
    global last_ai_analysis_time, cached_ai_result

    # 1. Fetch module snapshots
    vision_data = vision_module.get_latest_metrics()
    audio_data = audio_module.get_latest_metrics()
    attendance_list = vision_module.get_attendance_list()

    # 2. Update AI pedagogical analysis periodically (every 4 seconds) or on state shift
    now = time.time()
    if now - last_ai_analysis_time >= 4.0:
        latest_fused = fusion_engine.get_latest_metrics()
        cached_ai_result = genai_agent.analyze_session(
            visual_score=vision_data.get("visual_engagement_score", 75.0),
            audio_score=audio_data.get("audio_interaction_score", 50.0),
            fused_score=latest_fused.get("overall_engagement_score", 75.0),
            inactivity=vision_data.get("inactivity_detected", False),
            verified_count=vision_data.get("verified_count", 0),
            trend=latest_fused.get("trend", "steady")
        )
        last_ai_analysis_time = now

    # 3. Fuse signals
    fused_snapshot = fusion_engine.fuse_metrics(
        vision_metrics=vision_data,
        audio_metrics=audio_data,
        attendance_list=attendance_list,
        session_state=cached_ai_result.get("session_state")
    )

    return jsonify({
        "fusion": fused_snapshot,
        "vision": vision_data,
        "audio": audio_data,
        "ai_intelligence": cached_ai_result,
        "attendance_summary": {
            "total_detected": len(attendance_list),
            "verified_count": sum(1 for a in attendance_list if a.get("is_verified")),
            "sustained_threshold_seconds": vision_module.sustained_threshold
        },
        "timestamp": now
    })


@app.route("/api/metrics_history", methods=["GET"])
def get_metrics_history():
    """Returns rolling time-series points for the Chart.js visualization."""
    history = fusion_engine.get_time_series_history()
    return jsonify({
        "success": True,
        "count": len(history),
        "history": history
    })


@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    """Returns the live participant attendance records with sustained durations and verification states."""
    records = vision_module.get_attendance_list()
    return jsonify({
        "success": True,
        "count": len(records),
        "sustained_threshold_seconds": vision_module.sustained_threshold,
        "records": records
    })


@app.route("/api/attendance/reset", methods=["POST"])
def reset_attendance():
    """Resets the live attendance tracker and fusion engine for a new classroom session."""
    vision_module.reset_attendance()
    fusion_engine.reset_session()
    return jsonify({
        "success": True,
        "message": "Classroom session and attendance records have been successfully reset."
    })


@app.route("/api/export_attendance", methods=["GET"])
def export_attendance():
    """Exports attendance records in CSV or JSON format for download."""
    export_format = request.args.get("format", "csv").lower()
    attendance_records = vision_module.get_attendance_list()

    if export_format == "json":
        json_data = json.dumps(attendance_records, indent=2)
        return Response(
            json_data,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=attendance_records.json"}
        )

    csv_text = fusion_engine.export_attendance_csv(attendance_records)
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=attendance_records.csv"}
    )


@app.route("/api/export_analytics", methods=["GET"])
def export_analytics():
    """Exports session analytics time-series data in CSV or JSON format."""
    export_format = request.args.get("format", "csv").lower()
    history = fusion_engine.get_time_series_history()

    if export_format == "json":
        json_data = json.dumps(history, indent=2)
        return Response(
            json_data,
            mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=session_analytics.json"}
        )

    csv_text = fusion_engine.export_analytics_csv()
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=session_analytics.csv"}
    )


@app.route("/api/ai/suggestion", methods=["POST"])
def trigger_ai_suggestion():
    """Forces an immediate AI pedagogical suggestion evaluation."""
    global cached_ai_result, last_ai_analysis_time

    vision_data = vision_module.get_latest_metrics()
    audio_data = audio_module.get_latest_metrics()
    fused_data = fusion_engine.get_latest_metrics()

    cached_ai_result = genai_agent.analyze_session(
        visual_score=vision_data.get("visual_engagement_score", 75.0),
        audio_score=audio_data.get("audio_interaction_score", 50.0),
        fused_score=fused_data.get("overall_engagement_score", 75.0),
        inactivity=vision_data.get("inactivity_detected", False),
        verified_count=vision_data.get("verified_count", 0),
        trend=fused_data.get("trend", "steady"),
        force_refresh=True
    )
    last_ai_analysis_time = time.time()
    return jsonify(cached_ai_result)


@app.route("/api/ai/tts", methods=["POST"])
def generate_tts_audio():
    """Synthesizes speech audio for the provided pedagogical suggestion text."""
    data = request.get_json() or {}
    text = data.get("text") or cached_ai_result.get("ai_suggestion") or "Engagement is optimal."

    tts_result = genai_agent.generate_speech_audio(text)
    return jsonify(tts_result)


@app.route("/api/ai/generate_code", methods=["POST"])
def generate_classroom_code():
    """Generates an educational quiz, coding challenge, or interactive verification script."""
    data = request.get_json() or {}
    topic = data.get("topic", "Python Fundamentals & Data Processing")
    session_state = cached_ai_result.get("session_state", "Steady Session Pacing")

    code_res = genai_agent.generate_classroom_code_exercise(topic=topic, session_state=session_state)
    return jsonify(code_res)


@app.route("/api/ai/execute_code", methods=["POST"])
def execute_code_snippet():
    """Safely executes the provided Python code snippet in an isolated sandbox subprocess."""
    data = request.get_json() or {}
    code_str = data.get("code", "")

    result = code_executor.execute_python(code_str)
    return jsonify(result)


@app.route("/api/ai/concept_diagram", methods=["POST"])
def get_concept_diagram():
    """Returns SVG concept diagram summarizing engagement session dynamics."""
    fused = fusion_engine.get_latest_metrics()
    state = cached_ai_result.get("session_state", "Steady Session Pacing")
    score = fused.get("overall_engagement_score", 75.0)

    diagram_res = genai_agent.generate_visual_concept_summary(session_state=state, engagement_score=score)
    return jsonify(diagram_res)


@app.route("/api/audio/process_buffer", methods=["POST"])
def process_client_audio_buffer():
    """Accepts base64-encoded PCM audio buffers sent from the frontend microphone."""
    data = request.get_json() or {}
    audio_b64 = data.get("audio_buffer")

    if not audio_b64:
        # Step simulation if empty
        metrics = audio_module.simulate_classroom_audio_step()
        return jsonify({"success": True, "metrics": metrics, "source": "simulation"})

    metrics = audio_module.process_base64_audio(audio_b64)
    return jsonify({"success": True, "metrics": metrics, "source": "microphone"})


@app.route("/api/audio/simulation_mode", methods=["POST"])
def set_audio_sim_mode():
    """Sets the audio acoustic simulation profile: balanced, active_discussion, lecture_silence, question_answer."""
    data = request.get_json() or {}
    mode = data.get("mode", "balanced")
    audio_module.set_simulation_mode(mode)
    return jsonify({
        "success": True,
        "mode": audio_module.sim_mode,
        "message": f"Audio simulation preset changed to '{mode}'."
    })


@app.route("/api/settings/gemini_key", methods=["POST"])
def set_gemini_key():
    """Configures the Gemini API Key dynamically from the instructor dashboard."""
    data = request.get_json() or {}
    api_key = data.get("api_key", "").strip()

    genai_agent.set_api_key(api_key)
    return jsonify({
        "success": True,
        "gemini_active": genai_agent.is_gemini_available,
        "active_engine": genai_agent.last_used_source
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"================================================================")
    print(f"  MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT")
    print(f"  Privacy-Preserving Classroom Intelligence System")
    print(f"  Live at: http://{host}:{port}")
    print(f"================================================================")
    app.run(host=host, port=port, debug=False, threaded=True)
