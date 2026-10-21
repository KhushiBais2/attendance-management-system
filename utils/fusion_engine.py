"""
Multimodal Fusion & Analytics Engine (EPIC 5)
Fuses visual telemetry (65%) and audio interaction (35%) with inactivity penalty.
Maintains a rolling time-series window and persists logs to JSON and CSV formats.
"""

import os
import json
import csv
import io
import time
import threading
from typing import List, Dict, Any


class FusionEngine:
    """
    Multimodal Engagement Fusion & Session Analytics Aggregator.
    Fuses vision metrics, audio signals, and sustained attendance into unified pedagogical intelligence.
    """

    def __init__(self, storage_dir: str = "storage", max_history_points: int = 120):
        self.storage_dir = storage_dir
        self.max_history_points = max_history_points
        self.lock = threading.Lock()

        self.attendance_file = os.path.join(storage_dir, "attendance_logs.json")
        self.analytics_file = os.path.join(storage_dir, "session_analytics.json")
        os.makedirs(storage_dir, exist_ok=True)

        # Rolling history of multimodal data points (1 sample per second)
        self.time_series_history: List[Dict[str, Any]] = []

        # Current fused state
        self.current_state = {
            "overall_engagement_score": 75.0,
            "visual_engagement_score": 75.0,
            "audio_interaction_score": 50.0,
            "inactivity_detected": False,
            "participant_count": 0,
            "verified_count": 0,
            "session_state": "Steady Session Pacing",
            "trend": "steady",
            "timestamp": time.time()
        }

        # Auto-persistence timer
        self.last_persisted = 0.0
        self.persist_interval_seconds = 5.0

        self._init_storage_files()

    def _init_storage_files(self):
        """Ensures local storage JSON files exist and are valid JSON."""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

        if not os.path.exists(self.analytics_file):
            with open(self.analytics_file, "w", encoding="utf-8") as f:
                json.dump({
                    "session_start": time.time(),
                    "session_id": f"SESS-{int(time.time())}",
                    "snapshots": []
                }, f, indent=2)

    def fuse_metrics(self, vision_metrics: dict, audio_metrics: dict,
                     attendance_list: list, session_state: str = None) -> dict:
        """
        Fuses visual attention and audio interaction using 65/35 weighting with inactivity penalty.
        Formula:
            Base Fused = (0.65 * VisualScore) + (0.35 * AudioScore)
            If Inactivity Detected:
                Overall Engagement = max(5.0, Base Fused * 0.72)
            Else:
                Overall Engagement = Base Fused
        """
        v_score = float(vision_metrics.get("visual_engagement_score", 70.0))
        a_score = float(audio_metrics.get("audio_interaction_score", 50.0))
        inactivity = bool(vision_metrics.get("inactivity_detected", False))
        p_count = int(vision_metrics.get("participant_count", 0))
        v_count = int(vision_metrics.get("verified_count", 0))

        # Weighted calculation
        raw_fused = (0.65 * v_score) + (0.35 * a_score)

        # Inactivity penalty: penalize static, unengaged posture
        if inactivity:
            overall = max(5.0, raw_fused * 0.72)
        else:
            overall = raw_fused

        overall = round(min(100.0, max(0.0, overall)), 1)
        v_score = round(min(100.0, max(0.0, v_score)), 1)
        a_score = round(min(100.0, max(0.0, a_score)), 1)

        now = time.time()

        with self.lock:
            # Determine engagement trend based on recent history
            trend = "steady"
            if len(self.time_series_history) >= 5:
                recent_avg = sum(p["overall_engagement_score"] for p in self.time_series_history[-5:]) / 5.0
                diff = overall - recent_avg
                if diff > 3.5:
                    trend = "rising"
                elif diff < -3.5:
                    trend = "falling"

            snapshot = {
                "timestamp": now,
                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                "overall_engagement_score": overall,
                "visual_engagement_score": v_score,
                "audio_interaction_score": a_score,
                "inactivity_detected": inactivity,
                "participant_count": p_count,
                "verified_count": v_count,
                "session_state": session_state or self.current_state.get("session_state", "Steady Session Pacing"),
                "trend": trend,
                "head_yaw": vision_metrics.get("head_pose_yaw_avg", 0.0),
                "head_pitch": vision_metrics.get("head_pose_pitch_avg", 0.0),
                "rms_energy": audio_metrics.get("rms_energy", 0.0),
                "zcr": audio_metrics.get("zero_crossing_rate", 0.0)
            }

            self.current_state = snapshot
            self.time_series_history.append(snapshot)

            if len(self.time_series_history) > self.max_history_points:
                self.time_series_history.pop(0)

            # Periodic background persistence
            if now - self.last_persisted >= self.persist_interval_seconds:
                self._persist_to_storage(attendance_list)
                self.last_persisted = now

            return dict(self.current_state)

    def _persist_to_storage(self, attendance_list: list):
        """Writes current attendance and session analytics to storage JSON files."""
        try:
            # 1. Update attendance_logs.json
            with open(self.attendance_file, "w", encoding="utf-8") as f:
                json.dump(attendance_list, f, indent=2)

            # 2. Update session_analytics.json
            summary = {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_snapshots": len(self.time_series_history),
                "current_overall_engagement": self.current_state.get("overall_engagement_score", 0.0),
                "verified_attendees": self.current_state.get("verified_count", 0),
                "recent_trend": self.current_state.get("trend", "steady"),
                "time_series": self.time_series_history[-60:]  # Keep last 60 points in summary
            }
            with open(self.analytics_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            print(f"[FusionEngine] Persistence error: {e}")

    def get_latest_metrics(self) -> dict:
        """Returns the most recent fused metric snapshot."""
        with self.lock:
            return dict(self.current_state)

    def get_time_series_history(self) -> list:
        """Returns the time-series history for Chart.js plotting."""
        with self.lock:
            return list(self.time_series_history)

    def export_attendance_csv(self, attendance_list: list) -> str:
        """Exports verified attendance records as formatted CSV text."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Participant ID", "Student Name", "Verification Status",
                         "Sustained Presence (s)", "First Seen", "Last Seen", "Verified At"])

        for record in attendance_list:
            writer.writerow([
                record.get("id", ""),
                record.get("label", ""),
                record.get("status", ""),
                record.get("sustained_duration", 0.0),
                record.get("first_seen", ""),
                record.get("last_seen", ""),
                record.get("verified_at", "N/A")
            ])

        return output.getvalue()

    def export_analytics_csv(self) -> str:
        """Exports rolling analytics metrics as formatted CSV text."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Time", "Overall Engagement (%)",
                         "Visual Engagement (%)", "Audio Interaction (%)",
                         "Inactivity Detected", "Participants", "Verified Count", "Session State"])

        with self.lock:
            for s in self.time_series_history:
                writer.writerow([
                    s.get("timestamp", 0.0),
                    s.get("time_str", ""),
                    s.get("overall_engagement_score", 0.0),
                    s.get("visual_engagement_score", 0.0),
                    s.get("audio_interaction_score", 0.0),
                    s.get("inactivity_detected", False),
                    s.get("participant_count", 0),
                    s.get("verified_count", 0),
                    s.get("session_state", "")
                ])

        return output.getvalue()

    def reset_session(self):
        """Clears rolling time series and re-initializes session logs."""
        with self.lock:
            self.time_series_history.clear()
            self.current_state["overall_engagement_score"] = 75.0
            self.current_state["verified_count"] = 0
            self.current_state["participant_count"] = 0
            self._init_storage_files()
