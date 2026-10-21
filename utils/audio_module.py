"""
Audio Signal Processing Module (EPIC 3)
Extracts Root Mean Square (RMS) energy, Zero-Crossing Rate (ZCR),
speech activity flags, and conversational interaction dynamics.
Includes live buffer decoding and synthetic classroom audio simulation.
"""

import math
import time
import base64
import random
import threading
import numpy as np


class AudioModule:
    """
    Multimodal Audio Analysis Engine for Classroom Environments.
    Analyzes RMS energy and ZCR from live microphone buffers or dynamic acoustic simulation.
    """

    def __init__(self, sample_rate: int = 16000, frame_duration_sec: float = 0.5):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_sec)
        self.lock = threading.Lock()

        # Acoustic thresholds
        self.rms_silence_threshold = 0.015
        self.rms_speech_threshold = 0.040
        self.zcr_speech_min = 0.02
        self.zcr_speech_max = 0.45

        # Current state
        self.current_rms = 0.0
        self.current_zcr = 0.0
        self.speech_active = False
        self.audio_interaction_score = 0.0  # 0.0 - 100.0%
        self.last_updated = time.time()

        # Simulation dynamics state
        self.sim_mode = "balanced"  # balanced, active_discussion, lecture_silence, question_answer
        self.sim_phase = 0.0

        # Rolling history of scores for dynamic smoothing
        self.history_scores = []
        self.max_history = 30

    def compute_rms(self, audio_buffer: np.ndarray) -> float:
        """Computes Root Mean Square (RMS) energy of an audio buffer."""
        if len(audio_buffer) == 0:
            return 0.0
        # Prevent overflow by ensuring float dtype
        buffer_float = audio_buffer.astype(np.float32)
        rms = np.sqrt(np.mean(buffer_float ** 2))
        return float(rms)

    def compute_zcr(self, audio_buffer: np.ndarray) -> float:
        """
        Computes Zero-Crossing Rate (ZCR):
        Number of sign changes along the audio buffer normalized by sample length.
        """
        if len(audio_buffer) < 2:
            return 0.0
        buffer_float = audio_buffer.astype(np.float32)
        zero_crossings = np.mean(np.abs(np.diff(np.sign(buffer_float)))) / 2.0
        return float(zero_crossings)

    def process_buffer(self, audio_buffer: np.ndarray) -> dict:
        """
        Analyzes a raw audio buffer (1D float/int array) and computes multimodal audio metrics.
        """
        if audio_buffer is None or len(audio_buffer) == 0:
            return self.get_latest_metrics()

        # Normalize if integer array (e.g., 16-bit PCM)
        if np.issubdtype(audio_buffer.dtype, np.integer):
            max_val = np.iinfo(audio_buffer.dtype).max
            audio_buffer = audio_buffer.astype(np.float32) / float(max_val)
        else:
            audio_buffer = audio_buffer.astype(np.float32)

        rms = self.compute_rms(audio_buffer)
        zcr = self.compute_zcr(audio_buffer)

        # Detect speech activity based on dual criteria: energy & spectral crossing
        speech_active = (rms >= self.rms_speech_threshold) and (self.zcr_speech_min <= zcr <= self.zcr_speech_max)

        # Compute audio interaction score (0 - 100%)
        # Normalizes RMS energy logarithmically/linearly into engagement dynamics
        # Active classroom discussion typically has varying RMS (0.04 - 0.25) and healthy ZCR
        normalized_energy = min(1.0, max(0.0, (rms - self.rms_silence_threshold) / (0.20 - self.rms_silence_threshold)))
        spectral_confidence = 1.0 if (self.zcr_speech_min <= zcr <= self.zcr_speech_max) else 0.4

        raw_interaction = (normalized_energy * 0.75 + (1.0 if speech_active else 0.1) * 0.25) * spectral_confidence * 100.0
        raw_interaction = min(100.0, max(0.0, raw_interaction))

        with self.lock:
            self.current_rms = round(rms, 4)
            self.current_zcr = round(zcr, 4)
            self.speech_active = bool(speech_active)

            # Exponential smoothing for interaction score
            alpha = 0.35
            self.audio_interaction_score = round(alpha * raw_interaction + (1.0 - alpha) * self.audio_interaction_score, 1)

            self.history_scores.append(self.audio_interaction_score)
            if len(self.history_scores) > self.max_history:
                self.history_scores.pop(0)

            self.last_updated = time.time()

        return self.get_latest_metrics()

    def process_base64_audio(self, base64_str: str, sample_width: int = 2) -> dict:
        """
        Decodes base64-encoded PCM audio sent from the browser microphone.
        """
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(base64_str)

            if sample_width == 2:
                # 16-bit signed integer PCM
                audio_np = np.frombuffer(raw_bytes, dtype=np.int16)
            elif sample_width == 4:
                # 32-bit float PCM
                audio_np = np.frombuffer(raw_bytes, dtype=np.float32)
            else:
                audio_np = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0

            return self.process_buffer(audio_np)
        except Exception:
            # Fall back to simulation if parsing fails
            return self.simulate_classroom_audio_step()

    def set_simulation_mode(self, mode: str):
        """Sets simulation mode: 'balanced', 'active_discussion', 'lecture_silence', 'question_answer'."""
        with self.lock:
            if mode in ["balanced", "active_discussion", "lecture_silence", "question_answer"]:
                self.sim_mode = mode

    def simulate_classroom_audio_step(self) -> dict:
        """
        Generates realistic synthetic classroom acoustic signals when no microphone is active.
        Simulates teacher talk, student questions, collaborative murmur, or quiet reflection.
        """
        self.sim_phase += 0.2
        t = time.time()

        # Dynamic profile based on simulation mode
        if self.sim_mode == "active_discussion":
            # Multi-party chatter: higher RMS bursts, rich frequency shifts
            base_rms = 0.12 + 0.06 * math.sin(t * 1.5) + random.uniform(-0.03, 0.04)
            base_zcr = 0.18 + 0.05 * math.cos(t * 2.1) + random.uniform(-0.02, 0.03)
            speech_prob = 0.85
            target_interaction = 78.0 + random.uniform(-10.0, 15.0)

        elif self.sim_mode == "lecture_silence":
            # Very quiet individual study or heavy attention drop
            base_rms = 0.012 + random.uniform(0.001, 0.008)
            base_zcr = 0.04 + random.uniform(-0.01, 0.02)
            speech_prob = 0.08
            target_interaction = 14.0 + random.uniform(-5.0, 8.0)

        elif self.sim_mode == "question_answer":
            # Periodic bursts: teacher asks, student answers, brief pause
            cycle = (t % 14.0)
            if cycle < 4.0:
                # Teacher speaking
                base_rms = 0.08 + random.uniform(-0.02, 0.02)
                base_zcr = 0.14 + random.uniform(-0.02, 0.02)
                speech_prob = 0.75
                target_interaction = 62.0
            elif cycle < 6.0:
                # Brief wait-time pause
                base_rms = 0.018 + random.uniform(0.0, 0.01)
                base_zcr = 0.05
                speech_prob = 0.15
                target_interaction = 28.0
            elif cycle < 11.0:
                # Student response & applause/murmur
                base_rms = 0.095 + random.uniform(-0.02, 0.03)
                base_zcr = 0.16 + random.uniform(-0.02, 0.02)
                speech_prob = 0.80
                target_interaction = 74.0
            else:
                base_rms = 0.04 + random.uniform(-0.01, 0.01)
                base_zcr = 0.09
                speech_prob = 0.40
                target_interaction = 48.0

        else:  # balanced
            # Standard classroom rhythm
            mod = math.sin(t * 0.4)
            base_rms = 0.055 + 0.03 * mod + random.uniform(-0.015, 0.02)
            base_zcr = 0.12 + 0.03 * math.cos(t * 0.6) + random.uniform(-0.02, 0.02)
            speech_prob = 0.60
            target_interaction = 52.0 + 18.0 * mod + random.uniform(-6.0, 6.0)

        rms = max(0.005, float(base_rms))
        zcr = max(0.01, min(0.5, float(base_zcr)))
        speech_active = (random.random() < speech_prob) and (rms >= self.rms_silence_threshold)

        with self.lock:
            self.current_rms = round(rms, 4)
            self.current_zcr = round(zcr, 4)
            self.speech_active = bool(speech_active)

            # Smoothly transition score toward target
            alpha = 0.25
            self.audio_interaction_score = round(
                alpha * target_interaction + (1.0 - alpha) * self.audio_interaction_score, 1
            )
            self.audio_interaction_score = min(100.0, max(0.0, self.audio_interaction_score))

            self.history_scores.append(self.audio_interaction_score)
            if len(self.history_scores) > self.max_history:
                self.history_scores.pop(0)

            self.last_updated = time.time()

        return self.get_latest_metrics()

    def get_latest_metrics(self) -> dict:
        """Returns the latest audio analysis metrics snapshot."""
        with self.lock:
            # If metrics haven't been updated in 1.5 seconds, auto-step simulation
            if time.time() - self.last_updated > 1.5:
                # Release lock to avoid deadlock inside simulate_classroom_audio_step
                pass

        if time.time() - self.last_updated > 1.5:
            return self.simulate_classroom_audio_step()

        with self.lock:
            return {
                "rms_energy": self.current_rms,
                "zero_crossing_rate": self.current_zcr,
                "speech_active": self.speech_active,
                "audio_interaction_score": self.audio_interaction_score,
                "simulation_mode": self.sim_mode,
                "timestamp": time.time()
            }
