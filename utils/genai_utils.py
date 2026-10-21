"""
AI Service Layer & Pedagogical Agent (EPIC 4)
Integrates google-genai (gemini-2.0-flash) with an intelligent, deterministic 100% offline
pedagogical rule-based fallback agent. Includes Text-to-Speech (TTS), Code Generation,
and visual concept diagram synthesis hooks.
"""

import os
import time
import random
import threading
import wave
import struct
from dotenv import load_dotenv

load_dotenv()

# Attempt to load pyttsx3 or gTTS for server-side TTS
HAVE_PYTTSX3 = False
try:
    import pyttsx3
    HAVE_PYTTSX3 = True
except Exception:
    HAVE_PYTTSX3 = False

HAVE_GTTS = False
try:
    from gtts import gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False

# Attempt to load google-genai
HAVE_GOOGLE_GENAI = False
try:
    from google import genai
    from google.genai import types
    HAVE_GOOGLE_GENAI = True
except Exception:
    HAVE_GOOGLE_GENAI = False


class OfflinePedagogicalEngine:
    """
    Deterministic rule-based pedagogical reasoning engine.
    Ensures 100% system functionality even when offline, air-gapped, or without API keys.
    """

    SUGGESTIONS_ATTENTION_DROP = [
        "Study focus is dropping on the current derivation. Introduce a 60-second quick concept checkpoint or flashcard quiz.",
        "Gaze deviation detected across multiple student desks. Pause the lecture and pose a quick think-pair-share problem.",
        "Students are losing visual focus on the slides. Transition to a live board walkthrough or show a practical diagram.",
        "Noticeable slump in visual focus. Have students write down their single biggest takeaway from the past 10 minutes.",
        "Attention has dropped below 48%. Announce a 2-minute stretch break or ask a volunteer to summarize the key formula."
    ]

    SUGGESTIONS_PASSIVE_LISTENING = [
        "Class is in passive listening mode without questions. Prompt students to critique a deliberate misconception on the board.",
        "Learners are silently absorbing with zero verbal dialogue. Ask an open-ended 'Why?' question to stimulate student reasoning.",
        "High stillness and low vocal participation detected. Invite a student to explain the core principle to their neighbor.",
        "Passive listening alert: Transition from didactic lecturing to an active code walkthrough or problem drill.",
        "Student focus is quiet but static. Give learners 90 seconds to solve a small practice step independently before continuing."
    ]

    SUGGESTIONS_INTERACTION_PEAK = [
        "Classroom discussion and student energy are peaking! Channel this momentum into a timed collaborative problem-solving challenge.",
        "High verbal participation detected. Capture the active student insights on the board and synthesize opposing arguments.",
        "Active student dialogue is strong! Acknowledge two student answers, then formalize the core theorem before energy wanes.",
        "Vibrant debate detected! Challenge a student to connect their peer's insight to the previous study module.",
        "Great collaborative momentum! Transition into an applied coding exercise to solidify understanding."
    ]

    SUGGESTIONS_STEADY_PACING = [
        "Lecture pacing is balanced and attentive. Students are actively following along and taking notes.",
        "Steady study engagement observed. Reinforce the current theorem with a real-world edge case or practical example.",
        "Learner focus is stable. Optimal moment to introduce the next progressive complexity tier in the syllabus.",
        "Visual attention and classroom dialogue are in optimal harmony. Continue the planned lecture sequence.",
        "Consistent study attentiveness detected across learners. Acknowledge student focus before advancing to the next unit."
    ]

    @classmethod
    def evaluate(cls, visual_score: float, audio_score: float, fused_score: float,
                 inactivity: bool, verified_count: int, trend: str = "steady") -> tuple:
        """
        Deterministic rule-based classifier that maps multimodal signals to session states and suggestions.
        Returns: (session_state: str, suggestion: str)
        """
        # Rule 1: Attention Drop Detected
        if (visual_score < 48.0) or (trend == "falling" and visual_score < 60.0):
            state = "Attention Drop Detected"
            suggestion = random.choice(cls.SUGGESTIONS_ATTENTION_DROP)
            return state, suggestion

        # Rule 2: Interaction Peak
        if audio_score >= 65.0:
            state = "Interaction Peak"
            suggestion = random.choice(cls.SUGGESTIONS_INTERACTION_PEAK)
            return state, suggestion

        # Rule 3: Passive Listening / High Inactivity
        if inactivity or (audio_score < 24.0 and visual_score < 72.0):
            state = "Passive Listening / High Inactivity"
            suggestion = random.choice(cls.SUGGESTIONS_PASSIVE_LISTENING)
            return state, suggestion

        # Rule 4: Default Steady Session Pacing
        state = "Steady Session Pacing"
        suggestion = random.choice(cls.SUGGESTIONS_STEADY_PACING)
        return state, suggestion


class TeachingAssistantAgent:
    """
    Multimodal AI Pedagogical Agent.
    Leverages Gemini 2.0 Flash when an API key is present,
    and falls back to the deterministic offline pedagogical engine seamlessly.
    """

    def __init__(self, api_key: str = None, storage_dir: str = "storage"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.storage_dir = storage_dir
        self.audio_dir = os.path.join(storage_dir, "generated_audio")
        self.code_dir = os.path.join(storage_dir, "generated_code")
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)

        self.client = None
        self.is_gemini_available = False
        self._init_gemini_client()

        # Cache for throttling repetitive API requests (minimum 4 seconds between remote calls)
        self.last_api_call = 0.0
        self.min_interval_seconds = 4.0
        self.cached_state = "Steady Session Pacing"
        self.cached_suggestion = "Session pacing is balanced. System initialized and monitoring classroom engagement."
        self.last_used_source = "Offline Rule-Based Fallback Engine"

    def _init_gemini_client(self):
        """Initializes the google-genai Client if an API key is configured."""
        if HAVE_GOOGLE_GENAI and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.is_gemini_available = True
                print("[TeachingAssistantAgent] Gemini 2.0 Flash client initialized.")
            except Exception as e:
                print(f"[TeachingAssistantAgent] Failed to initialize Gemini client: {e}. Falling back to offline engine.")
                self.is_gemini_available = False
        else:
            self.is_gemini_available = False
            print("[TeachingAssistantAgent] Running in Offline Deterministic Mode.")

    def set_api_key(self, api_key: str):
        """Allows dynamic configuration of the Gemini API Key from the UI."""
        self.api_key = api_key.strip()
        self._init_gemini_client()

    def analyze_session(self, visual_score: float, audio_score: float, fused_score: float,
                        inactivity: bool, verified_count: int, trend: str = "steady",
                        force_refresh: bool = False) -> dict:
        """
        Analyzes multimodal metrics and generates context-aware teaching suggestions.
        """
        now = time.time()
        # Evaluate offline rule engine first for instant classification
        rule_state, rule_suggestion = OfflinePedagogicalEngine.evaluate(
            visual_score, audio_score, fused_score, inactivity, verified_count, trend
        )

        # If Gemini is not enabled, or throttled, return offline output
        if not self.is_gemini_available or (now - self.last_api_call < self.min_interval_seconds and not force_refresh):
            self.cached_state = rule_state
            self.cached_suggestion = rule_suggestion
            self.last_used_source = "Deterministic Rule-Based Fallback Engine (Offline)"
            return {
                "session_state": self.cached_state,
                "ai_suggestion": self.cached_suggestion,
                "agent_source": self.last_used_source,
                "timestamp": now
            }

        # Query Gemini 2.0 Flash for context-rich suggestion
        try:
            prompt = f"""You are an elite, empathetic classroom AI teaching assistant analyzing real-time multimodal classroom telemetry.
Telemetry:
- Visual Engagement Score: {visual_score:.1f}% (Optimal > 70%)
- Audio Interaction Score: {audio_score:.1f}% (Classroom discussion/speech activity)
- Fused Overall Engagement: {fused_score:.1f}%
- Inactivity / Motionless Alert: {"YES" if inactivity else "NO"}
- Verified Attendees: {verified_count}
- Engagement Trend: {trend}
- Rule-based State Hypothesis: {rule_state}

Task:
1. Confirm or refine the Session State into EXACTLY ONE of:
   ["Attention Drop Detected", "Passive Listening / High Inactivity", "Interaction Peak", "Steady Session Pacing"]
2. Provide a single, immediate, actionable, non-intrusive teaching suggestion (1-2 sentences) directly to the instructor.

Format response strictly as:
STATE: <One of the 4 exact states>
SUGGESTION: <Actionable advice>"""

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            text = response.text if hasattr(response, 'text') else str(response)
            parsed_state = rule_state
            parsed_suggestion = rule_suggestion

            for line in text.split("\n"):
                line_str = line.strip()
                if line_str.startswith("STATE:"):
                    candidate = line_str.replace("STATE:", "").strip()
                    if candidate in ["Attention Drop Detected", "Passive Listening / High Inactivity", "Interaction Peak", "Steady Session Pacing"]:
                        parsed_state = candidate
                elif line_str.startswith("SUGGESTION:"):
                    parsed_suggestion = line_str.replace("SUGGESTION:", "").strip()

            self.cached_state = parsed_state
            self.cached_suggestion = parsed_suggestion
            self.last_used_source = "Google GenAI (gemini-2.0-flash)"
            self.last_api_call = now

            return {
                "session_state": self.cached_state,
                "ai_suggestion": self.cached_suggestion,
                "agent_source": self.last_used_source,
                "timestamp": now
            }

        except Exception as e:
            print(f"[TeachingAssistantAgent] Gemini generation error: {e}. Falling back to offline engine.")
            self.cached_state = rule_state
            self.cached_suggestion = rule_suggestion
            self.last_used_source = f"Offline Fallback (Gemini Error: {type(e).__name__})"
            return {
                "session_state": self.cached_state,
                "ai_suggestion": self.cached_suggestion,
                "agent_source": self.last_used_source,
                "timestamp": now
            }

    def generate_speech_audio(self, text: str) -> dict:
        """
        Synthesizes TTS audio for the given pedagogical suggestion.
        Uses pyttsx3 (offline SAPI), gTTS, or deterministic WAV chime fallback.
        Returns file path and public URL.
        """
        timestamp = int(time.time() * 1000)
        filename = f"suggestion_{timestamp}.wav"
        filepath = os.path.join(self.audio_dir, filename)

        # 1. Try pyttsx3 (offline Windows TTS)
        if HAVE_PYTTSX3:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 165)
                engine.save_to_file(text, filepath)
                engine.runAndWait()
                if os.path.exists(filepath) and os.path.getsize(filepath) > 500:
                    return {
                        "success": True,
                        "file_path": filepath,
                        "audio_url": f"/storage/generated_audio/{filename}",
                        "tts_engine": "pyttsx3 (Offline SAPI)"
                    }
            except Exception as e:
                print(f"[TeachingAssistantAgent] pyttsx3 error: {e}")

        # 2. Try gTTS (online Google TTS)
        if HAVE_GTTS:
            try:
                mp3_filename = f"suggestion_{timestamp}.mp3"
                mp3_path = os.path.join(self.audio_dir, mp3_filename)
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(mp3_path)
                return {
                    "success": True,
                    "file_path": mp3_path,
                    "audio_url": f"/storage/generated_audio/{mp3_filename}",
                    "tts_engine": "gTTS (Google TTS)"
                }
            except Exception as e:
                print(f"[TeachingAssistantAgent] gTTS error: {e}")

        # 3. Deterministic synthetic audible notification tone fallback
        self._generate_synthetic_chime_wav(filepath)
        return {
            "success": True,
            "file_path": filepath,
            "audio_url": f"/storage/generated_audio/{filename}",
            "tts_engine": "Synthetic Audio Chime Fallback",
            "fallback_note": "Browser speech synthesis available for full narration."
        }

    def _generate_synthetic_chime_wav(self, filepath: str):
        """Generates a pleasant 2-tone harmonic chime as a standalone fallback WAV file."""
        sample_rate = 22050
        duration = 0.8
        n_samples = int(sample_rate * duration)
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = bytearray()
            for i in range(n_samples):
                t = i / sample_rate
                # Dual tone 523.25 Hz (C5) and 659.25 Hz (E5) with exponential decay
                decay = max(0.0, 1.0 - (t / duration) ** 0.8)
                val = 0.5 * math.sin(2.0 * math.pi * 523.25 * t) + 0.5 * math.sin(2.0 * math.pi * 659.25 * t)
                val = int(val * decay * 14000.0)
                frames.extend(struct.pack('<h', val))
            wav_file.writeframes(frames)

    def generate_classroom_code_exercise(self, topic: str = "Python Fundamentals",
                                         session_state: str = "Steady Session Pacing") -> dict:
        """
        Generates an interactive classroom code puzzle or comprehension check.
        Uses Gemini 2.0 Flash when available, with a curated offline repository fallback.
        """
        timestamp = int(time.time() * 1000)
        filename = f"exercise_{timestamp}.py"
        filepath = os.path.join(self.code_dir, filename)

        if self.is_gemini_available:
            try:
                prompt = f"""Generate a concise, self-contained Python coding exercise or mini-quiz (under 25 lines) suitable for a live classroom.
Context:
- Topic: {topic}
- Current Engagement State: {session_state}
Requirements:
- Must be valid Python code.
- Include a short docstring with the challenge question.
- Include a starter function or prompt with 1 or 2 test print assertions that execute cleanly.
- Output ONLY the executable python code block, no markdown enclosing."""

                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                code = response.text if hasattr(response, 'text') else str(response)
                code = code.replace("```python", "").replace("```", "").strip()

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(code)

                return {
                    "success": True,
                    "filename": filename,
                    "file_path": filepath,
                    "code": code,
                    "generator": "Google GenAI (gemini-2.0-flash)"
                }
            except Exception as e:
                print(f"[TeachingAssistantAgent] Code generation error: {e}. Falling back to offline templates.")

        # Offline curated exercises
        offline_templates = [
            '''"""
Classroom Quick Challenge: List Comprehension Filter
Goal: Filter all even numbers greater than 10 from the list and square them.
"""

def process_numbers(numbers):
    # Challenge: replace with your solution
    return [x**2 for x in numbers if x % 2 == 0 and x > 10]

# Verification test
test_data = [4, 12, 15, 18, 21, 24]
result = process_numbers(test_data)
print(f"Input: {test_data}")
print(f"Result: {result}")
assert result == [144, 324, 576], "Logic test failed"
print("[PASS] Verification Passed! Great job!")
''',
            '''"""
Classroom Quick Challenge: Real-Time Attendance Statistics
Goal: Compute the percentage of verified students given attendance statuses.
"""

def attendance_rate(statuses):
    if not statuses:
        return 0.0
    verified = sum(1 for s in statuses if s == "Present (Verified)")
    return round((verified / len(statuses)) * 100.0, 1)

# Verification test
mock_records = ["Present (Verified)", "Verifying (2.1s)", "Present (Verified)", "Present (Verified)"]
rate = attendance_rate(mock_records)
print(f"Class Attendance Rate: {rate}%")
assert rate == 75.0, "Verification mismatch"
print("[PASS] Statistics calculated accurately!")
''',
            '''"""
Classroom Quick Challenge: Multimodal Fusion Engine
Goal: Compute overall engagement using 65% visual and 35% audio weights.
"""

def fuse_scores(visual_score, audio_score, inactivity_flag=False):
    fused = 0.65 * visual_score + 0.35 * audio_score
    if inactivity_flag:
        fused *= 0.70  # Inactivity penalty
    return round(fused, 1)

# Test run
v_score = 80.0
a_score = 60.0
score_normal = fuse_scores(v_score, a_score, False)
score_inactive = fuse_scores(v_score, a_score, True)

print(f"Normal Engagement: {score_normal}%")
print(f"Engagement with Inactivity: {score_inactive}%")
assert score_normal == 73.0, "Score calculation error"
print("[PASS] Multimodal fusion logic verified!")
'''
        ]

        code = random.choice(offline_templates)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        return {
            "success": True,
            "filename": filename,
            "file_path": filepath,
            "code": code,
            "generator": "Curated Offline Pedagogical Repository"
        }

    def generate_visual_concept_summary(self, session_state: str, engagement_score: float) -> dict:
        """
        Produces a rich SVG/visual concept diagram summarizing the pedagogical session dynamics.
        Can be rendered in the browser or saved locally.
        """
        color_map = {
            "Attention Drop Detected": "#EF4444",
            "Passive Listening / High Inactivity": "#F59E0B",
            "Interaction Peak": "#10B981",
            "Steady Session Pacing": "#3B82F6"
        }
        accent = color_map.get(session_state, "#6366F1")

        svg_markup = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 280" width="100%" height="100%">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#111827"/>
      <stop offset="100%" stop-color="#1F2937"/>
    </linearGradient>
    <linearGradient id="meterGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#3B82F6"/>
      <stop offset="100%" stop-color="{accent}"/>
    </linearGradient>
  </defs>
  <rect width="540" height="280" rx="16" fill="url(#bgGrad)" stroke="#374151" stroke-width="1.5"/>
  <circle cx="270" cy="115" r="70" fill="none" stroke="#374151" stroke-width="12" stroke-dasharray="440" stroke-dashoffset="0"/>
  <circle cx="270" cy="115" r="70" fill="none" stroke="url(#meterGrad)" stroke-width="12"
          stroke-dasharray="440" stroke-dashoffset="{440 - (440 * (engagement_score / 100.0))}" stroke-linecap="round"/>
  <text x="270" y="112" fill="#F9FAFB" font-family="sans-serif" font-size="28" font-weight="bold" text-anchor="middle">{engagement_score:.1f}%</text>
  <text x="270" y="132" fill="#9CA3AF" font-family="sans-serif" font-size="12" text-anchor="middle">FUSED SCORE</text>
  <rect x="70" y="210" width="400" height="42" rx="8" fill="{accent}22" stroke="{accent}" stroke-width="1"/>
  <text x="270" y="236" fill="#F3F4F6" font-family="sans-serif" font-size="14" font-weight="600" text-anchor="middle">{session_state.upper()}</text>
</svg>'''
        return {
            "success": True,
            "svg_markup": svg_markup,
            "session_state": session_state,
            "engagement_score": engagement_score
        }
