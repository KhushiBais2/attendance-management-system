"""
Computer Vision Module (EPIC 1 & 2)
Handles automated sustained attendance tracking, head pose estimation,
landmark movement dynamics, inactivity detection, and synthetic classroom simulation.
Features MediaPipe FaceLandmarker with automatic OpenCV Haar Cascade fallback.
"""

import os
import time
import math
import random
import threading
import cv2
import numpy as np

# Standard 3D facial model points for head pose estimation (in mm)
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye corner
    (225.0, 170.0, -135.0),    # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype=np.float64)


class ParticipantRecord:
    """Tracks continuous presence, engagement history, and verification state for an individual learner."""

    def __init__(self, participant_id: str, label: str = "Learner", initial_pos: tuple = (0, 0)):
        self.id = participant_id
        self.label = label
        self.centroid = initial_pos
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.is_verified = False
        self.verified_timestamp = None
        self.sustained_duration = 0.0
        self.history_positions = [initial_pos]
        self.inactivity_counter = 0
        self.is_inactive = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.visual_score = 75.0

    def update_presence(self, centroid: tuple, sustained_threshold: float = 4.0):
        current_time = time.time()
        self.last_seen = current_time
        self.sustained_duration = current_time - self.first_seen

        # Movement tracking
        if self.centroid:
            dist = math.hypot(centroid[0] - self.centroid[0], centroid[1] - self.centroid[1])
            if dist < 1.8:  # Very little micro-movement
                self.inactivity_counter += 1
            else:
                self.inactivity_counter = max(0, self.inactivity_counter - 2)

        self.centroid = centroid
        self.history_positions.append(centroid)
        if len(self.history_positions) > 30:
            self.history_positions.pop(0)

        # Inactivity threshold (approx 4 seconds of stillness)
        self.is_inactive = self.inactivity_counter >= 35

        # Check sustained threshold
        if not self.is_verified and self.sustained_duration >= sustained_threshold:
            self.is_verified = True
            self.verified_timestamp = current_time

    def get_status_str(self, sustained_threshold: float = 4.0) -> str:
        if self.is_verified:
            return "Present (Verified)"
        remaining = max(0.0, sustained_threshold - self.sustained_duration)
        return f"Verifying ({self.sustained_duration:.1f}s / {sustained_threshold:.1f}s)"


class VisionModule:
    """
    Classroom Vision Processing System.
    Captures live webcam, uploaded video feeds, or generates synthetic classroom dynamics.
    Estimates 3D head pose and tracks sustained attendance.
    """

    def __init__(self, models_dir: str = "models", sustained_threshold: float = 4.0):
        self.models_dir = models_dir
        self.sustained_threshold = sustained_threshold
        self.lock = threading.Lock()

        # Video source configuration: 'simulation', 'webcam', 'upload'
        self.video_source_mode = "simulation"
        self.uploaded_video_path = None
        self.cap = None

        # Detectors
        self.mediapipe_landmarker = None
        self.haar_face_cascade = None
        self.haar_eye_cascade = None
        self.detector_type = "none"

        self._init_detectors()

        # Participant tracking state
        self.participants = {}
        self.max_missing_seconds = 2.5
        self.next_auto_id = 1

        # Global metrics snapshot
        self.current_metrics = {
            "participant_count": 0,
            "verified_count": 0,
            "visual_engagement_score": 75.0,
            "inactivity_detected": False,
            "head_pose_yaw_avg": 0.0,
            "head_pose_pitch_avg": 0.0,
            "detector_in_use": self.detector_type,
            "source_mode": self.video_source_mode,
            "timestamp": time.time()
        }

        # Simulation state
        self.sim_frame_count = 0
        self.sim_students = [
            {"id": "STU-01", "name": "Alex Chen", "roll": "CS-101", "action": "Note-Taking", "seat_x": 160, "seat_y": 240, "base_yaw": 0.0, "att_factor": 1.0, "start_delay": 0.0},
            {"id": "STU-02", "name": "Sophia Lopez", "roll": "CS-102", "action": "Attentive Study", "seat_x": 480, "seat_y": 240, "base_yaw": -5.0, "att_factor": 0.9, "start_delay": 3.0},
            {"id": "STU-03", "name": "David Kim", "roll": "CS-103", "action": "Reviewing Notes", "seat_x": 160, "seat_y": 440, "base_yaw": 12.0, "att_factor": 0.6, "start_delay": 1.0},
            {"id": "STU-04", "name": "Maya Patel", "roll": "CS-104", "action": "Active Question", "seat_x": 480, "seat_y": 440, "base_yaw": -3.0, "att_factor": 0.95, "start_delay": 6.0}
        ]
        self.sim_start_time = time.time()

    def _init_detectors(self):
        """Initializes MediaPipe FaceLandmarker with fallback to OpenCV Haar Cascades."""
        task_path = os.path.join(self.models_dir, "face_landmarker.task")
        haar_face_path = os.path.join(self.models_dir, "haarcascade_frontalface_default.xml")
        haar_eye_path = os.path.join(self.models_dir, "haarcascade_eye.xml")

        # 1. Try MediaPipe Tasks FaceLandmarker
        if os.path.exists(task_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=task_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=10,
                    min_face_detection_confidence=0.4,
                    min_face_presence_confidence=0.4,
                    min_tracking_confidence=0.4
                )
                self.mediapipe_landmarker = vision.FaceLandmarker.create_from_options(options)
                self.detector_type = "MediaPipe FaceMesh (v0.10+ Tasks)"
                print(f"[VisionModule] Successfully loaded MediaPipe FaceLandmarker.")
                return
            except Exception as e:
                print(f"[VisionModule] MediaPipe initialization error: {e}. Falling back to OpenCV.")

        # 2. Try OpenCV Haar Cascades
        if os.path.exists(haar_face_path):
            try:
                self.haar_face_cascade = cv2.CascadeClassifier(haar_face_path)
                if not self.haar_face_cascade.empty():
                    self.detector_type = "OpenCV Haar Cascade (Fallback)"
                    print(f"[VisionModule] Successfully loaded OpenCV Haar Cascade Face Detector.")
                    if os.path.exists(haar_eye_path):
                        self.haar_eye_cascade = cv2.CascadeClassifier(haar_eye_path)
                    return
            except Exception as e:
                print(f"[VisionModule] Haar cascade loading error: {e}")

        self.detector_type = "Heuristic Synthetic Detector"
        print(f"[VisionModule] Running in Heuristic Detector mode.")

    def set_source_mode(self, mode: str, video_file_path: str = None) -> bool:
        """Switches between 'simulation', 'webcam', and 'upload'."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.video_source_mode = mode

            if mode == "webcam":
                try:
                    # Attempt DirectShow on Windows for fast startup
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        print("[VisionModule] No physical webcam detected. Reverting to Simulation mode.")
                        self.video_source_mode = "simulation"
                        self.cap = None
                        return False
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return True
                except Exception as e:
                    print(f"[VisionModule] Webcam error: {e}. Reverting to Simulation.")
                    self.video_source_mode = "simulation"
                    return False

            elif mode == "upload":
                if video_file_path and os.path.exists(video_file_path):
                    self.uploaded_video_path = video_file_path
                    self.cap = cv2.VideoCapture(video_file_path)
                    return self.cap.isOpened()
                else:
                    self.video_source_mode = "simulation"
                    return False

            else:  # simulation
                self.video_source_mode = "simulation"
                self.sim_start_time = time.time()
                return True

    def reset_attendance(self):
        """Resets attendance records and tracking state."""
        with self.lock:
            self.participants.clear()
            self.next_auto_id = 1
            self.sim_start_time = time.time()

    def _estimate_head_pose_mediapipe(self, landmarks, img_w: int, img_h: int):
        """
        Calculates 3D Head Pose (Pitch, Yaw, Roll) using 6 key landmarks and cv2.solvePnP.
        Landmark indices:
            Nose tip: 1
            Chin: 152
            Left eye outer: 33
            Right eye outer: 263
            Left mouth corner: 61
            Right mouth corner: 291
        """
        try:
            image_points = np.array([
                (landmarks[1].x * img_w, landmarks[1].y * img_h),
                (landmarks[152].x * img_w, landmarks[152].y * img_h),
                (landmarks[33].x * img_w, landmarks[33].y * img_h),
                (landmarks[263].x * img_w, landmarks[263].y * img_h),
                (landmarks[61].x * img_w, landmarks[61].y * img_h),
                (landmarks[291].x * img_w, landmarks[291].y * img_h)
            ], dtype=np.float64)

            focal_length = img_w
            center = (img_w / 2.0, img_h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(
                FACE_3D_MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return 0.0, 0.0, 0.0

            rot_mat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

            pitch = angles[0] * 360.0
            yaw = angles[1] * 360.0
            roll = angles[2] * 360.0

            return float(yaw), float(pitch), float(roll)
        except Exception:
            return 0.0, 0.0, 0.0

    def _estimate_head_pose_haar(self, face_box, eyes, img_w: int, img_h: int):
        """Approximates head pose yaw & pitch from face bounding box and eye positions."""
        fx, fy, fw, fh = face_box
        face_cx = fx + fw / 2.0
        face_cy = fy + fh / 2.0

        yaw = ((face_cx - (img_w / 2.0)) / (img_w / 2.0)) * 25.0
        pitch = ((face_cy - (img_h / 2.0)) / (img_h / 2.0)) * 20.0

        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            eye_cx = (eyes_sorted[0][0] + eyes_sorted[1][0]) / 2.0
            yaw += ((eye_cx - face_cx) / (fw / 2.0)) * 15.0

        return float(yaw), float(pitch), 0.0

    def _match_or_create_participant(self, centroid: tuple, label: str = "Learner", fixed_id: str = None) -> ParticipantRecord:
        """Associates detected face centroid with existing participant or initializes a new track."""
        if fixed_id:
            if fixed_id in self.participants:
                return self.participants[fixed_id]
            record = ParticipantRecord(fixed_id, label, centroid)
            self.participants[fixed_id] = record
            return record

        best_id = None
        min_dist = 90.0  # Max pixel distance for same face association

        now = time.time()
        for pid, p in list(self.participants.items()):
            # Discard stale tracks
            if now - p.last_seen > self.max_missing_seconds:
                continue
            dist = math.hypot(p.centroid[0] - centroid[0], p.centroid[1] - centroid[1])
            if dist < min_dist:
                min_dist = dist
                best_id = pid

        if best_id is None:
            pid = f"STU-{self.next_auto_id:02d}"
            self.next_auto_id += 1
            record = ParticipantRecord(pid, label, centroid)
            self.participants[pid] = record
            return record

        return self.participants[best_id]

    def _calculate_visual_score(self, yaw: float, pitch: float, is_inactive: bool) -> float:
        """
        Computes visual engagement score (0 - 100%) based on frontal alignment and alertness.
        Optimal attention: looking toward camera/teacher (|yaw| < 15 deg, |pitch| < 12 deg).
        """
        # Yaw penalty (turning away sideways)
        yaw_penalty = min(60.0, (abs(yaw) / 45.0) * 60.0)
        # Pitch penalty (looking down at phone or looking up daydreaming)
        pitch_penalty = min(40.0, (abs(pitch) / 35.0) * 40.0)

        score = 100.0 - yaw_penalty - pitch_penalty
        if is_inactive:
            score -= 25.0  # Inactivity stillness penalty

        return max(5.0, min(100.0, score))

    def _draw_hud_overlay(self, frame: np.ndarray, face_box: tuple, record: ParticipantRecord, yaw: float, pitch: float):
        """Renders HUD telemetry, sustained attendance verification badge, and gaze direction vector."""
        x, y, w, h = face_box
        color_verified = (34, 197, 94)    # Green
        color_verifying = (245, 158, 11)  # Amber
        color_inactive = (239, 68, 68)    # Red

        box_color = color_verified if record.is_verified else color_verifying
        if record.is_inactive:
            box_color = color_inactive

        # Sleek corner brackets instead of heavy solid box
        corner_len = min(22, w // 4)
        thickness = 2
        # Top-left
        cv2.line(frame, (x, y), (x + corner_len, y), box_color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_len), box_color, thickness)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), box_color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), box_color, thickness)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), box_color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), box_color, thickness)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), box_color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), box_color, thickness)

        # Header Badge
        header_text = f"{record.id} | {record.label}"
        status_text = record.get_status_str(self.sustained_threshold)
        cv2.putText(frame, header_text, (x, max(20, y - 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # Verification Status Pill
        cv2.putText(frame, status_text, (x, max(36, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)

        # Engagement Meter Bar below face
        bar_y = y + h + 10
        bar_w = w
        cv2.rectangle(frame, (x, bar_y), (x + bar_w, bar_y + 5), (45, 45, 55), -1)
        fill_w = int(bar_w * (record.visual_score / 100.0))
        cv2.rectangle(frame, (x, bar_y), (x + fill_w, bar_y + 5), box_color, -1)

        # Head Pose Gaze Arrow
        center_x = x + w // 2
        center_y = y + h // 2
        arrow_dx = int(-yaw * 1.5)
        arrow_dy = int(pitch * 1.5)
        arrow_tip = (center_x + arrow_dx, center_y + arrow_dy)
        cv2.arrowedLine(frame, (center_x, center_y), arrow_tip, (0, 255, 255), 1, tipLength=0.3)

    def generate_synthetic_classroom_frame(self) -> np.ndarray:
        """
        Creates an animated, photorealistic synthetic classroom feed.
        Simulates individual student behaviors, blinks, head turns, notes taking,
        and showcases the sustained presence verification logic transition in real time!
        """
        self.sim_frame_count += 1
        elapsed = time.time() - self.sim_start_time

        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Classroom background gradient (modern dark tech classroom)
        for y in range(height):
            c_val = int(24 + 18 * (y / height))
            frame[y, :] = (c_val + 6, c_val + 4, c_val + 10)

        # Classroom blackboard / digital lecture screen top header
        cv2.rectangle(frame, (20, 12), (width - 20, 68), (28, 38, 48), -1)
        cv2.rectangle(frame, (20, 12), (width - 20, 68), (56, 78, 100), 2)
        cv2.putText(frame, "CS-301: ADVANCED ALGORITHMS | LECTURE 04: STUDY TELEMETRY", (32, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "TOPIC: GRAPH OPTIMIZATION & ATTENTION CONVERGENCE", (32, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 185, 225), 1, cv2.LINE_AA)

        # Draw 4 desks & student workstations
        detected_faces = []
        for stu in self.sim_students:
            # Check student arrival delay (demonstrates sustained presence logic)
            if elapsed < stu["start_delay"]:
                # Empty desk
                dx = stu["seat_x"] - 90
                dy = stu["seat_y"] - 60
                cv2.rectangle(frame, (dx, dy), (dx + 180, dy + 150), (32, 36, 46), -1)
                cv2.putText(frame, f"Reserved: {stu['name']}", (dx + 25, dy + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 110, 125), 1)
                continue

            # Student is present
            t_student = elapsed - stu["start_delay"]

            # Head movement dynamics
            nod = 3.0 * math.sin(t_student * 1.8 + stu["seat_x"])
            turn = stu["base_yaw"] + 8.0 * math.sin(t_student * 0.9)
            if stu["att_factor"] < 0.7:
                # Distracted student looks around more
                turn += 18.0 * math.sin(t_student * 0.4)
                nod += 6.0 * math.cos(t_student * 0.3)

            # Inactivity simulation: David occasionally remains static
            is_sim_inactive = False
            if stu["id"] == "STU-03" and (int(t_student) % 20 > 12):
                turn = 24.0
                nod = 14.0
                is_sim_inactive = True

            head_cx = stu["seat_x"] + int(turn * 0.8)
            head_cy = stu["seat_y"] + int(nod * 0.6)

            # Draw student avatar
            # Shoulders / torso
            cv2.ellipse(frame, (head_cx, head_cy + 75), (60, 35), 0, 0, 180, (48, 55, 75), -1)
            # Head base
            cv2.circle(frame, (head_cx, head_cy), 36, (190, 160, 145), -1)
            # Hair
            cv2.ellipse(frame, (head_cx, head_cy - 16), (36, 26), 0, 180, 360, (30, 25, 25), -1)

            # Eyes with blinking
            blink = (math.sin(t_student * 4.0 + stu["seat_y"]) > 0.94)
            eye_offset_x = int(-turn * 0.3)
            left_eye = (head_cx - 12 + eye_offset_x, head_cy - 2)
            right_eye = (head_cx + 12 + eye_offset_x, head_cy - 2)
            if blink:
                cv2.line(frame, (left_eye[0] - 5, left_eye[1]), (left_eye[0] + 5, left_eye[1]), (30, 20, 20), 2)
                cv2.line(frame, (right_eye[0] - 5, right_eye[1]), (right_eye[0] + 5, right_eye[1]), (30, 20, 20), 2)
            else:
                cv2.circle(frame, left_eye, 4, (255, 255, 255), -1)
                cv2.circle(frame, left_eye, 2, (30, 20, 20), -1)
                cv2.circle(frame, right_eye, 4, (255, 255, 255), -1)
                cv2.circle(frame, right_eye, 2, (30, 20, 20), -1)

            # Mouth
            mouth_y = head_cy + 18
            cv2.line(frame, (head_cx - 6, mouth_y), (head_cx + 6, mouth_y), (140, 70, 70), 2)

            # Face box for tracking
            face_box = (head_cx - 44, head_cy - 48, 88, 102)

            # Register participant tracking
            record = self._match_or_create_participant((head_cx, head_cy), stu["name"], fixed_id=stu["id"])
            record.update_presence((head_cx, head_cy), self.sustained_threshold)
            record.is_inactive = is_sim_inactive
            record.yaw = float(turn)
            record.pitch = float(nod)
            record.visual_score = self._calculate_visual_score(turn, nod, is_sim_inactive)

            # Draw HUD overlay
            self._draw_hud_overlay(frame, face_box, record, turn, nod)
            detected_faces.append(record)

        self._update_aggregate_metrics(detected_faces)
        return frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes an incoming video frame (from webcam or uploaded file).
        Applies MediaPipe FaceMesh / OpenCV Haar Cascade detection,
        head pose calculation, and HUD rendering.
        """
        img_h, img_w, _ = frame.shape
        detected_faces = []

        # 1. Primary: MediaPipe FaceLandmarker
        if self.mediapipe_landmarker is not None:
            try:
                import mediapipe as mp
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.mediapipe_landmarker.detect(mp_image)

                if detection_result.face_landmarks:
                    for landmarks in detection_result.face_landmarks:
                        # Compute bounding box
                        xs = [lm.x * img_w for lm in landmarks]
                        ys = [lm.y * img_h for lm in landmarks]
                        min_x, max_x = max(0, int(min(xs))), min(img_w, int(max(xs)))
                        min_y, max_y = max(0, int(min(ys))), min(img_h, int(max(ys)))
                        w, h = max(10, max_x - min_x), max(10, max_y - min_y)
                        cx, cy = int((min_x + max_x) / 2), int((min_y + max_y) / 2)

                        yaw, pitch, roll = self._estimate_head_pose_mediapipe(landmarks, img_w, img_h)

                        record = self._match_or_create_participant((cx, cy))
                        record.update_presence((cx, cy), self.sustained_threshold)
                        record.yaw = yaw
                        record.pitch = pitch
                        record.roll = roll
                        record.visual_score = self._calculate_visual_score(yaw, pitch, record.is_inactive)

                        self._draw_hud_overlay(frame, (min_x, min_y, w, h), record, yaw, pitch)
                        detected_faces.append(record)

                    self._update_aggregate_metrics(detected_faces)
                    return frame
            except Exception as e:
                print(f"[VisionModule] MediaPipe frame error: {e}. Falling back to Haar Cascade.")

        # 2. Fallback: OpenCV Haar Cascade
        if self.haar_face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(45, 45)
            )

            for (fx, fy, fw, fh) in faces:
                cx, cy = fx + fw // 2, fy + fh // 2
                eyes = []
                if self.haar_eye_cascade is not None:
                    roi_gray = gray[fy:fy + fh, fx:fx + fw]
                    detected_eyes = self.haar_eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(15, 15))
                    for (ex, ey, ew, eh) in detected_eyes:
                        eyes.append((fx + ex + ew // 2, fy + ey + eh // 2))

                yaw, pitch, roll = self._estimate_head_pose_haar((fx, fy, fw, fh), eyes, img_w, img_h)

                record = self._match_or_create_participant((cx, cy))
                record.update_presence((cx, cy), self.sustained_threshold)
                record.yaw = yaw
                record.pitch = pitch
                record.roll = roll
                record.visual_score = self._calculate_visual_score(yaw, pitch, record.is_inactive)

                self._draw_hud_overlay(frame, (fx, fy, fw, fh), record, yaw, pitch)
                detected_faces.append(record)

        self._update_aggregate_metrics(detected_faces)
        return frame

    def _update_aggregate_metrics(self, detected_records: list):
        """Calculates global class-level visual telemetry snapshot."""
        now = time.time()
        # Clean up stale participants absent for too long
        with self.lock:
            active_pids = [p.id for p in detected_records]
            for pid in list(self.participants.keys()):
                if pid not in active_pids and (now - self.participants[pid].last_seen > self.max_missing_seconds):
                    # We retain verified attendees in the persistent registry, but they aren't active in the current frame
                    pass

            count = len(detected_records)
            verified = sum(1 for p in self.participants.values() if p.is_verified)

            if count > 0:
                avg_visual = float(np.mean([p.visual_score for p in detected_records]))
                avg_yaw = float(np.mean([p.yaw for p in detected_records]))
                avg_pitch = float(np.mean([p.pitch for p in detected_records]))
                inactivity = any(p.is_inactive for p in detected_records)
            else:
                avg_visual = 65.0
                avg_yaw = 0.0
                avg_pitch = 0.0
                inactivity = False

            self.current_metrics = {
                "participant_count": count,
                "verified_count": verified,
                "visual_engagement_score": round(avg_visual, 1),
                "inactivity_detected": inactivity,
                "head_pose_yaw_avg": round(avg_yaw, 1),
                "head_pose_pitch_avg": round(avg_pitch, 1),
                "detector_in_use": self.detector_type,
                "source_mode": self.video_source_mode,
                "timestamp": now
            }

    def get_frame(self) -> np.ndarray:
        """Retrieves and processes the next frame from the active source."""
        if self.video_source_mode == "simulation":
            return self.generate_synthetic_classroom_frame()

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Loop video if upload ended
                if self.video_source_mode == "upload" and self.uploaded_video_path:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

            if ret and frame is not None:
                return self.process_frame(frame)

        # Fallback to simulation if hardware capture fails
        return self.generate_synthetic_classroom_frame()

    def get_attendance_list(self) -> list:
        """Returns sorted list of participant attendance records."""
        with self.lock:
            records = []
            for pid, p in self.participants.items():
                records.append({
                    "id": p.id,
                    "label": p.label,
                    "is_verified": p.is_verified,
                    "status": p.get_status_str(self.sustained_threshold),
                    "sustained_duration": round(p.sustained_duration, 1),
                    "threshold_seconds": self.sustained_threshold,
                    "first_seen": time.strftime("%H:%M:%S", time.localtime(p.first_seen)),
                    "last_seen": time.strftime("%H:%M:%S", time.localtime(p.last_seen)),
                    "verified_at": time.strftime("%H:%M:%S", time.localtime(p.verified_timestamp)) if p.verified_timestamp else None,
                    "visual_score": round(p.visual_score, 1),
                    "is_inactive": p.is_inactive
                })
            # Sort verified first, then by ID
            records.sort(key=lambda r: (not r["is_verified"], r["id"]))
            return records

    def get_latest_metrics(self) -> dict:
        """Returns the current vision metrics snapshot."""
        with self.lock:
            return dict(self.current_metrics)
