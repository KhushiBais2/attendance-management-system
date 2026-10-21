/**
 * EduPulse AI - Classroom & Study Session Intelligence Controller
 * Real-Time Study Telemetry, Academic Pacing Analytics,
 * Sustained Attendance Verification, Focus Timer & Practice Quiz Sandbox.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Academic Application State
    const AppState = {
        isPolling: true,
        pollIntervalMs: 1000,
        currentSource: "simulation",
        audioSimMode: "balanced",
        studyMode: "lecture",
        activeCourse: "CS-301: Deep Learning & Neural Systems",
        micActive: false,
        audioContext: null,
        micStream: null,
        audioProcessor: null,
        sustainedThresholdSeconds: 4.0,
        geminiApiKey: localStorage.getItem("gemini_api_key") || "",
        chart: null,
        maxChartPoints: 30,

        // Pomodoro / Study Focus Timer
        timerDurationSeconds: 25 * 60,
        timerRemainingSeconds: 25 * 60,
        timerRunning: false,
        timerInterval: null
    };

    // DOM Elements - Navigation & Academic Context
    const systemStatusPill = document.getElementById("systemStatusPill");
    const systemStatusText = document.getElementById("systemStatusText");
    const aiModePill = document.getElementById("aiModePill");
    const aiModeText = document.getElementById("aiModeText");
    const btnSettings = document.getElementById("btnSettings");
    const btnResetSession = document.getElementById("btnResetSession");
    const courseSelect = document.getElementById("courseSelect");

    // DOM Elements - Study Session Mode Strip
    const modeButtons = document.querySelectorAll(".mode-btn");
    const studyTimerText = document.getElementById("studyTimerText");
    const btnToggleTimer = document.getElementById("btnToggleTimer");

    // DOM Elements - AI Teaching Suggestions Banner
    const sessionStateBadge = document.getElementById("sessionStateBadge");
    const sessionStateText = document.getElementById("sessionStateText");
    const aiSuggestionText = document.getElementById("aiSuggestionText");
    const btnPlayAudio = document.getElementById("btnPlayAudio");
    const btnGenerateCode = document.getElementById("btnGenerateCode");
    const btnConceptDiagram = document.getElementById("btnConceptDiagram");
    const ttsAudioPlayer = document.getElementById("ttsAudioPlayer");

    // DOM Elements - Academic KPI Cards
    const kpiLearnersCount = document.getElementById("kpiLearnersCount");
    const kpiVerifiedCount = document.getElementById("kpiVerifiedCount");
    const kpiVerifiedBadge = document.getElementById("kpiVerifiedBadge");
    const kpiVisualScore = document.getElementById("kpiVisualScore");
    const kpiVisualTrend = document.getElementById("kpiVisualTrend");
    const kpiVisualFill = document.getElementById("kpiVisualFill");
    const kpiHeadPoseText = document.getElementById("kpiHeadPoseText");
    const kpiInactivityFlag = document.getElementById("kpiInactivityFlag");
    const kpiAudioScore = document.getElementById("kpiAudioScore");
    const kpiSpeechBadge = document.getElementById("kpiSpeechBadge");
    const kpiAudioFill = document.getElementById("kpiAudioFill");
    const kpiAudioDetails = document.getElementById("kpiAudioDetails");
    const kpiAudioModeText = document.getElementById("kpiAudioModeText");
    const kpiFusedScore = document.getElementById("kpiFusedScore");
    const kpiFusedFill = document.getElementById("kpiFusedFill");
    const kpiTrendText = document.getElementById("kpiTrendText");
    const kpiInactivityPenaltyText = document.getElementById("kpiInactivityPenaltyText");
    const kpiGradePill = document.getElementById("kpiGradePill");

    // DOM Elements - Classroom Video & HUD
    const classroomVideoFeed = document.getElementById("classroomVideoFeed");
    const currentSourceTag = document.getElementById("currentSourceTag");
    const btnSourceSim = document.getElementById("btnSourceSim");
    const btnSourceWebcam = document.getElementById("btnSourceWebcam");
    const btnSourceUpload = document.getElementById("btnSourceUpload");
    const videoFileInput = document.getElementById("videoFileInput");
    const btnFullscreen = document.getElementById("btnFullscreen");
    const videoWrapper = document.getElementById("videoWrapper");
    const hudDetectorLabel = document.getElementById("hudDetectorLabel");
    const hudPresenceThreshold = document.getElementById("hudPresenceThreshold");

    // DOM Elements - Acoustic Controls
    const simPills = document.querySelectorAll(".sim-pill");
    const btnToggleMic = document.getElementById("btnToggleMic");
    const micToggleText = document.getElementById("micToggleText");

    // DOM Elements - Attendance Ledger Table
    const attendanceTableBody = document.getElementById("attendanceTableBody");
    const tableVerifiedTotal = document.getElementById("tableVerifiedTotal");
    const tableTotalCount = document.getElementById("tableTotalCount");
    const btnExportCSV = document.getElementById("btnExportCSV");
    const btnExportJSON = document.getElementById("btnExportJSON");
    const btnExportAnalytics = document.getElementById("btnExportAnalytics");

    // DOM Elements - Modals
    const settingsModal = document.getElementById("settingsModal");
    const btnCloseSettings = document.getElementById("btnCloseSettings");
    const btnCancelSettings = document.getElementById("btnCancelSettings");
    const btnSaveSettings = document.getElementById("btnSaveSettings");
    const inputGeminiApiKey = document.getElementById("inputGeminiApiKey");
    const btnToggleKeyVis = document.getElementById("btnToggleKeyVis");
    const inputSustainedThreshold = document.getElementById("inputSustainedThreshold");

    const codeModal = document.getElementById("codeModal");
    const btnCloseCodeModal = document.getElementById("btnCloseCodeModal");
    const btnCloseCodeBtn = document.getElementById("btnCloseCodeBtn");
    const btnRunCode = document.getElementById("btnRunCode");
    const btnRegenerateCode = document.getElementById("btnRegenerateCode");
    const codeEditor = document.getElementById("codeEditor");
    const terminalOutput = document.getElementById("terminalOutput");
    const execMeta = document.getElementById("execMeta");
    const codeGeneratorLabel = document.getElementById("codeGeneratorLabel");

    const diagramModal = document.getElementById("diagramModal");
    const btnCloseDiagramModal = document.getElementById("btnCloseDiagramModal");
    const btnDismissDiagram = document.getElementById("btnDismissDiagram");
    const diagramContainer = document.getElementById("diagramContainer");

    // =========================================================================
    // 1. Chart.js Initialization (Academic Lecture Attention Trends)
    // =========================================================================
    function initChart() {
        const ctx = document.getElementById("multimodalChart").getContext("2d");

        AppState.chart = new Chart(ctx, {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Study Engagement Index (SEI %)",
                        data: [],
                        borderColor: "#7c3aed",
                        backgroundColor: "rgba(124, 58, 237, 0.14)",
                        borderWidth: 2.5,
                        tension: 0.35,
                        fill: true,
                        pointRadius: 2,
                        pointHoverRadius: 5
                    },
                    {
                        label: "Classroom Dialogue & Questions %",
                        data: [],
                        borderColor: "#f59e0b",
                        backgroundColor: "transparent",
                        borderWidth: 2,
                        borderDash: [4, 4],
                        tension: 0.3,
                        pointRadius: 0
                    },
                    {
                        label: "Visual Focus & Notes Attention %",
                        data: [],
                        borderColor: "#3b82f6",
                        backgroundColor: "transparent",
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: { mode: "index", intersect: false },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 10 }, maxTicksLimit: 8 }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: {
                            color: "#64748b",
                            font: { family: "JetBrains Mono", size: 10 },
                            stepSize: 20,
                            callback: (v) => v + "%"
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "rgba(15, 23, 42, 0.95)",
                        titleColor: "#f8fafc",
                        bodyColor: "#cbd5e1",
                        borderColor: "rgba(59, 130, 246, 0.3)",
                        borderWidth: 1,
                        padding: 10,
                        bodyFont: { family: "JetBrains Mono", size: 11 }
                    }
                }
            }
        });
    }

    // =========================================================================
    // 2. Real-Time Telemetry Updates
    // =========================================================================
    async function fetchTelemetry() {
        if (!AppState.isPolling) return;

        try {
            const [metricsRes, attendanceRes] = await Promise.all([
                fetch("/api/metrics"),
                fetch("/api/attendance")
            ]);

            if (metricsRes.ok) {
                const metrics = await metricsRes.json();
                updateKPIs(metrics);
                updateAIIntelligence(metrics.ai_intelligence);
                updateChartData(metrics.fusion);
            }

            if (attendanceRes.ok) {
                const attendance = await attendanceRes.json();
                updateAttendanceTable(attendance);
            }
        } catch (err) {
            console.error("Study telemetry fetch error:", err);
            systemStatusText.textContent = "Connecting...";
            systemStatusPill.querySelector(".pulse-dot").classList.remove("online");
        }
    }

    function updateKPIs(data) {
        const fusion = data.fusion || {};
        const vision = data.vision || {};
        const audio = data.audio || {};
        const summary = data.attendance_summary || {};

        // 1. Student Roll Call Card
        const detected = summary.total_detected || 0;
        const verified = summary.verified_count || 0;
        kpiLearnersCount.textContent = detected;
        kpiVerifiedCount.textContent = verified;

        // 2. Lecture Visual Focus Card
        const vScore = vision.visual_engagement_score || 70.0;
        kpiVisualScore.textContent = `${Math.round(vScore)}%`;
        kpiVisualFill.style.width = `${Math.min(100, Math.max(0, vScore))}%`;

        const yaw = vision.head_pose_yaw_avg || 0.0;
        const pitch = vision.head_pose_pitch_avg || 0.0;
        kpiHeadPoseText.innerHTML = `Board Gaze: Yaw ${yaw > 0 ? "+" : ""}${yaw.toFixed(1)}&deg; &bull; Pitch ${pitch > 0 ? "+" : ""}${pitch.toFixed(1)}&deg;`;

        if (vision.inactivity_detected) {
            kpiInactivityFlag.className = "inactivity-indicator alert";
            kpiInactivityFlag.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Inattention Alert`;
        } else {
            kpiInactivityFlag.className = "inactivity-indicator";
            kpiInactivityFlag.innerHTML = `<i class="fa-solid fa-pen-nib"></i> Taking Notes`;
        }

        // 3. Verbal Participation Card
        const aScore = audio.audio_interaction_score || 50.0;
        kpiAudioScore.textContent = `${Math.round(aScore)}%`;
        kpiAudioFill.style.width = `${Math.min(100, Math.max(0, aScore))}%`;

        if (audio.speech_active) {
            kpiSpeechBadge.style.display = "inline-flex";
            kpiSpeechBadge.innerHTML = `<span class="speech-dot"></span> Active Dialogue`;
        } else {
            kpiSpeechBadge.style.display = "none";
        }

        const rms = audio.rms_energy || 0.0;
        const zcr = audio.zero_crossing_rate || 0.0;
        kpiAudioDetails.textContent = `RMS: ${rms.toFixed(3)} • ZCR: ${zcr.toFixed(3)}`;
        kpiAudioModeText.textContent = `Acoustic: ${audio.simulation_mode || AppState.audioSimMode}`;

        // 4. Study Engagement Index (SEI) Card
        const fScore = fusion.overall_engagement_score || 75.0;
        kpiFusedScore.textContent = `${Math.round(fScore)}%`;
        kpiFusedFill.style.width = `${Math.min(100, Math.max(0, fScore))}%`;

        // Calculate Academic Focus Grade
        if (fScore >= 85) {
            kpiGradePill.textContent = "Grade A+ • Deep Focus";
            kpiGradePill.style.background = "rgba(16, 185, 129, 0.2)";
            kpiGradePill.style.color = "#6ee7b7";
        } else if (fScore >= 75) {
            kpiGradePill.textContent = "Grade A • Optimal Focus";
            kpiGradePill.style.background = "rgba(37, 99, 235, 0.2)";
            kpiGradePill.style.color = "#93c5fd";
        } else if (fScore >= 60) {
            kpiGradePill.textContent = "Grade B • Steady Focus";
            kpiGradePill.style.background = "rgba(124, 58, 237, 0.2)";
            kpiGradePill.style.color = "#c4b5fd";
        } else {
            kpiGradePill.textContent = "Grade C • Attention Check";
            kpiGradePill.style.background = "rgba(239, 68, 68, 0.2)";
            kpiGradePill.style.color = "#fca5a5";
        }

        const trend = fusion.trend || "steady";
        if (trend === "rising") {
            kpiTrendText.innerHTML = `<i class="fa-solid fa-arrow-trend-up" style="color: #10b981;"></i> Rising Class Attention`;
        } else if (trend === "falling") {
            kpiTrendText.innerHTML = `<i class="fa-solid fa-arrow-trend-down" style="color: #ef4444;"></i> Focus Fatigue Detected`;
        } else {
            kpiTrendText.innerHTML = `<i class="fa-solid fa-minus" style="color: #3b82f6;"></i> Steady Lecture Pacing`;
        }

        if (vision.inactivity_detected) {
            kpiInactivityPenaltyText.textContent = "Cognitive Fatigue Penalty (-28%)";
            kpiInactivityPenaltyText.style.color = "#f87171";
        } else {
            kpiInactivityPenaltyText.textContent = "No Fatigue Penalty";
            kpiInactivityPenaltyText.style.color = "#64748b";
        }

        if (vision.detector_in_use) {
            hudDetectorLabel.textContent = vision.detector_in_use;
        }
    }

    function updateAIIntelligence(ai) {
        if (!ai) return;

        const state = ai.session_state || "Steady Session Pacing";
        const suggestion = ai.ai_suggestion || "Lecture pacing is balanced and attentive.";
        const source = ai.agent_source || "Deterministic Offline Pedagogical Engine";

        sessionStateText.textContent = state;
        aiSuggestionText.textContent = suggestion;

        sessionStateBadge.style.borderColor = "";
        sessionStateBadge.style.background = "";

        if (state === "Attention Drop Detected") {
            sessionStateBadge.style.borderColor = "var(--academic-red)";
            sessionStateBadge.style.background = "rgba(239, 68, 68, 0.2)";
            sessionStateBadge.querySelector(".badge-dot").style.backgroundColor = "var(--academic-red)";
        } else if (state === "Interaction Peak") {
            sessionStateBadge.style.borderColor = "var(--academic-green)";
            sessionStateBadge.style.background = "rgba(16, 185, 129, 0.2)";
            sessionStateBadge.querySelector(".badge-dot").style.backgroundColor = "var(--academic-green)";
        } else if (state === "Passive Listening / High Inactivity") {
            sessionStateBadge.style.borderColor = "var(--academic-amber)";
            sessionStateBadge.style.background = "rgba(245, 158, 11, 0.2)";
            sessionStateBadge.querySelector(".badge-dot").style.backgroundColor = "var(--academic-amber)";
        } else {
            sessionStateBadge.style.borderColor = "var(--academic-blue)";
            sessionStateBadge.style.background = "rgba(37, 99, 235, 0.2)";
            sessionStateBadge.querySelector(".badge-dot").style.backgroundColor = "var(--academic-blue)";
        }

        if (source.includes("Gemini") || source.includes("google-genai")) {
            aiModeText.textContent = "Gemini 2.0 Flash";
            aiModePill.style.borderColor = "rgba(16, 185, 129, 0.4)";
            aiModePill.style.color = "#6ee7b7";
        } else {
            aiModeText.textContent = "Offline Pedagogical Engine";
            aiModePill.style.borderColor = "rgba(59, 130, 246, 0.35)";
            aiModePill.style.color = "#bfdbfe";
        }
    }

    function updateChartData(fused) {
        if (!AppState.chart || !fused) return;

        const timeStr = fused.time_str || new Date().toLocaleTimeString();
        const fScore = fused.overall_engagement_score || 0;
        const aScore = fused.audio_interaction_score || 0;
        const vScore = fused.visual_engagement_score || 0;

        AppState.chart.data.labels.push(timeStr);
        AppState.chart.data.datasets[0].data.push(fScore);
        AppState.chart.data.datasets[1].data.push(aScore);
        AppState.chart.data.datasets[2].data.push(vScore);

        if (AppState.chart.data.labels.length > AppState.maxChartPoints) {
            AppState.chart.data.labels.shift();
            AppState.chart.data.datasets[0].data.shift();
            AppState.chart.data.datasets[1].data.shift();
            AppState.chart.data.datasets[2].data.shift();
        }

        AppState.chart.update("none");
    }

    function updateAttendanceTable(data) {
        const records = data.records || [];
        const threshold = data.sustained_threshold_seconds || 4.0;
        hudPresenceThreshold.textContent = `Verification: ${threshold.toFixed(1)}s Continuous Study Focus`;

        tableTotalCount.textContent = records.length;
        const verifiedCount = records.filter(r => r.is_verified).length;
        tableVerifiedTotal.textContent = verifiedCount;

        if (records.length === 0) {
            attendanceTableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="empty-state">
                        <i class="fa-solid fa-graduation-cap"></i> Initializing classroom student roster...
                    </td>
                </tr>
            `;
            return;
        }

        // Student action descriptions for study context
        const studyRoles = {
            "STU-01": "Roll #CS-101 • Note-Taking",
            "STU-02": "Roll #CS-102 • Attentive Study",
            "STU-03": "Roll #CS-103 • Reviewing Notes",
            "STU-04": "Roll #CS-104 • Active Question"
        };

        let html = "";
        records.forEach(rec => {
            const isVerified = rec.is_verified;
            const statusClass = isVerified ? "verified" : "verifying";
            const statusIcon = isVerified ? "fa-circle-check" : "fa-clock";
            const initial = (rec.label || "S").charAt(0).toUpperCase();
            const roleDesc = studyRoles[rec.id] || "Roll #S-Class • Student";

            let postureHtml = '<span style="color: #10b981; font-weight: 700;"><i class="fa-solid fa-pen"></i> Attentive & Note-Taking</span>';
            if (rec.is_inactive) {
                postureHtml = '<span style="color: #ef4444; font-weight: 700;"><i class="fa-solid fa-bed"></i> Needs Focus Check</span>';
            } else if (rec.visual_score > 85) {
                postureHtml = '<span style="color: #3b82f6; font-weight: 700;"><i class="fa-solid fa-eye"></i> High Slide Focus</span>';
            }

            html += `
                <tr>
                    <td class="participant-id">${rec.id}</td>
                    <td>
                        <div class="student-avatar-cell">
                            <div class="avatar-badge">${initial}</div>
                            <div>
                                <div>${rec.label}</div>
                                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 400;">${roleDesc}</div>
                            </div>
                        </div>
                    </td>
                    <td>${rec.first_seen || "--"}</td>
                    <td>${rec.last_seen || "--"}</td>
                    <td class="duration-meter">
                        ${rec.sustained_duration.toFixed(1)}s
                    </td>
                    <td>
                        <span class="status-tag ${statusClass}">
                            <i class="fa-solid ${statusIcon}"></i>
                            ${rec.status}
                        </span>
                    </td>
                    <td>
                        <strong>${rec.visual_score.toFixed(0)}%</strong>
                    </td>
                    <td>
                        ${postureHtml}
                    </td>
                </tr>
            `;
        });

        attendanceTableBody.innerHTML = html;
    }

    // =========================================================================
    // 3. Study Session Mode & Pomodoro Timer
    // =========================================================================
    modeButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            modeButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            AppState.studyMode = btn.dataset.mode;

            // Automatically harmonize acoustic simulation profile with study mode
            const modeAcousticMap = {
                "lecture": "balanced",
                "discussion": "active_discussion",
                "studyhall": "lecture_silence",
                "groupproject": "question_answer"
            };

            const targetSim = modeAcousticMap[AppState.studyMode] || "balanced";
            simPills.forEach(pill => {
                if (pill.dataset.sim === targetSim) pill.click();
            });
        });
    });

    // Pomodoro Timer Logic
    btnToggleTimer.addEventListener("click", () => {
        if (AppState.timerRunning) {
            clearInterval(AppState.timerInterval);
            AppState.timerRunning = false;
            btnToggleTimer.textContent = "Resume";
            btnToggleTimer.style.background = "rgba(245, 158, 11, 0.2)";
        } else {
            AppState.timerRunning = true;
            btnToggleTimer.textContent = "Pause";
            btnToggleTimer.style.background = "rgba(239, 68, 68, 0.2)";

            AppState.timerInterval = setInterval(() => {
                if (AppState.timerRemainingSeconds > 0) {
                    AppState.timerRemainingSeconds--;
                    const mins = Math.floor(AppState.timerRemainingSeconds / 60);
                    const secs = AppState.timerRemainingSeconds % 60;
                    studyTimerText.textContent = `Focus Timer: ${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                } else {
                    clearInterval(AppState.timerInterval);
                    AppState.timerRunning = false;
                    studyTimerText.textContent = "Study Block Complete! Take 5 min Break.";
                    btnToggleTimer.textContent = "Reset";
                    alert("Study Focus Block Complete! Great focus session.");
                    AppState.timerRemainingSeconds = AppState.timerDurationSeconds;
                }
            }, 1000);
        }
    });

    // Course dropdown change listener
    courseSelect.addEventListener("change", (e) => {
        AppState.activeCourse = e.target.options[e.target.selectedIndex].text;
        const hudTag = document.querySelector(".hud-course-tag");
        if (hudTag) hudTag.textContent = e.target.value;
    });

    // =========================================================================
    // 4. Video Source Switcher & Lecture Video Upload Handling
    // =========================================================================
    async function setVideoSource(source, videoPath = null) {
        try {
            const res = await fetch("/api/video_source", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source: source, video_path: videoPath })
            });

            const data = await res.json();
            AppState.currentSource = data.active_source;

            [btnSourceSim, btnSourceWebcam, btnSourceUpload].forEach(btn => btn.classList.remove("active"));
            if (AppState.currentSource === "simulation") {
                btnSourceSim.classList.add("active");
                currentSourceTag.textContent = "Simulation: 4 Student Desks";
            } else if (AppState.currentSource === "webcam") {
                btnSourceWebcam.classList.add("active");
                currentSourceTag.textContent = "Live Classroom Webcam";
            } else if (AppState.currentSource === "upload") {
                btnSourceUpload.classList.add("active");
                currentSourceTag.textContent = "Lecture Video Replay";
            }

            classroomVideoFeed.src = `/video_feed?t=${Date.now()}`;
        } catch (err) {
            console.error("Error setting video source:", err);
        }
    }

    btnSourceSim.addEventListener("click", () => setVideoSource("simulation"));
    btnSourceWebcam.addEventListener("click", () => setVideoSource("webcam"));
    btnSourceUpload.addEventListener("click", () => videoFileInput.click());

    videoFileInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("video_file", file);

        currentSourceTag.textContent = "Uploading lecture video...";
        try {
            const res = await fetch("/api/upload_video", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                setVideoSource("upload", data.file_path);
            } else {
                alert(`Upload failed: ${data.error || "Unknown error"}`);
            }
        } catch (err) {
            console.error("Lecture video upload error:", err);
            alert("Video upload failed.");
        }
    });

    btnFullscreen.addEventListener("click", () => {
        if (!document.fullscreenElement) {
            videoWrapper.requestFullscreen().catch(err => console.error(err));
        } else {
            document.exitFullscreen();
        }
    });

    // =========================================================================
    // 5. Acoustic Simulation Profile Control
    // =========================================================================
    simPills.forEach(pill => {
        pill.addEventListener("click", async () => {
            simPills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");

            const mode = pill.dataset.sim;
            AppState.audioSimMode = mode;

            try {
                await fetch("/api/audio/simulation_mode", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ mode: mode })
                });
            } catch (err) {
                console.error("Error setting audio sim mode:", err);
            }
        });
    });

    // =========================================================================
    // 6. Browser Live Microphone Input (Web Audio API)
    // =========================================================================
    btnToggleMic.addEventListener("click", async () => {
        if (AppState.micActive) {
            stopMicrophone();
        } else {
            startMicrophone();
        }
    });

    async function startMicrophone() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            AppState.micStream = stream;
            AppState.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

            const source = AppState.audioContext.createMediaStreamSource(stream);
            AppState.audioProcessor = AppState.audioContext.createScriptProcessor(4096, 1, 1);

            AppState.audioProcessor.onaudioprocess = (e) => {
                if (!AppState.micActive) return;
                const inputData = e.inputBuffer.getChannelData(0);

                const pcm16 = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                const bytes = new Uint8Array(pcm16.buffer);
                let binary = "";
                for (let i = 0; i < bytes.byteLength; i++) {
                    binary += String.fromCharCode(bytes[i]);
                }
                const b64 = btoa(binary);

                fetch("/api/audio/process_buffer", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ audio_buffer: b64 })
                }).catch(() => {});
            };

            source.connect(AppState.audioProcessor);
            AppState.audioProcessor.connect(AppState.audioContext.destination);

            AppState.micActive = true;
            btnToggleMic.classList.add("active");
            btnToggleMic.querySelector("i").className = "fa-solid fa-microphone";
            micToggleText.textContent = "Classroom Mic On";
        } catch (err) {
            console.error("Microphone access error:", err);
            alert("Could not access classroom microphone: " + err.message);
        }
    }

    function stopMicrophone() {
        AppState.micActive = false;
        if (AppState.audioProcessor) {
            AppState.audioProcessor.disconnect();
            AppState.audioProcessor = null;
        }
        if (AppState.audioContext) {
            AppState.audioContext.close();
            AppState.audioContext = null;
        }
        if (AppState.micStream) {
            AppState.micStream.getTracks().forEach(t => t.stop());
            AppState.micStream = null;
        }

        btnToggleMic.classList.remove("active");
        btnToggleMic.querySelector("i").className = "fa-solid fa-microphone-slash";
        micToggleText.textContent = "Classroom Mic Off";
    }

    // =========================================================================
    // 7. AI Teaching & Study Coach: Audio Advice, Practice Quiz, Concept Map
    // =========================================================================
    btnPlayAudio.addEventListener("click", async () => {
        const text = aiSuggestionText.textContent;
        btnPlayAudio.querySelector("span").textContent = "Synthesizing...";

        try {
            const res = await fetch("/api/ai/tts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });

            const data = await res.json();
            if (data.success && data.audio_url) {
                ttsAudioPlayer.src = `${data.audio_url}?t=${Date.now()}`;
                ttsAudioPlayer.play().catch(e => {
                    speakTextWithBrowser(text);
                });
            } else {
                speakTextWithBrowser(text);
            }
        } catch (err) {
            speakTextWithBrowser(text);
        } finally {
            btnPlayAudio.querySelector("span").textContent = "Listen Study Tip";
        }
    });

    function speakTextWithBrowser(text) {
        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }

    // Practice Quiz & Code Sandbox
    btnGenerateCode.addEventListener("click", async () => {
        codeModal.classList.add("open");
        await requestNewCodeExercise();
    });

    btnRegenerateCode.addEventListener("click", async () => {
        await requestNewCodeExercise();
    });

    async function requestNewCodeExercise() {
        terminalOutput.textContent = "Generating study practice challenge for " + AppState.activeCourse + "...";
        try {
            const res = await fetch("/api/ai/generate_code", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ topic: AppState.activeCourse })
            });
            const data = await res.json();
            if (data.success) {
                codeEditor.value = data.code;
                codeGeneratorLabel.textContent = `Generator: ${data.generator}`;
                terminalOutput.textContent = 'Practice challenge ready. Click "Execute Python Code" to test student solution.';
            }
        } catch (err) {
            terminalOutput.textContent = "Failed to generate challenge: " + err.message;
        }
    }

    btnRunCode.addEventListener("click", async () => {
        const code = codeEditor.value;
        terminalOutput.textContent = "Running student code verification in sandbox process...";
        execMeta.textContent = "Running...";

        try {
            const res = await fetch("/api/ai/execute_code", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code: code })
            });
            const data = await res.json();

            execMeta.textContent = `Runtime: ${data.execution_time_ms} ms`;
            if (data.success) {
                terminalOutput.style.color = "#10b981";
                terminalOutput.textContent = data.stdout || "Code verified successfully with no errors.";
            } else {
                terminalOutput.style.color = "#ef4444";
                terminalOutput.textContent = (data.stderr || "Execution error.") + (data.stdout ? "\n" + data.stdout : "");
            }
        } catch (err) {
            terminalOutput.style.color = "#ef4444";
            terminalOutput.textContent = "Sandbox Execution Error: " + err.message;
        }
    });

    btnCloseCodeModal.addEventListener("click", () => codeModal.classList.remove("open"));
    btnCloseCodeBtn.addEventListener("click", () => codeModal.classList.remove("open"));

    // Pedagogical Concept Map
    btnConceptDiagram.addEventListener("click", async () => {
        diagramModal.classList.add("open");
        diagramContainer.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Rendering pedagogical concept diagram...';
        try {
            const res = await fetch("/api/ai/concept_diagram", { method: "POST" });
            const data = await res.json();
            if (data.success && data.svg_markup) {
                diagramContainer.innerHTML = data.svg_markup;
            } else {
                diagramContainer.innerHTML = "<p>Diagram unavailable</p>";
            }
        } catch (err) {
            diagramContainer.innerHTML = "<p>Failed to load diagram: " + err.message + "</p>";
        }
    });

    btnCloseDiagramModal.addEventListener("click", () => diagramModal.classList.remove("open"));
    btnDismissDiagram.addEventListener("click", () => diagramModal.classList.remove("open"));

    // =========================================================================
    // 8. Settings & Gemini Configuration
    // =========================================================================
    btnSettings.addEventListener("click", () => {
        inputGeminiApiKey.value = AppState.geminiApiKey;
        inputSustainedThreshold.value = AppState.sustainedThresholdSeconds;
        settingsModal.classList.add("open");
    });
    aiModePill.addEventListener("click", () => btnSettings.click());

    btnCloseSettings.addEventListener("click", () => settingsModal.classList.remove("open"));
    btnCancelSettings.addEventListener("click", () => settingsModal.classList.remove("open"));

    btnToggleKeyVis.addEventListener("click", () => {
        const type = inputGeminiApiKey.getAttribute("type") === "password" ? "text" : "password";
        inputGeminiApiKey.setAttribute("type", type);
        btnToggleKeyVis.querySelector("i").className = type === "password" ? "fa-solid fa-eye" : "fa-solid fa-eye-slash";
    });

    btnSaveSettings.addEventListener("click", async () => {
        const key = inputGeminiApiKey.value.trim();
        const threshold = parseFloat(inputSustainedThreshold.value) || 4.0;

        AppState.geminiApiKey = key;
        AppState.sustainedThresholdSeconds = threshold;
        localStorage.setItem("gemini_api_key", key);
        localStorage.setItem("sustained_threshold", threshold.toString());

        try {
            await fetch("/api/settings/gemini_key", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: key })
            });
        } catch (err) {
            console.error("Error saving API key:", err);
        }

        settingsModal.classList.remove("open");
        alert("Classroom settings updated successfully.");
    });

    // Reset Lecture Session
    btnResetSession.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to start a new classroom period and reset attendance logs?")) return;

        try {
            const res = await fetch("/api/attendance/reset", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                if (AppState.chart) {
                    AppState.chart.data.labels = [];
                    AppState.chart.data.datasets.forEach(d => d.data = []);
                    AppState.chart.update();
                }
                alert("New lecture session started. Attendance and metrics reset.");
            }
        } catch (err) {
            console.error("Error resetting session:", err);
        }
    });

    // =========================================================================
    // 9. Gradebook Exports (CSV & JSON)
    // =========================================================================
    btnExportCSV.addEventListener("click", () => {
        window.location.href = "/api/export_attendance?format=csv";
    });

    btnExportJSON.addEventListener("click", () => {
        window.location.href = "/api/export_attendance?format=json";
    });

    btnExportAnalytics.addEventListener("click", () => {
        window.location.href = "/api/export_analytics?format=csv";
    });

    // =========================================================================
    // 10. Startup & Initial Sync
    // =========================================================================
    initChart();

    fetch("/api/status")
        .then(res => res.json())
        .then(status => {
            if (status.detector_in_use) hudDetectorLabel.textContent = status.detector_in_use;
            if (status.sustained_threshold_seconds) {
                AppState.sustainedThresholdSeconds = status.sustained_threshold_seconds;
                hudPresenceThreshold.textContent = `Verification: ${status.sustained_threshold_seconds.toFixed(1)}s Continuous Study Focus`;
            }
        })
        .catch(() => {});

    if (AppState.geminiApiKey) {
        fetch("/api/settings/gemini_key", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: AppState.geminiApiKey })
        }).catch(() => {});
    }

    setInterval(fetchTelemetry, AppState.pollIntervalMs);
});
