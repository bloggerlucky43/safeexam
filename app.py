import streamlit as st
import numpy as np
import os
import cv2
import librosa
import uuid
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time
import json
import requests
from src.config import settings
from src.face_module import FaceLivenessDetector

# Backend API configuration
API_URL = settings.API_URL

# Inline Lucide Vector SVGs for premium rendering without generic emojis
ICONS = {
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield" style="vertical-align: middle; margin-right: 8px; color: #58a6ff;"><path d="M20 13c0 5-3.5 7.5-7.66 9.7a1 1 0 0 1-.68 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 .76-.97l8-2a1 1 0 0 1 .48 0l8 2A1 1 0 0 1 20 6z"/></svg>',
    "user": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-user" style="vertical-align: middle; margin-right: 8px; color: #8b949e;"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "camera": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-camera" style="vertical-align: middle; margin-right: 8px; color: #58a6ff;"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>',
    "sheet": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-clipboard-list" style="vertical-align: middle; margin-right: 8px; color: #58a6ff;"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/></svg>',
    "video": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-video" style="vertical-align: middle; margin-right: 6px; color: #58a6ff;"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>',
    "audio": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-volume-2" style="vertical-align: middle; margin-right: 6px; color: #58a6ff;"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-activity" style="vertical-align: middle; margin-right: 8px; color: #58a6ff;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-alert-triangle" style="vertical-align: middle; margin-right: 8px; color: #f85149;"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
    "check": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-check-circle" style="vertical-align: middle; margin-right: 8px; color: #3fb950;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "trash": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash-2" style="vertical-align: middle;"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>',
    "plus": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-plus" style="vertical-align: middle; margin-right: 6px;"><line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12/></svg>'
}

def api_get_questions():
    try:
        res = requests.get(f"{API_URL}/exams/questions")
        if res.status_code == 200:
            return res.json().get("questions", [])
    except Exception as e:
        pass
    return ["Question 1: Explain the primary mechanisms of biometric spoofing in voice systems."]

# Page Configuration
st.set_page_config(page_title="SafeExam - Biometric Exam Proctoring", layout="wide")

# Custom Premium Styling Injection
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #f0f6fc !important;
        font-weight: 600 !important;
        letter-spacing: -0.3px;
    }
    
    /* Premium Cards */
    .premium-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .premium-card-header {
        border-bottom: 1px solid #21262d;
        padding-bottom: 12px;
        margin-bottom: 16px;
        font-weight: 600;
        font-size: 1.15em;
        color: #ffffff;
    }
    
    /* Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    .badge-success {
        background-color: rgba(46, 164, 79, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 164, 79, 0.3);
    }
    
    .badge-failed {
        background-color: rgba(248, 81, 73, 0.15);
        color: #ff7b72;
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    
    .badge-warn {
        background-color: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.3);
    }
    
    /* Styled Input/TextArea */
    textarea {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        border-radius: 8px !important;
    }
    
    textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.15) !important;
    }
    
    /* Streamlit Buttons style adaptation */
    div.stButton > button {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        border: 1px solid rgba(240, 246, 252, 0.1) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    div.stButton > button:hover {
        background-color: #388bfd !important;
        border-color: #58a6ff !important;
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

# Define WebRTC Video and Audio Processors
class VideoProcessor:
    def __init__(self, session_id):
        self.session_id = session_id
        self.out = None
        self.face_detector = FaceLivenessDetector()
        self.state = self.face_detector.create_default_state()
        self.frame_count = 0
        self.warnings = []
        self.multiple_faces_detected = False
        self.no_face_detected = False
        self.movement_status = "Focused"
        self.recording = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if not self.recording:
            return frame
            
        img = frame.to_ndarray(format="bgr24")
        # Resize to standard 360p (480x360) to further reduce file size, making processing, merge, and upload times even faster
        img = cv2.resize(img, (480, 360))
        h, w, _ = img.shape
        
        # Initialize video writer (using standard mp4v codec for cross-platform compatibility)
        if self.out is None:
            os.makedirs("data/uploads", exist_ok=True)
            self.video_path = f"data/uploads/video_{self.session_id}.mp4"
            self.out = cv2.VideoWriter(
                self.video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                10.0,  # 10 fps
                (w, h)
            )
            
        self.frame_count += 1
        # Run face checks every 5 frames to conserve resources
        if self.frame_count % 5 == 0:
            try:
                score, telemetry = self.face_detector.analyze_face_with_telemetry(img, self.state)
                self.multiple_faces_detected = telemetry.get("multiple_faces", False)
                self.no_face_detected = telemetry.get("no_face", False)
                self.movement_status = telemetry.get("movement_status", "Focused")
                
                # Append warnings with local timestamp
                timestamp = time.strftime("%H:%M:%S")
                if self.multiple_faces_detected:
                    self.warnings.append(f"[{timestamp}] Multiple Faces Detected")
                if self.no_face_detected:
                    self.warnings.append(f"[{timestamp}] No Face Detected")
                elif self.movement_status in ["Looking Left", "Looking Right"]:
                    self.warnings.append(f"[{timestamp}] Looking Away ({self.movement_status})")
            except Exception as e:
                pass
                
        # Draw on-screen real-time proctoring alerts
        if self.multiple_faces_detected:
            cv2.putText(img, "ALERT: MULTIPLE FACES DETECTED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.no_face_detected:
            cv2.putText(img, "ALERT: NO FACE DETECTED!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.movement_status in ["Looking Left", "Looking Right"]:
            cv2.putText(img, f"ALERT: LOOKING AWAY ({self.movement_status.upper()})!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.out is not None:
            self.out.write(img)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")
        
    def stop(self):
        self.recording = False
        if self.out is not None:
            self.out.release()
            self.out = None

class AudioProcessor:
    def __init__(self, session_id):
        self.session_id = session_id
        self.audio_buffer = []
        self.sample_rate = 48000
        self.recording = True
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if not self.recording:
            return frame
        self.sample_rate = frame.sample_rate
        sound = frame.to_ndarray()
        if sound.ndim > 1:
            sound = sound[0]
        self.audio_buffer.append(sound)
        return frame
        
    def stop_and_save(self):
        self.recording = False
        if not self.audio_buffer:
            return None
        os.makedirs("data/uploads", exist_ok=True)
        audio_path = f"data/uploads/audio_{self.session_id}.wav"
        all_samples = np.concatenate(self.audio_buffer)
        import scipy.io.wavfile as wav
        wav.write(audio_path, self.sample_rate, all_samples)
        return audio_path

# Premium Header Layout (Using Clean Vector SVG Shield Icon)
st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 28px;">
        <div style="background-color: rgba(31, 111, 235, 0.1); padding: 14px; border-radius: 12px; border: 1px solid rgba(31, 111, 235, 0.25); display: flex; align-items: center; justify-content: center; color: #58a6ff;">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield"><path d="M20 13c0 5-3.5 7.5-7.66 9.7a1 1 0 0 1-.68 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 .76-.97l8-2a1 1 0 0 1 .48 0l8 2A1 1 0 0 1 20 6z"/></svg>
        </div>
        <div>
            <h1 style="margin: 0; font-size: 2.1em; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; line-height: 1.2;">SafeExam</h1>
            <p style="margin: 0; color: #8b949e; font-size: 1em;">Intelligent Multimodal Proctoring Platform</p>
        </div>
    </div>
""", unsafe_allow_html=True)

role = st.sidebar.selectbox("Select Role / Portal", ["Student - Exam Portal", "Admin - Monitoring Dashboard"])

# ----------------- STUDENT PORTAL -----------------
if role == "Student - Exam Portal":
    st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 18px;">
            {ICONS['user']}
            <h2 style="margin: 0; font-size: 1.5em;">Student Examination Portal</h2>
        </div>
    """, unsafe_allow_html=True)
    
    if "student_id" not in st.session_state:
        st.session_state["student_id"] = str(uuid.uuid4())[:8]
    if "student_registered" not in st.session_state:
        st.session_state["student_registered"] = False
        
    student_id = st.session_state["student_id"]
    
    if not st.session_state["student_registered"]:
        st.markdown(f"""
            <div class="premium-card">
                <div class="premium-card-header">Candidate Registration</div>
                <div style="color: #8b949e; font-size: 0.95em; line-height: 1.5; margin-bottom: 12px;">
                    Please input your official details below to register and unlock the examination sheet.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        name_input = st.text_input("Full Name", placeholder="e.g. Alice Smith")
        matric_input = st.text_input("Matric Number", placeholder="e.g. SCI/2026/0042")
        
        if st.button("Register & Start Exam", use_container_width=True):
            if name_input.strip() and matric_input.strip():
                st.session_state["student_name"] = name_input.strip()
                st.session_state["matric_number"] = matric_input.strip()
                st.session_state["student_registered"] = True
                st.rerun()
            else:
                st.error("Please enter a valid name and matric number.")
        st.stop()
        
    student_name = st.session_state["student_name"]
    matric_number = st.session_state["matric_number"]
    
    st.markdown(f"""
        <div class="premium-card" style="padding: 16px 20px; border-left: 4px solid #1f6feb;">
            <div style="font-size: 1.05em; font-weight: 500; color: #ffffff;">
                Candidate: {student_name} ({matric_number})
            </div>
            <div style="font-size: 0.9em; color: #8b949e; margin-top: 4px;">
                Session ID: <code>{student_id}</code> | Ensure camera and microphone permissions are active. Your face and voice are monitored in real-time.
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 12px; gap: 8px;">
                {ICONS['camera']}
                <span style="font-size: 1.15em; font-weight: 500; color: #f0f6fc;">Live Proctoring Stream</span>
            </div>
        """, unsafe_allow_html=True)
        
        webrtc_ctx = webrtc_streamer(
            key="liveness_exam_stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=lambda: VideoProcessor(student_id),
            audio_processor_factory=lambda: AudioProcessor(student_id),
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
        )

    with col2:
        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 12px; gap: 8px;">
                {ICONS['sheet']}
                <span style="font-size: 1.15em; font-weight: 500; color: #f0f6fc;">Examination Response Sheet</span>
            </div>
        """, unsafe_allow_html=True)
        
        if webrtc_ctx.state.playing:
            questions_list = api_get_questions()
            responses = {}
            for i, q in enumerate(questions_list):
                responses[f"Question {i+1}"] = st.text_area(f"Question {i+1}: {q}", height=120, key=f"ans_input_{i}")
                
            if st.button("Submit Exam Data", use_container_width=True):
                video_proc = webrtc_ctx.video_processor
                audio_proc = webrtc_ctx.audio_processor
                
                if video_proc and audio_proc:
                    with st.spinner("Finalizing media streams and uploading data..."):
                        # Stop streams and write local files
                        video_proc.stop()
                        audio_path = audio_proc.stop_and_save()
                        video_path = video_proc.video_path
                        warnings_list = video_proc.warnings
                        
                        if os.path.exists(video_path) and audio_path and os.path.exists(audio_path):
                            try:
                                with open(video_path, 'rb') as vf, open(audio_path, 'rb') as af:
                                    files = {
                                        'video_file': ('video.mp4', vf, 'video/mp4'),
                                        'audio_file': ('audio.wav', af, 'audio/wav')
                                    }
                                    data = {
                                        'student_id': student_id,
                                        'student_name': student_name,
                                        'matric_number': matric_number,
                                        'responses': json.dumps(responses),
                                        'warnings': json.dumps(warnings_list)
                                    }
                                    res = requests.post(f"{API_URL}/exams/submit", files=files, data=data)
                                    if res.status_code == 200:
                                        res_data = res.json()
                                        st.success("🎉 Exam Submitted Successfully!")
                                        
                                        score = res_data.get('score', 0.0)
                                        status = res_data.get('verification_status', 'Unknown')
                                        
                                        col_s1, col_s2 = st.columns(2)
                                        with col_s1:
                                            st.metric("Identity Confidence Score", f"{score * 100:.1f}%")
                                        with col_s2:
                                            if "SUCCESS" in status:
                                                st.markdown('<span class="status-badge badge-success">VERIFICATION PASSED</span>', unsafe_allow_html=True)
                                            else:
                                                st.markdown('<span class="status-badge badge-failed">VERIFICATION FLAGGED</span>', unsafe_allow_html=True)
                                                st.error("Liveness verification did not satisfy thresholds. Results flagged for manual review.")
                                    else:
                                        st.error(f"Failed to submit to backend API: {res.text}")
                            except Exception as e:
                                st.error(f"Error submitting data to backend: {e}")
                        else:
                            st.error("Unable to locate written video/audio recordings on disk.")
                else:
                    st.error("Proctoring streams are not active. Press 'Start' to unlock the response sheet.")
        else:
            st.warning("⚠️ You must click the 'Start' button on the camera feed to unlock the examination questions.")

# ----------------- ADMIN DASHBOARD -----------------
elif role == "Admin - Monitoring Dashboard":
    st.header("Administrator Monitoring Portal")
    
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False
        
    if not st.session_state["admin_authenticated"]:
        st.warning("🔒 Authenticated login required for authorized proctors.")
        pwd = st.text_input("Enter Admin Password", type="password")
        if st.button("Authenticate Proctor"):
            try:
                auth_res = requests.post(f"{API_URL}/token", data={"username": "admin", "password": pwd})
                if auth_res.status_code == 200:
                    st.session_state["admin_authenticated"] = True
                    st.session_state["access_token"] = auth_res.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Authentication failed. Incorrect password.")
            except Exception as e:
                st.error("Failed to establish link with backend authentication endpoint.")
    else:
        # Logout button in sidebar
        if st.sidebar.button("Logout Admin Portal"):
            st.session_state["admin_authenticated"] = False
            st.session_state["access_token"] = None
            st.rerun()

        # Load Admin Panel Components in Tabs
        tab1, tab2 = st.tabs(["Proctor Review Dashboard", "Configure Questions"])
        
        headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}

        # TAB 1: PROCTOR LOGS AND REPLAYS
        with tab1:
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 12px; gap: 8px;">
                    {ICONS['activity']}
                    <span style="font-size: 1.15em; font-weight: 500; color: #f0f6fc;">Exam Submissions & Biometric Verification Audit</span>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                sub_res = requests.get(f"{API_URL}/exams/submissions", headers=headers)
                if sub_res.status_code == 200:
                    submissions = sub_res.json().get("submissions", [])
                    if not submissions:
                        st.info("No exam submissions recorded yet.")
                    else:
                        # Build candidate selector labels
                        options = []
                        sub_dict = {}
                        for sub in submissions:
                            label = f"{sub['student_name']} ({sub['matric_number']}) - {sub['status']} ({sub['score']*100:.1f}%)"
                            options.append(label)
                            sub_dict[label] = sub
                            
                        selected_label = st.selectbox("Select Candidate Submission to Audit", options)
                        selected_sub = sub_dict[selected_label]
                        
                        st.divider()
                        
                        col_info, col_telemetry = st.columns([1, 1])
                        
                        with col_info:
                            st.markdown(f"### Candidate Details")
                            st.write(f"**Name:** {selected_sub['student_name']}")
                            st.write(f"**Matric:** {selected_sub['matric_number']}")
                            st.write(f"**Session ID:** `{selected_sub['session_id']}`")
                            
                            sub_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(selected_sub['submitted_at']))
                            st.write(f"**Submitted At:** {sub_time}")
                            
                            # Score indicators
                            l_score = selected_sub['score']
                            l_status = selected_sub['status']
                            if "SUCCESS" in l_status:
                                badge = '<span class="status-badge badge-success">BIOMETRIC LIVENESS PASSED</span>'
                            else:
                                badge = '<span class="status-badge badge-failed">BIOMETRIC LIVENESS FAILED</span>'
                            st.markdown(f"**Score:** {l_score*100:.2f}%  |  {badge}", unsafe_allow_html=True)
                            
                        with col_telemetry:
                            st.markdown("### Proctor Warning Logs")
                            warnings_raw = selected_sub.get("warnings", "[]")
                            try:
                                warnings_list = json.loads(warnings_raw)
                            except:
                                warnings_list = [warnings_raw] if warnings_raw else []
                                
                            if not warnings_list:
                                st.markdown(f"""
                                    <div style="background-color: rgba(46, 164, 79, 0.1); border: 1px solid rgba(46, 164, 79, 0.25); padding: 12px 16px; border-radius: 8px; color: #3fb950; display: flex; align-items: center; gap: 8px; font-size: 0.95em;">
                                        {ICONS['check']}
                                        <span>No suspicious behavioral alerts detected during exam.</span>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                for warn in warnings_list:
                                    st.markdown(f"""
                                        <div style="color: #ff7b72; display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 0.95em;">
                                            {ICONS['alert']}
                                            <span>{warn}</span>
                                        </div>
                                    """, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Display Student's Answers
                        st.markdown("### Written Responses")
                        try:
                            responses_dict = json.loads(selected_sub['response'])
                            for q, ans in responses_dict.items():
                                st.markdown(f"**{q}**")
                                st.text_area("", value=ans, height=100, disabled=True, key=f"ans_view_{q}_{selected_sub['session_id']}")
                        except Exception as e:
                            st.write("Could not decode answers raw text:", selected_sub['response'])

                        st.divider()

                        # Audio and Video Replay Section
                        st.markdown("### Recorded Video & Audio Session Audit")
                        col_v, col_a = st.columns(2)
                        
                        with col_v:
                            st.markdown(f"""
                                <div style="display: flex; align-items: center; margin-bottom: 12px; gap: 4px;">
                                    {ICONS['video']}
                                    <span style="font-size: 1.05em; font-weight: 500; color: #f0f6fc;">Video Replay Log</span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            v_path = selected_sub['video_path']
                            if v_path:
                                if v_path.startswith("http://") or v_path.startswith("https://"):
                                    video_url = v_path
                                else:
                                    v_filename = os.path.basename(v_path)
                                    video_url = f"{API_URL}/uploads/exams/{v_filename}"
                                st.video(video_url)
                            else:
                                st.info("No video log recorded.")
                                
                        with col_a:
                            st.markdown(f"""
                                <div style="display: flex; align-items: center; margin-bottom: 12px; gap: 4px;">
                                    {ICONS['audio']}
                                    <span style="font-size: 1.05em; font-weight: 500; color: #f0f6fc;">Audio Replay Log</span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            a_path = selected_sub['audio_path']
                            if a_path:
                                if a_path.startswith("http://") or a_path.startswith("https://"):
                                    audio_url = a_path
                                else:
                                    a_filename = os.path.basename(a_path)
                                    audio_url = f"{API_URL}/uploads/exams/{a_filename}"
                                st.audio(audio_url)
                            else:
                                st.info("No audio log recorded.")
                else:
                    st.error("Failed to fetch submissions from API.")
            except Exception as e:
                st.error(f"Error fetching submissions: {e}")

        # TAB 2: EDIT EXAM QUESTIONS
        with tab2:
            st.subheader("Question Bank Configuration")
            try:
                # Load questions from API
                q_res = requests.get(f"{API_URL}/exams/questions")
                if q_res.status_code == 200:
                    current_qs = q_res.json().get("questions", [])
                    
                    st.write("Modify the question bank below. Click 'Save Exam Questions' to propagate changes to the backend.")
                    
                    # Store questions in session_state for dynamic CRUD
                    if "questions_list" not in st.session_state:
                        st.session_state["questions_list"] = current_qs
                        
                    # Display existing questions with Delete option
                    updated_list = []
                    for idx, q in enumerate(st.session_state["questions_list"]):
                        col_q, col_del = st.columns([12, 1])
                        with col_q:
                            edited_q = st.text_input(f"Question {idx+1}", value=q, key=f"edit_q_{idx}")
                            updated_list.append(edited_q)
                        with col_del:
                            st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True) # offset spacing
                            if st.button("🗑️", key=f"del_q_{idx}"):
                                st.session_state["questions_list"].pop(idx)
                                st.rerun()
                                
                    # Add new question input
                    new_q = st.text_input("Add New Question", placeholder="Type a new question...")
                    if st.button("Add to Bank"):
                        if new_q.strip():
                            st.session_state["questions_list"].append(new_q.strip())
                            st.rerun()
                            
                    st.divider()
                    
                    # Save back to API
                    if st.button("Save Exam Questions", use_container_width=True):
                        # Use updated text inputs
                        payload = {"questions": [q for q in updated_list if q.strip()]}
                        save_res = requests.post(f"{API_URL}/exams/questions", headers=headers, json=payload)
                        if save_res.status_code == 200:
                            st.success("Questions updated successfully!")
                            st.session_state["questions_list"] = payload["questions"]
                        else:
                            st.error(f"Failed to update questions: {save_res.text}")
                else:
                    st.error("Failed to load questions from backend API.")
            except Exception as e:
                st.error(f"Connection failure: {e}")
