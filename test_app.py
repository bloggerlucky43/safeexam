"""
Streamlit WebRTC Frontend for Multi-Modal Biometric Liveness Detection.
Captures live browser video/audio, routes it to Face/Voice modules, and displays gauge charts.
"""
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import cv2
import time
from collections import deque
import plotly.graph_objects as go

from src.face_module import FaceLivenessDetector
from src.voice_module import VoiceLivenessDetector
from src.fusion import fuse_scores

# --- Page Configuration ---
st.set_page_config(
    page_title="Liveness Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme styling injected via CSS
st.markdown("""
    <style>
    .reportview-container {
        background: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'history' not in st.session_state:
    st.session_state['history'] = deque(maxlen=5) # Keep last 5 attempts
if 'face_detector' not in st.session_state:
    st.session_state['face_detector'] = FaceLivenessDetector()
if 'voice_detector' not in st.session_state:
    st.session_state['voice_detector'] = VoiceLivenessDetector()

# Configuration for WebRTC connecting (deals with STUN/TURN for permissions)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Callback classes for stream processors ---
class VideoProcessor:
    def __init__(self):
        self.frame_buffer = []
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # We passively collect frames for analysis when triggered,
        # but for real-time feedback, we could detect the face here and draw a bounding box.
        # Since MediaPipe is already processing during the analyze block, we will just echo for now
        # to save UI rendering threads.
        
        # Save frame to buffer if we are "recording"
        if len(self.frame_buffer) < 90: # Cap at ~3 seconds (30fps)
            self.frame_buffer.append(img)
        else:
            self.frame_buffer.pop(0)
            self.frame_buffer.append(img)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class AudioProcessor:
    def __init__(self):
        self.audio_buffer = np.array([])
        
    def recv(self, frame):
        sound = frame.to_ndarray()
        
        # Keep last 3 seconds of audio at 48kHz (roughly 144,000 samples)
        if len(self.audio_buffer) < 144000:
            self.audio_buffer = np.append(self.audio_buffer, sound)
        else:
            self.audio_buffer = np.append(self.audio_buffer[len(sound):], sound)
            
        return av.AudioFrame.from_ndarray(sound, layout='mono')

# --- Helper Methods ---
def draw_gauge(score, status):
    """Draws a Plotly Dial Gauge indicating the final liveness score."""
    color = "green" if score > 0.8 else "red"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Liveness Score<br><span style='font-size:0.8em;color:{color}'>{status}</span>"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps' : [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "darkgray"}],
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
        }
    ))
    return fig

# --- Layout Details ---
st.title("🛡️ Multi-Modal Biometric Liveness")
st.markdown("Ensure your camera and microphone are authorized. The system uses spatial texture (face) and spectral distribution (voice) to defeat spoofing attacks.")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_ctx = webrtc_streamer(
        key="liveness-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True
    )

with col2:
    st.subheader("Verification Dashboard")
    gauge_placeholder = st.empty()
    
    if st.button("🔍 Verify Liveness (Capture 3s)", use_container_width=True):
        if not webrtc_ctx.state.playing:
            st.error("Please start the video/audio stream first.")
        else:
            with st.spinner("Analyzing captured stream..."):
                video_proc = webrtc_ctx.video_processor
                audio_proc = webrtc_ctx.audio_processor
                
                # Retrieve buffers
                if video_proc and audio_proc:
                    frames = video_proc.frame_buffer
                    audio_data = audio_proc.audio_buffer
                    
                    if len(frames) == 0 or len(audio_data) == 0:
                        st.warning("Insufficient data. Please stay in frame and speak briefly.")
                    else:
                        # 1. Evaluate Vision
                        face_scores = []
                        for f in frames[::3]:  # Sample every 3rd frame
                            face_scores.append(st.session_state['face_detector'].get_liveness_score(f))
                        
                        f_score = float(np.median(face_scores)) if face_scores else 0.0
                        
                        # 2. Evaluate Audio
                        # WebRTC streams audio at 48000 Hz typically
                        v_score = st.session_state['voice_detector'].analyze_audio(audio_data, original_sr=48000)
                        
                        # 3. Fuse
                        final_score, status = fuse_scores(f_score, v_score)
                        
                        # Render Gauge
                        gauge_placeholder.plotly_chart(draw_gauge(final_score, status), use_container_width=True)
                        
                        # Log Attempt
                        timestamp = time.strftime("%H:%M:%S")
                        st.session_state['history'].appendleft({
                            "time": timestamp,
                            "score": f"{final_score*100:.1f}%",
                            "status": status
                        })

# --- Sidebar Logging ---
with st.sidebar:
    st.header("📋 Access Logs")
    st.write("Last 5 attempts:")
    for log in st.session_state['history']:
        icon = "✅" if "SUCCESS" in log['status'] else "❌"
        st.markdown(f"**{log['time']}** - {icon} {log['score']}  \n*{log['status']}*")
        st.divider()
