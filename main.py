import os
import uuid
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.config import settings
from src.database import get_db, init_db, StudentSession, ExamSubmission
from src.face_module import FaceLivenessDetector
from src.voice_module import VoiceLivenessDetector
from src.fusion import fuse_scores
from src.storage import storage
from src.utils import log
from src.session_store import (
    get_exam_questions, 
    set_exam_questions, 
    submit_exam_response, 
    get_all_exam_submissions
)
import cv2
import numpy as np
import librosa

# Security & Auth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Biometric Liveness API", version="1.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static folder for video/audio logs
os.makedirs("data/uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="data/uploads"), name="uploads")

# Global models (Thread-safe refactoring pending in Phase 4)
face_detector = FaceLivenessDetector()
voice_detector = VoiceLivenessDetector()

# Initialize DB on startup
@app.on_event("startup")
def on_startup():
    init_db()
    
    # Analyze loaded admin password for diagnostic purposes without printing the secret
    raw_pwd = settings.ADMIN_PASSWORD or ""
    clean_pwd = settings.cleaned_admin_password
    has_quotes = raw_pwd.startswith(('"', "'")) and raw_pwd.endswith(('"', "'"))
    has_whitespace = len(raw_pwd) != len(raw_pwd.strip())
    is_default = (clean_pwd == "admin123")
    
    log.info("API Startup Complete", 
             database_url=settings.DATABASE_URL,
             admin_pwd_length=len(raw_pwd),
             admin_pwd_is_default=is_default,
             admin_pwd_has_quotes=has_quotes,
             admin_pwd_has_whitespace=has_whitespace)

# --- AUTH UTILS ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# --- SCHEMAS ---
class VerificationResponse(BaseModel):
    session_id: str
    status: str
    message: str

# --- ENDPOINTS ---

@app.post("/token")
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    # Sanitize inputs and settings (strip trailing/leading whitespace and quotes)
    configured_password = settings.cleaned_admin_password
    input_password = form_data.password.strip().strip("'").strip('"') if form_data.password else ""
    
    if form_data.username == "admin" and input_password == configured_password:
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Incorrect username or password")

@app.get("/")
async def root():
    return {"status": "online", "timestamp": datetime.utcnow().isoformat()}

def process_liveness_task(session_id: str, video_path: str, audio_path: str, student_name: str, matric_number: str):
    """Heavy lifting background task for liveness detection."""
    db = next(get_db())
    try:
        log.info("Processing Liveness Task", session_id=session_id)
        
        # 1. Face Analysis
        cap = cv2.VideoCapture(video_path)
        frame_scores = []
        face_state = face_detector.create_default_state()
        while cap.isOpened() and len(frame_scores) < 30:
            ret, frame = cap.read()
            if not ret: break
            score, telemetry = face_detector.analyze_face_with_telemetry(frame, state=face_state)
            frame_scores.append(score)
        cap.release()
        
        face_score = float(np.median(frame_scores)) if frame_scores else 0.0
        
        # 2. Voice Analysis
        y, sr = librosa.load(audio_path, sr=None)
        voice_score = voice_detector.analyze_audio(y, original_sr=sr)
        
        # 3. Fusion
        final_score, status = fuse_scores(face_score, voice_score)
        
        # 4. Update Database
        session = db.query(StudentSession).filter(StudentSession.session_id == session_id).first()
        if not session:
            session = StudentSession(session_id=session_id)
            db.add(session)
        
        session.last_updated = datetime.utcnow().timestamp()
        session.movement_status = status
        session.student_name = student_name
        session.matric_number = matric_number
        db.commit()
        
        log.info("Liveness Task Completed", session_id=session_id, final_score=final_score)
        
    except Exception as e:
        log.error("Liveness Task Failed", session_id=session_id, error=str(e))
    finally:
        # Cleanup temp files from storage if needed (storage.delete depends on implementation)
        db.close()

def save_upload_file_temporarily(upload_file: UploadFile) -> str:
    temp_dir = "data/uploads/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"upload_{uuid.uuid4()}_{upload_file.filename}")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_file_path

@app.post("/verify", response_model=VerificationResponse)
@limiter.limit("10/minute")
async def verify_liveness(
    request: Request,
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    student_name: str = "Unknown",
    matric_number: str = "Unknown"
):
    session_id = str(uuid.uuid4())
    
    # Save files using storage abstraction
    video_ext = os.path.splitext(video_file.filename)[1] if video_file.filename else ".mp4"
    audio_ext = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
    
    temp_video = f"temp_{session_id}{video_ext}"
    temp_audio = f"temp_{session_id}{audio_ext}"
    
    # Save locally to avoid Windows Permission Denied locks
    temp_video_path = save_upload_file_temporarily(video_file)
    temp_audio_path = save_upload_file_temporarily(audio_file)
    
    video_path = storage.save(temp_video_path, f"raw/{temp_video}") 
    audio_path = storage.save(temp_audio_path, f"raw/{temp_audio}")
    
    # Clean up local temporary files
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
    background_tasks.add_task(
        process_liveness_task, 
        session_id, video_path, audio_path, student_name, matric_number
    )
    
    return VerificationResponse(
        session_id=session_id,
        status="Processing",
        message="Verification queued in background."
    )

# --- EXAM ENDPOINTS ---

class QuestionsPayload(BaseModel):
    questions: List[str]

@app.get("/exams/questions")
async def get_questions():
    questions = get_exam_questions()
    return {"questions": questions}

@app.post("/exams/questions")
async def post_questions(payload: QuestionsPayload, username: str = Depends(get_current_user)):
    if username != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    set_exam_questions(payload.questions)
    return {"status": "success", "message": "Questions updated successfully."}

@app.get("/exams/submissions")
async def get_submissions(username: str = Depends(get_current_user)):
    if username != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    submissions = get_all_exam_submissions()
    return {"submissions": submissions}

def merge_video_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    try:
        # Multiplexing & Transcoding: Encode video to H.264 (libx264) and audio to AAC
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return True
        else:
            log.error("FFmpeg merge failed", stdout=result.stdout, stderr=result.stderr)
            return False
    except Exception as e:
        log.error("FFmpeg merge exception", error=str(e))
        return False

@app.post("/exams/submit")
async def submit_exam(
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    student_id: str = Form(...),
    student_name: str = Form(...),
    matric_number: str = Form(...),
    responses: str = Form(...),
    warnings: str = Form(...)
):
    try:
        # Save uploaded files to local temporary files to bypass Windows file locks
        temp_video_path = save_upload_file_temporarily(video_file)
        temp_audio_path = save_upload_file_temporarily(audio_file)
        
        # 1. Face Analysis on the local video file
        cap = cv2.VideoCapture(temp_video_path)
        frame_scores = []
        face_state = face_detector.create_default_state()
        while cap.isOpened() and len(frame_scores) < 60:
            ret, frame = cap.read()
            if not ret: break
            score, telemetry = face_detector.analyze_face_with_telemetry(frame, state=face_state)
            frame_scores.append(score)
        cap.release()
        face_score = float(np.median(frame_scores)) if frame_scores else 0.0
        
        # 2. Voice Analysis on the local audio file (Librosa doesn't support reading HTTP URLs directly)
        y, sr = librosa.load(temp_audio_path, sr=None)
        voice_score = voice_detector.analyze_audio(y, original_sr=sr)
        
        # 3. Fusion
        final_score, status = fuse_scores(face_score, voice_score)
        
        # 4. Save files to permanent storage (e.g. Supabase Storage bucket)
        video_ext = os.path.splitext(video_file.filename)[1] if video_file.filename else ".mp4"
        audio_ext = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
        video_filename = f"video_{student_id}{video_ext}"
        audio_filename = f"audio_{student_id}{audio_ext}"
        
        # Merge Video and Audio into a single file before uploading
        temp_merged_path = os.path.join("data/uploads/temp", f"merged_{student_id}{video_ext}")
        merge_success = merge_video_audio(temp_video_path, temp_audio_path, temp_merged_path)
        
        if merge_success and os.path.exists(temp_merged_path):
            video_url = storage.save(temp_merged_path, f"exams/{video_filename}")
        else:
            video_url = storage.save(temp_video_path, f"exams/{video_filename}")
            
        audio_url = storage.save(temp_audio_path, f"exams/{audio_filename}")
        
        # Clean up local temporary files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(temp_merged_path):
            os.remove(temp_merged_path)
        
        # 5. Save metadata & URLs to Database
        submit_exam_response(
            student_id=student_id,
            student_name=student_name,
            matric_number=matric_number,
            response=responses,
            video_path=video_url,
            audio_path=audio_url,
            score=final_score,
            status=status,
            warnings=warnings
        )
        
        return {
            "status": "success",
            "score": final_score,
            "verification_status": status,
            "message": "Exam submission and liveness verification completed successfully."
        }
    except Exception as e:
        log.error("Exam submission failed", student_id=student_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Submission processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
