import time
import logging
from src.database import SessionLocal, StudentSession, ExamSubmission, ExamConfig
import json

logger = logging.getLogger(__name__)

def update_student_telemetry(student_id, student_name, matric_number, data):
    db = SessionLocal()
    try:
        session = db.query(StudentSession).filter(StudentSession.session_id == student_id).first()
        if not session:
            session = StudentSession(session_id=student_id)
            db.add(session)
            
        session.last_updated = time.time()
        session.movement_status = data.get('movement_status', 'Unknown')
        session.multiple_faces = 1 if data.get('multiple_faces') else 0
        session.no_face = 1 if data.get('no_face') else 0
        session.warning = data.get('warning', '')
        session.student_name = student_name
        session.matric_number = matric_number
        
        db.commit()
    except Exception as e:
        logger.error(f"Error updating student telemetry: {e}")
        db.rollback()
    finally:
        db.close()

def get_all_students_telemetry():
    db = SessionLocal()
    result = {}
    try:
        sessions = db.query(StudentSession).all()
        for s in sessions:
            result[s.session_id] = {
                "student_name": s.student_name or 'Unknown',
                "matric_number": s.matric_number or 'Unknown',
                "last_updated": s.last_updated,
                "telemetry": {
                    "movement_status": s.movement_status,
                    "multiple_faces": bool(s.multiple_faces),
                    "no_face": bool(s.no_face),
                    "warning": s.warning
                }
            }
    except Exception as e:
        logger.error(f"Error fetching all students telemetry: {e}")
    finally:
        db.close()
    return result

def submit_exam_response(student_id, student_name, matric_number, response, video_path=None, audio_path=None, score=None, status=None, warnings=None):
    db = SessionLocal()
    try:
        # 1. Update/Create Exam Submission
        submission = db.query(ExamSubmission).filter(ExamSubmission.session_id == student_id).first()
        if not submission:
            submission = ExamSubmission(session_id=student_id)
            db.add(submission)
        submission.response = response
        submission.submitted_at = time.time()
        submission.video_path = video_path
        submission.audio_path = audio_path
        submission.score = score
        submission.status = status
        submission.warnings = warnings
        
        # 2. Update Student Session Status
        session = db.query(StudentSession).filter(StudentSession.session_id == student_id).first()
        if not session:
            session = StudentSession(session_id=student_id)
            db.add(session)
            
        session.last_updated = time.time()
        session.movement_status = "Exam Submitted"
        session.student_name = student_name
        session.matric_number = matric_number
        if warnings:
            session.warning = warnings
        
        db.commit()
    except Exception as e:
        logger.error(f"Error submitting exam response: {e}")
        db.rollback()
    finally:
        db.close()

def get_exam_submission(student_id):
    db = SessionLocal()
    try:
        sub = db.query(ExamSubmission).filter(ExamSubmission.session_id == student_id).first()
        if sub:
            return {
                "response": sub.response,
                "submitted_at": sub.submitted_at,
                "video_path": sub.video_path,
                "audio_path": sub.audio_path,
                "score": sub.score,
                "status": sub.status,
                "warnings": sub.warnings
            }
    except Exception as e:
        logger.error(f"Error fetching exam submission: {e}")
    finally:
        db.close()
    return None

def get_all_exam_submissions():
    db = SessionLocal()
    result = []
    try:
        submissions = db.query(ExamSubmission).all()
        for sub in submissions:
            student = db.query(StudentSession).filter(StudentSession.session_id == sub.session_id).first()
            result.append({
                "session_id": sub.session_id,
                "student_name": student.student_name if student else "Unknown",
                "matric_number": student.matric_number if student else "Unknown",
                "response": sub.response,
                "submitted_at": sub.submitted_at,
                "video_path": sub.video_path,
                "audio_path": sub.audio_path,
                "score": sub.score,
                "status": sub.status,
                "warnings": sub.warnings
            })
    except Exception as e:
        logger.error(f"Error fetching all exam submissions: {e}")
    finally:
        db.close()
    return result

def delete_student_record(student_id):
    db = SessionLocal()
    try:
        db.query(StudentSession).filter(StudentSession.session_id == student_id).delete()
        db.query(ExamSubmission).filter(ExamSubmission.session_id == student_id).delete()
        db.commit()
    except Exception as e:
        logger.error(f"Error deleting student record: {e}")
        db.rollback()
    finally:
        db.close()

def get_exam_questions():
    db = SessionLocal()
    try:
        config = db.query(ExamConfig).filter(ExamConfig.id == 1).first()
        if config:
            try:
                return json.loads(config.question)
            except (json.JSONDecodeError, TypeError):
                return [config.question]
        return ["No question set."]
    except Exception as e:
        logger.error(f"Error fetching exam questions: {e}")
    finally:
        db.close()
    return ["Error loading question from database."]

def set_exam_questions(questions: list):
    db = SessionLocal()
    try:
        config = db.query(ExamConfig).filter(ExamConfig.id == 1).first()
        if not config:
            config = ExamConfig(id=1)
            db.add(config)
        config.question = json.dumps(questions)
        config.updated_at = time.time()
        db.commit()
    except Exception as e:
        logger.error(f"Error setting exam questions: {e}")
        db.rollback()
    finally:
        db.close()
