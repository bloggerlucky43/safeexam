from sqlalchemy import create_engine, Column, String, Float, Integer, Text, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import time
from src.config import settings

# Engine setup
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class StudentSession(Base):
    __tablename__ = "student_sessions"
    session_id = Column(String, primary_key=True, index=True)
    last_updated = Column(Float)
    movement_status = Column(String)
    multiple_faces = Column(Integer)
    no_face = Column(Integer)
    warning = Column(Text)
    student_name = Column(String, default="Unknown")
    matric_number = Column(String, default="Unknown")

class ExamSubmission(Base):
    __tablename__ = "exam_submissions"
    session_id = Column(String, primary_key=True)
    response = Column(Text)
    submitted_at = Column(Float)
    video_path = Column(String, nullable=True)
    audio_path = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    status = Column(String, nullable=True)
    warnings = Column(Text, nullable=True)

class ExamConfig(Base):
    __tablename__ = "exam_config"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text)
    updated_at = Column(Float)

def init_db():
    # Only perform schema recreation check for SQLite
    if "sqlite" in settings.DATABASE_URL:
        need_recreate = False
        db = SessionLocal()
        try:
            # Try to query the new column to see if it exists
            db.query(ExamSubmission).filter(ExamSubmission.video_path == None).first()
        except Exception:
            need_recreate = True
        finally:
            db.close()

        if need_recreate:
            try:
                Base.metadata.drop_all(bind=engine)
            except Exception:
                pass
            
    Base.metadata.create_all(bind=engine)
    
    # Initialize with default question if empty
    db = SessionLocal()
    if not db.query(ExamConfig).filter(ExamConfig.id == 1).first():
        default_q = '["Question 1: Explain the geopolitical implications of Active Gravity Control."]'
        config = ExamConfig(id=1, question=default_q, updated_at=time.time())
        db.add(config)
        db.commit()
    db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
