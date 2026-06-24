from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./liveness_records.db"

    # Security
    JWT_SECRET: str = "super-secret-key-change-me"
    ADMIN_PASSWORD: str = "admin123"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    @property
    def cleaned_admin_password(self) -> str:
        return self.ADMIN_PASSWORD.strip().strip("'").strip('"') if self.ADMIN_PASSWORD else ""

    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_URL: str = "http://localhost:8000"

    # Storage
    STORAGE_TYPE: str = "local" # local, s3, or supabase
    LOCAL_STORAGE_PATH: str = "./data/uploads"
    S3_BUCKET: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"

    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_BUCKET: str = "exam-logs"

    # ML Model Settings
    FACE_MODEL_PATH: str = "face_landmarker.task"
    VOICE_TARGET_SR: int = 16000

    # WebRTC Settings
    RTC_ICE_SERVERS: Optional[str] = None

    @property
    def ice_servers(self) -> list:
        import json
        if self.RTC_ICE_SERVERS:
            try:
                cleaned_servers = self.RTC_ICE_SERVERS.strip()
                if (cleaned_servers.startswith("'") and cleaned_servers.endswith("'")) or \
                   (cleaned_servers.startswith('"') and cleaned_servers.endswith('"')):
                    cleaned_servers = cleaned_servers[1:-1]
                return json.loads(cleaned_servers)
            except Exception as e:
                pass
        
        # Robust default public STUN/TURN servers (includes Open Relay Project for NAT traversal)
        return [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {
                "urls": [
                    "turn:relay.metered.ca:80",
                    "turn:relay.metered.ca:443",
                    "turn:relay.metered.ca:443?transport=tcp"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["stun:relay.metered.ca:80"]
            }
        ]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
