import os
import shutil
from abc import ABC, abstractmethod
from src.config import settings

class StorageProvider(ABC):
    @abstractmethod
    def save(self, file_path: str, destination: str) -> str:
        pass

    @abstractmethod
    def delete(self, file_id: str):
        pass

class LocalStorageProvider(StorageProvider):
    def __init__(self, base_path: str = settings.LOCAL_STORAGE_PATH):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def save(self, file_path: str, destination: str) -> str:
        full_dest = os.path.join(self.base_path, destination)
        os.makedirs(os.path.dirname(full_dest), exist_ok=True)
        shutil.copyfile(file_path, full_dest)
        return full_dest

    def delete(self, file_id: str):
        full_path = os.path.join(self.base_path, file_id)
        if os.path.exists(full_path):
            os.remove(full_path)

class S3StorageProvider(StorageProvider):
    def __init__(self):
        # boto3 should be added to requirements if using S3
        import boto3
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket = settings.S3_BUCKET

    def save(self, file_path: str, destination: str) -> str:
        self.s3.upload_file(file_path, self.bucket, destination)
        return f"s3://{self.bucket}/{destination}"

    def delete(self, file_id: str):
        self.s3.delete_object(Bucket=self.bucket, Key=file_id)

import requests

class SupabaseStorageProvider(StorageProvider):
    def __init__(self):
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_KEY
        self.bucket = settings.SUPABASE_BUCKET

    def save(self, file_path: str, destination: str) -> str:
        # Upload via Supabase Storage REST API
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/octet-stream"
        }
        # Remove leading slashes if any
        dest_path = destination.lstrip("/")
        upload_url = f"{self.url}/storage/v1/object/{self.bucket}/{dest_path}"
        
        with open(file_path, "rb") as f:
            response = requests.post(upload_url, headers=headers, data=f)
            
            if response.status_code != 200:
                # If it already exists, Supabase might return 400. In case of overlap, we can try PUT
                if "Duplicate" in response.text or response.status_code == 400:
                    # Try PUT request to overwrite
                    put_url = upload_url
                    f.seek(0)
                    response = requests.put(put_url, headers=headers, data=f)
                    
                if response.status_code != 200:
                    raise Exception(f"Supabase Storage Upload failed: {response.text}")
                
        # Return the public URL for playback
        return f"{self.url}/storage/v1/object/public/{self.bucket}/{dest_path}"

    def delete(self, file_id: str):
        headers = {
            "Authorization": f"Bearer {self.key}"
        }
        delete_url = f"{self.url}/storage/v1/object/{self.bucket}/{file_id}"
        requests.delete(delete_url, headers=headers)

def get_storage() -> StorageProvider:
    if settings.STORAGE_TYPE == "s3":
        return S3StorageProvider()
    elif settings.STORAGE_TYPE == "supabase":
        return SupabaseStorageProvider()
    return LocalStorageProvider()

storage = get_storage()
