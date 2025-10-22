"""
Doküman İşleme Modülü
PDF, TXT ve diğer doküman formatlarını işleyen yardımcı fonksiyonlar.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Doküman işleme sınıfı."""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.md']
    
    def validate_file(self, file_path: str) -> bool:
        """
        Dosyanın geçerli olup olmadığını kontrol eder.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            Dosya geçerli mi?
        """
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_ext}")
            return False
        
        # Dosya boyutu kontrolü (max 50MB)
        file_size = os.path.getsize(file_path)
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            logger.error(f"File too large: {file_size} bytes (max: {max_size})")
            return False
        
        return True
    
    def save_uploaded_file(self, uploaded_file, upload_dir: str = "./data/uploads") -> Optional[str]:
        """
        Streamlit uploaded file'ı kaydet.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            upload_dir: Yükleme dizini
            
        Returns:
            Kaydedilen dosyanın yolu
        """
        try:
            # Upload dizinini oluştur
            os.makedirs(upload_dir, exist_ok=True)
            
            # Dosya yolu
            file_path = os.path.join(upload_dir, uploaded_file.name)
            
            # Dosyayı kaydet
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"File saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            return None
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Dosyadan metadata çıkarır.
        
        Args:
            file_path: Dosya yolu
            
        Returns:
            Metadata dictionary
        """
        try:
            path_obj = Path(file_path)
            stat = path_obj.stat()
            
            metadata = {
                "filename": path_obj.name,
                "file_size": stat.st_size,
                "file_extension": path_obj.suffix,
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime,
                "file_path": str(path_obj.absolute())
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}
    
    def cleanup_temp_files(self, temp_dir: str):
        """
        Geçici dosyaları temizler.
        
        Args:
            temp_dir: Temizlenecek dizin
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up {temp_dir}: {str(e)}")


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Dosya bilgilerini döndürür.
    
    Args:
        file_path: Dosya yolu
        
    Returns:
        Dosya bilgileri
    """
    processor = DocumentProcessor()
    
    if not processor.validate_file(file_path):
        return {"error": "Invalid file"}
    
    metadata = processor.extract_metadata(file_path)
    metadata["is_valid"] = True
    
    return metadata


def format_file_size(size_bytes: int) -> str:
    """
    Dosya boyutunu okunabilir formata çevirir.
    
    Args:
        size_bytes: Byte cinsinden boyut
        
    Returns:
        Formatlanmış boyut stringi
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"