"""
Yardımcı Fonksiyonlar
RAG pipeline ve web arayüzü için genel yardımcı fonksiyonlar.
"""

import os
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Logging konfigürasyonu yapar.
    
    Args:
        log_level: Log seviyesi (DEBUG, INFO, WARNING, ERROR)
        log_file: Log dosyası (opsiyonel)
    """
    level = getattr(logging, log_level.upper())
    
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=level, format=format_string)


def check_api_key(api_key_name: str = "GEMINI_API_KEY") -> bool:
    """
    API key varlığını kontrol eder.
    
    Args:
        api_key_name: Kontrol edilecek API key ismi
        
    Returns:
        API key mevcut mu?
    """
    api_key = os.getenv(api_key_name)
    if not api_key or api_key.strip() == "":
        logger.error(f"{api_key_name} not found in environment variables")
        return False
    
    if api_key == f"your_{api_key_name.lower()}_here":
        logger.error(f"{api_key_name} is not configured properly")
        return False
    
    return True


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Konfigürasyon ayarlarını validate eder.
    
    Args:
        config: Konfigürasyon dictionary
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = ["chunk_size", "chunk_overlap", "retrieval_k"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Value validations
    if "chunk_size" in config:
        if not isinstance(config["chunk_size"], int) or config["chunk_size"] <= 0:
            errors.append("chunk_size must be a positive integer")
    
    if "chunk_overlap" in config:
        if not isinstance(config["chunk_overlap"], int) or config["chunk_overlap"] < 0:
            errors.append("chunk_overlap must be a non-negative integer")
        
        if "chunk_size" in config and config["chunk_overlap"] >= config["chunk_size"]:
            errors.append("chunk_overlap must be less than chunk_size")
    
    if "retrieval_k" in config:
        if not isinstance(config["retrieval_k"], int) or config["retrieval_k"] <= 0:
            errors.append("retrieval_k must be a positive integer")
    
    if "temperature" in config:
        if not isinstance(config["temperature"], (int, float)) or config["temperature"] < 0 or config["temperature"] > 2:
            errors.append("temperature must be between 0 and 2")
    
    return len(errors) == 0, errors


def create_session_id() -> str:
    """
    Unique session ID oluşturur.
    
    Returns:
        Session ID
    """
    timestamp = str(time.time())
    hash_object = hashlib.md5(timestamp.encode())
    return hash_object.hexdigest()[:8]


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Chat geçmişini formatlar.
    
    Args:
        messages: Chat mesajları listesi
        
    Returns:
        Formatlanmış chat geçmişi
    """
    formatted = []
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        if timestamp:
            time_str = datetime.fromtimestamp(float(timestamp)).strftime("%H:%M")
            formatted.append(f"[{time_str}] {role.title()}: {content}")
        else:
            formatted.append(f"{role.title()}: {content}")
    
    return "\n".join(formatted)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Metni belirtilen uzunlukta keser.
    
    Args:
        text: Kesilecek metin
        max_length: Maximum uzunluk
        suffix: Kesme sonrası eklenecek suffix
        
    Returns:
        Kesilmiş metin
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """
    Dosya adını güvenli hale getirir.
    
    Args:
        filename: Orijinal dosya adı
        
    Returns:
        Sanitized dosya adı
    """
    # Tehlikeli karakterleri kaldır
    unsafe_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\']
    safe_name = filename
    
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Boşlukları underscore ile değiştir
    safe_name = safe_name.replace(' ', '_')
    
    # Çoklu underscoreları tek underscore yap
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    
    # Başta ve sonda underscore varsa kaldır
    safe_name = safe_name.strip('_')
    
    return safe_name or "unnamed_file"


def measure_performance(func):
    """
    Fonksiyon performansını ölçen decorator.
    
    Args:
        func: Ölçülecek fonksiyon
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


def save_chat_session(session_data: Dict[str, Any], session_dir: str = "./data/sessions"):
    """
    Chat session'ını kaydeder.
    
    Args:
        session_data: Session verisi
        session_dir: Session dizini
    """
    try:
        os.makedirs(session_dir, exist_ok=True)
        
        session_id = session_data.get("session_id", create_session_id())
        filename = f"session_{session_id}_{int(time.time())}.json"
        filepath = os.path.join(session_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Chat session saved: {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving chat session: {str(e)}")


def load_chat_session(session_file: str) -> Optional[Dict[str, Any]]:
    """
    Chat session'ını yükler.
    
    Args:
        session_file: Session dosya yolu
        
    Returns:
        Session verisi veya None
    """
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Error loading chat session: {str(e)}")
        return None


def get_system_info() -> Dict[str, Any]:
    """
    Sistem bilgilerini döndürür.
    
    Returns:
        Sistem bilgileri
    """
    import platform
    import psutil
    
    try:
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('.').total
        }
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}


class Timer:
    """Basit timer sınıfı."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Timer başlat."""
        self.start_time = time.time()
    
    def stop(self):
        """Timer durdur."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        """Geçen süreyi döndür."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def format_error_message(error: Exception) -> str:
    """
    Hata mesajlarını kullanıcı dostu formatta düzenler.
    
    Args:
        error: Exception nesnesi
        
    Returns:
        Formatlanmış hata mesajı
    """
    error_msg = str(error).lower()
    
    # Quota/Rate limit hataları
    if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
        if "huggingface" in error_msg or "fallback" in error_msg:
            return """
            � **Fallback Embeddings Kullanıldı**
            
            Gemini API quota'sı aşıldığı için HuggingFace embeddings kullanılıyor.
            
            **Durum:** Sistem çalışıyor, performans biraz daha yavaş olabilir.
            """
        else:
            return """
            �🚫 **API Quota Aşıldı - Fallback Aktif**
            
            Google Gemini API quota'sı aşıldı, alternatif embedding sistemi devreye alınıyor.
            
            **Sistem durumu:**
            - ✅ Sistem çalışmaya devam ediyor
            - 🔄 HuggingFace embeddings kullanılıyor  
            - ⚡ Biraz daha yavaş olabilir
            
            **Çözüm önerileri:**
            - Sistem şu an çalışır durumda, devam edebilirsiniz
            - [Google AI Studio](https://aistudio.google.com/app/apikey)'da quota durumunuzu kontrol edin
            - Ücretli plana geçerek daha hızlı performance alın
            """
    
    # API key hataları
    elif "api key" in error_msg or "authentication" in error_msg:
        return """
        🔑 **API Key Sorunu**
        
        Google Gemini API key'iniz geçersiz veya eksik.
        
        **Çözüm:**
        - .env dosyasındaki GEMINI_API_KEY'i kontrol edin
        - [Google AI Studio](https://aistudio.google.com/app/apikey)'dan yeni key alın
        """
    
    # Dosya hataları
    elif "file" in error_msg or "pdf" in error_msg:
        return """
        📄 **Dosya İşleme Sorunu**
        
        PDF dosyası işlenirken hata oluştu.
        
        **Çözüm:**
        - Dosyanın bozuk olmadığını kontrol edin
        - Daha küçük PDF dosyası deneyin
        - Dosya isminde özel karakter olmamasına dikkat edin
        """
    
    # Genel hata
    else:
        return f"""
        ❌ **Beklenmeyen Hata**
        
        {str(error)}
        
        Bu sorunu geliştirici ekibe bildirin.
        """