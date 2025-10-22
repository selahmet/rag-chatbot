"""
RAG Pipeline Test Modülü
Unit test'ler ve integration test'ler için test sınıfları.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Proje modüllerini import et
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline import RAGPipeline, create_rag_pipeline
from src.document_processor import DocumentProcessor, validate_file, format_file_size
from src.utils import (
    setup_logging, check_api_key, create_session_id, 
    truncate_text, sanitize_filename, Timer
)


class TestUtils(unittest.TestCase):
    """Utility fonksiyonları için test sınıfı."""
    
    def test_truncate_text(self):
        """Metin kısaltma fonksiyonu testi."""
        # Normal metin
        text = "Bu bir test metnidir. Çok uzun bir metin."
        result = truncate_text(text, 20)
        self.assertEqual(result, "Bu bir test metnidir")
        
        # Kısa metin
        short_text = "Kısa"
        result = truncate_text(short_text, 20)
        self.assertEqual(result, short_text)
        
        # Boş metin
        result = truncate_text("", 20)
        self.assertEqual(result, "")
    
    def test_sanitize_filename(self):
        """Dosya adı temizleme fonksiyonu testi."""
        # Özel karakterler
        filename = "test<file>name|with:special*chars?.pdf"
        result = sanitize_filename(filename)
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertNotIn("|", result)
        self.assertTrue(result.endswith(".pdf"))
        
        # Normal dosya adı
        normal_file = "normal_file.pdf"
        result = sanitize_filename(normal_file)
        self.assertEqual(result, normal_file)
    
    def test_create_session_id(self):
        """Session ID oluşturma testi."""
        id1 = create_session_id()
        id2 = create_session_id()
        
        # Her session ID farklı olmalı
        self.assertNotEqual(id1, id2)
        
        # Uzunluk kontrolü
        self.assertEqual(len(id1), 8)
        self.assertEqual(len(id2), 8)
    
    def test_timer(self):
        """Timer sınıfı testi."""
        import time
        
        timer = Timer()
        timer.start()
        time.sleep(0.1)  # 100ms bekle
        timer.stop()
        
        elapsed = timer.elapsed()
        self.assertGreater(elapsed, 0.05)  # En az 50ms
        self.assertLess(elapsed, 0.5)     # En fazla 500ms
    
    def test_format_file_size(self):
        """Dosya boyutu formatlama testi."""
        # Bytes
        self.assertEqual(format_file_size(500), "500 B")
        
        # KB
        self.assertEqual(format_file_size(1024), "1.00 KB")
        self.assertEqual(format_file_size(1536), "1.50 KB")
        
        # MB
        self.assertEqual(format_file_size(1024 * 1024), "1.00 MB")
        self.assertEqual(format_file_size(1024 * 1024 * 2.5), "2.50 MB")


class TestDocumentProcessor(unittest.TestCase):
    """Document processor için test sınıfı."""
    
    def setUp(self):
        """Test setup."""
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir)
    
    def test_validate_file(self):
        """Dosya validasyon testi."""
        # Geçerli PDF dosyası
        valid, error = validate_file("test.pdf", max_size_mb=10)
        self.assertTrue(valid)
        self.assertIsNone(error)
        
        # Geçersiz dosya türü
        valid, error = validate_file("test.txt", max_size_mb=10)
        self.assertFalse(valid)
        self.assertIn("PDF", error)
        
        # Uzantısız dosya
        valid, error = validate_file("test", max_size_mb=10)
        self.assertFalse(valid)
        self.assertIn("PDF", error)
    
    def test_extract_metadata(self):
        """Metadata extraction testi."""
        # Mock file object
        mock_file = MagicMock()
        mock_file.name = "test_document.pdf"
        mock_file.size = 1024 * 1024  # 1 MB
        
        metadata = self.processor.extract_metadata(mock_file)
        
        self.assertEqual(metadata["filename"], "test_document.pdf")
        self.assertEqual(metadata["size"], 1024 * 1024)
        self.assertIn("upload_time", metadata)
        self.assertEqual(metadata["type"], "application/pdf")


class TestRAGPipelineUnit(unittest.TestCase):
    """RAG Pipeline unit testleri."""
    
    def setUp(self):
        """Test setup."""
        self.config = {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "temperature": 0.1,
            "retrieval_k": 2
        }
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    def test_pipeline_creation(self):
        """Pipeline oluşturma testi."""
        # Mock bağımlılıkları
        with patch('src.rag_pipeline.GoogleGenerativeAIEmbeddings'), \
             patch('src.rag_pipeline.ChatGoogleGenerativeAI'):
            
            pipeline = RAGPipeline(self.config)
            
            self.assertEqual(pipeline.chunk_size, 500)
            self.assertEqual(pipeline.chunk_overlap, 50)
            self.assertEqual(pipeline.temperature, 0.1)
            self.assertEqual(pipeline.retrieval_k, 2)
    
    def test_config_validation(self):
        """Konfigürasyon validasyon testi."""
        # Geçersiz chunk_size
        invalid_config = self.config.copy()
        invalid_config["chunk_size"] = -100
        
        with self.assertRaises(ValueError):
            RAGPipeline(invalid_config)
        
        # Geçersiz temperature
        invalid_config = self.config.copy()
        invalid_config["temperature"] = 2.0
        
        with self.assertRaises(ValueError):
            RAGPipeline(invalid_config)


class TestRAGPipelineIntegration(unittest.TestCase):
    """RAG Pipeline integration testleri."""
    
    def setUp(self):
        """Test setup."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf_path = os.path.join(self.temp_dir, "test.pdf")
        
        # Basit test PDF oluştur (mock)
        with open(self.test_pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\nTest content for RAG pipeline testing.")
    
    def tearDown(self):
        """Test cleanup."""
        shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch('src.rag_pipeline.PyPDFLoader')
    @patch('src.rag_pipeline.GoogleGenerativeAIEmbeddings')
    @patch('src.rag_pipeline.ChatGoogleGenerativeAI')
    @patch('src.rag_pipeline.Chroma')
    def test_full_pipeline_workflow(self, mock_chroma, mock_llm, mock_embeddings, mock_loader):
        """Tam pipeline workflow testi."""
        # Mock document
        mock_doc = MagicMock()
        mock_doc.page_content = "Bu bir test dokümandır. RAG pipeline testi için kullanılır."
        mock_doc.metadata = {"source": self.test_pdf_path}
        
        mock_loader.return_value.load.return_value = [mock_doc]
        
        # Mock vectorstore
        mock_vectorstore = MagicMock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Mock retriever
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        # Pipeline oluştur ve test et
        config = {
            "chunk_size": 100,
            "chunk_overlap": 20,
            "temperature": 0.1,
            "retrieval_k": 2
        }
        
        pipeline = RAGPipeline(config)
        
        # Dokümanları yükle
        documents = pipeline.load_documents([self.test_pdf_path])
        self.assertEqual(len(documents), 1)
        
        # Dokümanları işle
        chunks = pipeline.process_documents(documents)
        self.assertGreater(len(chunks), 0)
        
        # Vector store oluştur
        pipeline.create_vectorstore(chunks)
        self.assertIsNotNone(pipeline.vectorstore)
        
    # Manual RAG'de ayrıca bir chain gerekmiyor; vectorstore hazır olduğunda soru sorulabilir


class TestErrorHandling(unittest.TestCase):
    """Hata yönetimi testleri."""
    
    def test_missing_api_key(self):
        """Eksik API key testi."""
        # API key'i geçici olarak kaldır
        original_key = os.environ.get("GEMINI_API_KEY")
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        
        try:
            self.assertFalse(check_api_key())
        finally:
            # API key'i geri yükle
            if original_key:
                os.environ["GEMINI_API_KEY"] = original_key
    
    def test_invalid_file_path(self):
        """Geçersiz dosya yolu testi."""
        config = {"chunk_size": 1000, "chunk_overlap": 200}
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            with patch('src.rag_pipeline.GoogleGenerativeAIEmbeddings'), \
                 patch('src.rag_pipeline.ChatGoogleGenerativeAI'):
                
                pipeline = RAGPipeline(config)
                
                # Olmayan dosya yolu
                with self.assertRaises(Exception):
                    pipeline.load_documents(["/nonexistent/file.pdf"])


class TestPerformance(unittest.TestCase):
    """Performans testleri."""
    
    def test_text_processing_performance(self):
        """Metin işleme performans testi."""
        # Büyük metin oluştur
        large_text = "Bu bir test cümlesidir. " * 1000  # ~25KB
        
        # Performansı ölç
        timer = Timer()
        timer.start()
        
        result = truncate_text(large_text, 1000)
        
        timer.stop()
        
        # Performans kriterleri
        self.assertLess(timer.elapsed(), 0.1)  # 100ms'den az
        self.assertLessEqual(len(result), 1000)


def create_test_suite():
    """Test suite oluştur."""
    suite = unittest.TestSuite()
    
    # Test sınıflarını ekle
    test_classes = [
        TestUtils,
        TestDocumentProcessor, 
        TestRAGPipelineUnit,
        TestRAGPipelineIntegration,
        TestErrorHandling,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == "__main__":
    # Logging setup
    setup_logging("INFO")
    
    # Test suite'i çalıştır
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Sonuçları göster
    print(f"\n{'='*60}")
    print(f"Test Sonuçları:")
    print(f"Toplam: {result.testsRun}")
    print(f"Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Başarısız: {len(result.failures)}")
    print(f"Hata: {len(result.errors)}")
    
    if result.failures:
        print(f"\nBaşarısız Testler:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nHatalı Testler:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('Exception:')[-1].strip()}")