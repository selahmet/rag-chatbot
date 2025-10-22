"""
RAG Pipeline Implementation
Bu modül, RAG (Retrieval Augmented Generation) pipeline'ının temel bileşenlerini içerir.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv

# Environment variables yükle
load_dotenv()

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_on_quota_error(max_retries=3, initial_delay=60):
    """
    Quota hatalarında retry yapar.
    
    Args:
        max_retries: Maksimum retry sayısı
        initial_delay: İlk bekleme süresi (saniye)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                        if attempt < max_retries - 1:
                            delay = initial_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Quota exceeded, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"Max retries exceeded for quota error: {e}")
                            raise Exception(f"API quota exceeded. Please wait or upgrade your plan. Original error: {e}")
                    else:
                        # Non-quota error, re-raise immediately
                        raise e
            return None
        return wrapper
    return decorator


class RAGPipeline:
    """
    RAG Pipeline implementasyonu.
    Doküman yükleme, chunking, embedding, retrieval ve generation işlemlerini yönetir.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "models/embedding-001",
        llm_model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 500,
        retrieval_k: int = 4
    ):
        """
        RAG Pipeline'ı başlatır.
        
        Args:
            chunk_size: Metin parçalarının boyutu
            chunk_overlap: Parçalar arası örtüşme
            embedding_model: Embedding modeli
            llm_model: Language model
            temperature: LLM temperature ayarı
            max_tokens: Maximum token sayısı
            retrieval_k: Geri getirilen doküman sayısı
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_k = retrieval_k
        
        # API key kontrolü
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Bileşenleri başlat
        self._initialize_components()
        
        # Vector store
        self.vectorstore = None
        self.retriever = None
    
    def _initialize_components(self):
        """RAG bileşenlerini başlatır."""
        try:
            # API key'i environment variables'dan al
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not found")
            
            # Embedding model - sadece Gemini kullan, fallback create_vectorstore'da
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                google_api_key=api_key
            )
            logger.info("Using Google Gemini embeddings")
            
            # LLM
            self.llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                google_api_key=api_key
            )
            
            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        PDF dokümanlarını yükler.
        
        Args:
            file_paths: PDF dosya yollarının listesi
            
        Returns:
            Yüklenen dokümanların listesi
        """
        documents = []
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} pages from {file_path}")
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Dokümanları parçalara böler (chunking).
        
        Args:
            documents: Kaynak dokümanlar
            
        Returns:
            Parçalara bölünmüş dokümanlar
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def create_vectorstore(self, documents: List[Document], persist_directory: str = "./chroma_db"):
        """
        Vector store oluşturur ve dokümanları indexler.
        
        Args:
            documents: Indexlenecek dokümanlar
            persist_directory: Vector store'un kaydedileceği dizin
        """
        # Önce Gemini ile dene, quota hatası varsa fallback'e geç
        try:
            return self._create_vectorstore_with_retry(documents, persist_directory, use_fallback=False)
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "429" in error_msg or "rate limit" in error_msg:
                logger.warning("Gemini quota exceeded, switching to HuggingFace embeddings fallback")
                try:
                    return self._create_vectorstore_with_retry(documents, persist_directory, use_fallback=True)
                except Exception as fallback_error:
                    logger.error(f"Fallback embedding also failed: {fallback_error}")
                    raise Exception(f"❌ Hem Gemini hem de fallback embedding başarısız oldu. Lütfen daha sonra tekrar deneyin.")
            else:
                raise e
    
    def _create_vectorstore_with_retry(self, documents: List[Document], persist_directory: str, use_fallback: bool = False):
        """
        Vector store oluşturur - retry mekanizması ile.
        """
        # Fallback embedding kullanılacaksa değiştir
        original_embeddings = None
        if use_fallback:
            try:
                # Yeni LangChain HuggingFace import'unu dene
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    # Eski import'u dene
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                
                original_embeddings = self.embeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"  # sentence-transformers/ prefix'i kaldırıldı
                )
                logger.info("Switched to HuggingFace embeddings (local)")
                
            except ImportError as ie:
                logger.error(f"HuggingFace embeddings import failed: {ie}")
                raise Exception("HuggingFace embeddings yüklenemedi. Lütfen paket güncellemelerini kontrol edin.")
            except Exception as ee:
                logger.error(f"HuggingFace embeddings initialization failed: {ee}")
                raise Exception("HuggingFace embeddings başlatılamadı. Model indiriliyor olabilir, lütfen tekrar deneyin.")
        
        try:
            # Basit ve direkt yaklaşım - small files için optimized
            total_docs = len(documents)
            logger.info(f"Creating vector store with {total_docs} document chunks")
            
            # Rate limiting için küçük bekleme
            if not use_fallback:
                time.sleep(1)  # Gemini için 1 saniye
            
            # Dokümanları direkt işle
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            
            # Retriever oluştur
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_k}
            )
            
            embedding_type = "HuggingFace" if use_fallback else "Gemini"
            logger.info(f"Vector store created with {len(documents)} documents using {embedding_type} embeddings")
            
        except Exception as e:
            # Fallback durumunda original embedding'i geri yükle
            if use_fallback and original_embeddings:
                self.embeddings = original_embeddings
            raise e
    
    def load_vectorstore(self, persist_directory: str = "./chroma_db"):
        """
        Mevcut vector store'u yükler.
        
        Args:
            persist_directory: Vector store'un bulunduğu dizin
        """
        try:
            if not os.path.exists(persist_directory):
                raise FileNotFoundError(f"Vector store directory not found: {persist_directory}")
            
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_k}
            )
            
            logger.info("Vector store loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    

    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Soru sorar ve manual RAG pipeline'ından yanıt alır.
        
        Args:
            question: Sorulacak soru
            
        Returns:
            Yanıt ve metadata içeren dictionary
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized. Add documents first.")
            
            # 1. Retrieval: Benzer dokümanları bul
            docs = self.vectorstore.similarity_search(question, k=3)
            
            # 2. Context oluştur
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 3. Prompt oluştur
            prompt = f"""Verilen bağlam bilgisini kullanarak soruyu yanıtla. Eğer bağlam soruyu yanıtlamak için yeterli değilse, 'Verilen bağlamda bu soruyu yanıtlamak için yeterli bilgi yok' de.

Bağlam:
{context}

Soru: {question}

Yanıt:"""

            # 4. LLM'e sor (string prompt ile)
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Source documents'ları temizle
            sources = []
            for doc in docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            result = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "source_count": len(sources)
            }
            
            logger.info(f"Query processed: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": f"Hata oluştu: {str(e)}",
                "sources": [],
                "source_count": 0
            }
    
    def get_similarity_search(self, question: str, k: Optional[int] = None) -> List[Document]:
        """
        Similarity search yapar (debugging için).
        
        Args:
            question: Arama sorgusu
            k: Döndürülecek doküman sayısı
            
        Returns:
            En benzer dokümanlar
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vector store not initialized")
            
            k = k or self.retrieval_k
            docs = self.vectorstore.similarity_search(question, k=k)
            
            logger.info(f"Found {len(docs)} similar documents")
            return docs
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []


def create_rag_pipeline(config: Optional[Dict[str, Any]] = None) -> RAGPipeline:
    """
    RAG Pipeline factory function.
    
    Args:
        config: Konfigürasyon parametreleri
        
    Returns:
        Yapılandırılmış RAG Pipeline
    """
    if config is None:
        config = {}
    
    # Default values from environment variables
    defaults = {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200)),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "models/embedding-001"),
        "llm_model": os.getenv("LLM_MODEL", "gemini-2.0-flash"),
        "temperature": float(os.getenv("TEMPERATURE", 0.3)),
        "max_tokens": int(os.getenv("MAX_TOKENS", 500)),
        "retrieval_k": int(os.getenv("RETRIEVAL_K", 4))
    }
    
    # Merge with provided config
    final_config = {**defaults, **config}
    
    return RAGPipeline(**final_config)