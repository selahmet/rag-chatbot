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
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
        self.rag_chain = None
    
    def _initialize_components(self):
        """RAG bileşenlerini başlatır."""
        try:
            # API key'i environment variables'dan al
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not found")
            
            # Embedding model - fallback ile
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.embedding_model,
                    google_api_key=api_key
                )
                logger.info("Using Google Gemini embeddings")
            except Exception as e:
                logger.warning(f"Gemini embeddings failed: {e}")
                logger.info("Falling back to HuggingFace embeddings")
                # Fallback: Sentence Transformers
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    logger.info("Using HuggingFace embeddings as fallback")
                except ImportError:
                    logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                    raise e
            
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
    
    @retry_on_quota_error(max_retries=2, initial_delay=30)
    def create_vectorstore(self, documents: List[Document], persist_directory: str = "./chroma_db"):
        """
        Vector store oluşturur ve dokümanları indexler.
        
        Args:
            documents: Indexlenecek dokümanlar
            persist_directory: Vector store'un kaydedileceği dizin
        """
        try:
            # Batch processing - küçük gruplar halinde işle
            batch_size = 10  # API quota'sını korumak için küçük batch
            total_docs = len(documents)
            
            if total_docs <= batch_size:
                # Küçük dokümanlar, direkt işle
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=persist_directory
                )
            else:
                # Büyük dokümanları batch'ler halinde işle
                logger.info(f"Processing {total_docs} documents in batches of {batch_size}")
                
                # İlk batch ile vectorstore oluştur
                first_batch = documents[:batch_size]
                self.vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=persist_directory
                )
                
                # Kalan batch'leri ekle
                for i in range(batch_size, total_docs, batch_size):
                    batch = documents[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}")
                    
                    # Rate limiting - batch'ler arası bekleme
                    time.sleep(2)
                    
                    # Batch'i ekle
                    self.vectorstore.add_documents(batch)
            
            # Retriever oluştur
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.retrieval_k}
            )
            
            logger.info(f"Vector store created with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            # Quota hatası ise kullanıcıya daha net mesaj ver
            if "quota" in str(e).lower() or "429" in str(e):
                raise Exception(f"❌ API quota aşıldı. Lütfen biraz bekleyin veya Google AI Studio'da quota'nızı kontrol edin. Hata: {str(e)}")
            raise
    
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
    
    def setup_rag_chain(self):
        """RAG chain'ini kurar."""
        try:
            # System prompt
            system_prompt = (
                "Sen bir soru-cevap asistanısın. Aşağıdaki bağlamı kullanarak "
                "soruları yanıtla. Eğer cevabı bilmiyorsan, bilmediğini söyle. "
                "Yanıtını maksimum 3 cümle ile sınırla ve öz tut. "
                "Her zaman bağlamda verilen bilgilere dayalı yanıt ver.\n\n"
                "Bağlam: {context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Question-answer chain
            question_answer_chain = create_stuff_documents_chain(
                self.llm, prompt
            )
            
            # RAG chain
            self.rag_chain = create_retrieval_chain(
                self.retriever, question_answer_chain
            )
            
            logger.info("RAG chain setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Soru sorar ve RAG pipeline'ından yanıt alır.
        
        Args:
            question: Sorulacak soru
            
        Returns:
            Yanıt ve metadata içeren dictionary
        """
        try:
            if not self.rag_chain:
                raise ValueError("RAG chain not initialized. Call setup_rag_chain() first.")
            
            response = self.rag_chain.invoke({"input": question})
            
            # Source documents'ları temizle
            sources = []
            if "context" in response:
                for doc in response["context"]:
                    source_info = {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
            
            result = {
                "question": question,
                "answer": response.get("answer", "Yanıt bulunamadı."),
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