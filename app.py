"""
RAG Chatbot Streamlit Uygulaması
Retrieval Augmented Generation tabanlı chatbot web arayüzü.
"""

import streamlit as st
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

# Proje dizinini sys.path'e ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Local imports
from src.utils import (
    setup_logging, check_api_key, create_session_id, 
    truncate_text, sanitize_filename, Timer
)
from src.document_processor import DocumentProcessor, format_file_size

# Page config
st.set_page_config(
    page_title="RAG Chatbot Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging setup
setup_logging("INFO")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = create_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def check_dependencies():
    """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder."""
    try:
        import langchain
        import chromadb
        import google.generativeai
        return True, ""
    except ImportError as e:
        return False, f"Eksik kütüphane: {str(e)}"


def display_api_key_input():
    """API key giriş arayüzü."""
    st.header("🔑 API Konfigürasyonu")
    
    # Check environment variable first
    env_api_key = os.getenv("GEMINI_API_KEY")
    if env_api_key and env_api_key != "your_gemini_api_key_here":
        st.success("✅ Gemini API Key environment variable'dan yüklendi.")
        return True
    
    # Input field for API key
    api_key = st.text_input(
        "Gemini API Key girin:",
        type="password",
        help="Google AI Studio'dan API key alabilirsiniz: https://ai.google.dev/gemini-api/docs/api-key"
    )
    
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        st.success("✅ API Key ayarlandı.")
        return True
    else:
        st.warning("⚠️ Devam etmek için API key gerekli.")
        return False


def initialize_rag_pipeline():
    """RAG pipeline'ı başlatır."""
    if st.session_state.rag_pipeline is None:
        try:
            with st.spinner("RAG sistemi başlatılıyor..."):
                # Import here to avoid dependency issues
                from src.rag_pipeline import create_rag_pipeline
                
                config = {
                    "chunk_size": st.session_state.get("chunk_size", 1000),
                    "chunk_overlap": st.session_state.get("chunk_overlap", 200),
                    "temperature": st.session_state.get("temperature", 0.3),
                    "retrieval_k": st.session_state.get("retrieval_k", 4),
                }
                
                st.session_state.rag_pipeline = create_rag_pipeline(config)
                st.success("✅ RAG sistemi hazır.")
                
        except Exception as e:
            st.error(f"❌ RAG sistemi başlatılamadı: {str(e)}")
            st.stop()


def display_file_upload():
    """Dosya yükleme arayüzü."""
    st.header("📁 Doküman Yükleme")
    
    uploaded_files = st.file_uploader(
        "PDF dosyalarını seçin:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Birden fazla PDF dosyası seçebilirsiniz."
    )
    
    if uploaded_files:
        processor = DocumentProcessor()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Yüklenen Dosyalar:")
            
            for idx, file in enumerate(uploaded_files):
                with st.expander(f"📄 {file.name}"):
                    file_size = len(file.getbuffer())
                    st.write(f"**Boyut:** {format_file_size(file_size)}")
                    st.write(f"**Tip:** {file.type}")
        
        with col2:
            if st.button("🔄 Dosyaları İşle", type="primary"):
                process_uploaded_files(uploaded_files)


def process_uploaded_files(uploaded_files):
    """Yüklenen dosyaları işler."""
    try:
        with st.spinner("Dokümanlar işleniyor..."):
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            # Save uploaded files
            processor = DocumentProcessor()
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, sanitize_filename(file.name))
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                file_paths.append(temp_path)
            
            # Initialize pipeline if not exists
            if st.session_state.rag_pipeline is None:
                initialize_rag_pipeline()
            
            pipeline = st.session_state.rag_pipeline
            
            # Process documents
            with Timer() as timer:
                # Load documents
                documents = pipeline.load_documents(file_paths)
                
                if not documents:
                    st.error("❌ Dokümanlar yüklenemedi.")
                    return
                
                # Process (chunk) documents
                chunks = pipeline.process_documents(documents)
                
                # Create vector store
                pipeline.create_vectorstore(chunks)
                
                # Setup RAG chain
                pipeline.setup_rag_chain()
            
            # Cleanup temporary files
            shutil.rmtree(temp_dir)
            
            # Update session state
            st.session_state.vectorstore_ready = True
            st.session_state.uploaded_files = [file.name for file in uploaded_files]
            
            st.success(f"✅ {len(documents)} sayfa, {len(chunks)} parça işlendi. ({timer.elapsed():.2f}s)")
            
    except Exception as e:
        st.error(f"❌ Doküman işleme hatası: {str(e)}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def display_chat_interface():
    """Chat arayüzü."""
    st.header("💬 Chatbot")
    
    if not st.session_state.vectorstore_ready:
        st.info("ℹ️ Önce dokümanları yükleyip işleyin.")
        return
    
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Kaynaklar"):
                        for idx, source in enumerate(message["sources"]):
                            st.write(f"**Kaynak {idx + 1}:**")
                            st.write(f"• {truncate_text(source['content'], 200)}")
                            if "metadata" in source and "source" in source["metadata"]:
                                st.write(f"• Dosya: {source['metadata']['source']}")
    
    # Chat input
    if prompt := st.chat_input("Sorunuzu yazın..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Yanıt hazırlanıyor..."):
                try:
                    response = st.session_state.rag_pipeline.query(prompt)
                    
                    answer = response["answer"]
                    sources = response.get("sources", [])
                    
                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 Kaynaklar"):
                            for idx, source in enumerate(sources):
                                st.write(f"**Kaynak {idx + 1}:**")
                                st.write(f"• {truncate_text(source['content'], 200)}")
                                if "metadata" in source and "source" in source["metadata"]:
                                    st.write(f"• Dosya: {Path(source['metadata']['source']).name}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Hata: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def display_sidebar():
    """Sidebar arayüzü."""
    with st.sidebar:
        st.title("⚙️ Ayarlar")
        
        # System info
        with st.expander("📊 Sistem Bilgisi"):
            st.write(f"**Session ID:** {st.session_state.session_id}")
            st.write(f"**Yüklenen Dosyalar:** {len(st.session_state.uploaded_files)}")
            st.write(f"**Vector Store:** {'✅ Hazır' if st.session_state.vectorstore_ready else '❌ Yok'}")
            st.write(f"**Mesajlar:** {len(st.session_state.messages)}")
        
        # Configuration
        with st.expander("🔧 Konfigürasyon"):
            st.session_state.chunk_size = st.slider(
                "Chunk Size", 
                min_value=200, 
                max_value=2000, 
                value=1000,
                help="Metin parçalarının boyutu"
            )
            
            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap", 
                min_value=0, 
                max_value=500, 
                value=200,
                help="Parçalar arası örtüşme"
            )
            
            st.session_state.temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3,
                help="Yanıt rastgeleliği"
            )
            
            st.session_state.retrieval_k = st.slider(
                "Retrieval K", 
                min_value=1, 
                max_value=10, 
                value=4,
                help="Geri getirilen doküman sayısı"
            )
        
        # Actions
        st.subheader("🎯 Aksiyonlar")
        
        if st.button("🗑️ Sohbeti Temizle"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Sistemi Sıfırla"):
            st.session_state.clear()
            st.rerun()


def main():
    """Ana uygulama fonksiyonu."""
    st.title("🤖 RAG Chatbot Assistant")
    st.markdown("*Retrieval Augmented Generation ile Güçlendirilmiş Chatbot*")
    
    # Check dependencies
    deps_ok, error_msg = check_dependencies()
    if not deps_ok:
        st.error(f"❌ Bağımlılık hatası: {error_msg}")
        st.info("🔧 Gerekli kütüphaneleri yüklemek için: `pip install -r requirements.txt`")
        st.stop()
    
    # Display sidebar
    display_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔑 Kurulum", "📁 Dokümanlar", "💬 Sohbet"])
    
    with tab1:
        st.info("👋 RAG Chatbot'a hoş geldiniz! Başlamak için API key'inizi girin.")
        api_ready = display_api_key_input()
        
        if api_ready:
            st.success("✅ Sistem hazır. Dokümanlar sekmesine geçebilirsiniz.")
    
    with tab2:
        if not check_api_key():
            st.warning("⚠️ Önce API key'inizi ayarlayın.")
        else:
            display_file_upload()
    
    with tab3:
        if not check_api_key():
            st.warning("⚠️ Önce API key'inizi ayarlayın.")
        elif not st.session_state.vectorstore_ready:
            st.warning("⚠️ Önce dokümanları yükleyin.")
        else:
            display_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔗 **Proje Detayları:** [GitHub Repository](https://github.com/your-username/rag-chatbot) | "
        "📖 **Dokümantasyon:** [chatbot.prompt.md](chatbot.prompt.md)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Uygulama hatası: {str(e)}")
        st.exception(e)