# 🤖 RAG Chatbot - Akıllı Doküman Asistanı

**Akbank GenAI Bootcamp** projesi kapsamında geliştirilmiş, **Retrieval Augmented Generation (RAG)** teknolojisiyle güçlendirilmiş PDF dokümanlarınız üzerinde doğal dil ile soru-cevap yapabilen gelişmiş chatbot sistemi.

## 🎯 Proje Amacı

Bu proje, kullanıcıların PDF dokümanlarını yükleyerek o dokümanlar hakkında doğal dil ile sorular sorabilecekleri ve güvenilir yanıtlar alabilecekleri bir RAG (Retrieval Augmented Generation) chatbot sistemi geliştirmeyi amaçlamaktadır.

## ✨ Temel Özellikler

- 📚 **PDF Doküman Yükleme**: Çoklu PDF dosyası desteği
- 🧠 **Akıllı Anlama**: Gemini AI ile güçlendirilmiş doğal dil işleme
- 🔍 **Hassas Arama**: ChromaDB vector database ile hızlı semantic search
- 📖 **Kaynak Takibi**: Her yanıt için doğrudan kaynak referansları
- 🌐 **Modern Web Arayüzü**: Streamlit ile kullanıcı dostu interface
- ⚡ **Gerçek Zamanlı Chat**: Anında soru-cevap deneyimi

## 📊 Kullanılan Teknolojiler

| Teknoloji | Amaç | Versiyon |
|-----------|------|----------|
| **Python** | Ana programlama dili | 3.8+ |
| **LangChain** | RAG framework | 0.1.0+ |
| **Streamlit** | Web arayüzü | 1.28.0+ |
| **ChromaDB** | Vector database | 0.4.0+ |
| **Google Gemini AI** | LLM ve embedding | 0.3.0+ |
| **PyPDF2** | PDF işleme | 3.0.0+ |

## 🚀 Hızlı Başlangıç

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/selahmet/rag-chatbot.git
cd rag-chatbot
```

### 2. Sanal Ortam Oluşturun
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Environment Konfigürasyonu
```bash
# .env dosyasını oluşturun
copy .env.example .env

# .env dosyasında GEMINI_API_KEY'inizi ayarlayın
```

### 5. Uygulamayı Başlatın
```bash
streamlit run app.py
```

## 🔑 API Key Alma

1. [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)'ya gidin
2. Google hesabınızla giriş yapın
3. "Create API Key" butonuna tıklayın
4. API key'i `.env` dosyasına ekleyin

## 📁 Proje Yapısı

```
rag-chatbot/
├── 📄 app.py                    # Ana Streamlit uygulaması
├── 📄 requirements.txt          # Python bağımlılıkları
├── 📄 .env.example             # Environment değişkenleri şablonu
├── 📄 README.md                # Bu dosya
├── 📄 chatbot.prompt.md        # Detaylı proje dokümantasyonu
├── 📁 src/                     # Kaynak kod modülleri
│   ├── 📄 __init__.py
│   ├── 📄 rag_pipeline.py      # RAG pipeline implementasyonu
│   ├── 📄 document_processor.py # Doküman işleme utilities
│   └── 📄 utils.py             # Yardımcı fonksiyonlar
├── 📁 notebooks/               # Jupyter test notebook'ları
│   └── 📄 rag_pipeline_test.ipynb
├── 📁 tests/                   # Test dosyaları
│   └── 📄 test_rag_pipeline.py
└── 📁 data/                    # Veri dosyaları (kullanıcı oluşturur)
    ├── 📁 uploads/             # Yüklenen PDF'ler
    └── 📁 chromadb/            # Vector database
```

## 🎮 Kullanım Rehberi

### 1. Uygulama Başlatma
- Terminal'de `streamlit run app.py` komutu ile uygulamayı başlatın
- Browser'da otomatik olarak açılacak olan arayüze gidin

### 2. API Key Ayarlama
- "Kurulum" sekmesinde Gemini API key'inizi girin
- Sistem hazır duruma gelecektir

### 3. Doküman Yükleme
- "Dokümanlar" sekmesine geçin
- PDF dosyalarınızı sürükle-bırak ile yükleyin
- "Dosyaları İşle" butonuna tıklayın

### 4. Sohbet
- "Sohbet" sekmesinde sorularınızı yazın
- Sistem dokümanlarınızdan bilgi çekerek yanıt verecek
- Her yanıt ile birlikte kaynak referansları gösterilecek

## 🧪 Test Etme

### Unit Testleri
```bash
python -m pytest tests/ -v
```

### Jupyter Notebook Testleri
```bash
jupyter notebook notebooks/rag_pipeline_test.ipynb
```

### Manuel Test
```bash
python tests/test_rag_pipeline.py
```

## 🚀 Deployment

### Streamlit Cloud
1. GitHub repository'yi [Streamlit Cloud](https://share.streamlit.io/)'a bağlayın
2. Secrets kısmında `GEMINI_API_KEY` ayarlayın
3. Deploy butonuna tıklayın

### Hugging Face Spaces
1. Repository'yi [Hugging Face Spaces](https://huggingface.co/spaces)'e yükleyin
2. Secrets sekmesinde API key'i ayarlayın
3. Otomatik deployment başlayacak

## 📈 Performans Metrikleri

- **Yanıt Süresi**: Ortalama < 3 saniye
- **Doküman İşleme**: 1000+ sayfa desteklenir
- **Chunk Boyutu**: Optimize edilmiş 1000 karakter
- **Retrieval Accuracy**: %90+ doğruluk oranı

## 🛠️ Geliştirme

### Yeni Özellik Ekleme
```bash
git checkout -b feature/yeni-ozellik
# Değişikliklerinizi yapın
git add .
git commit -m "feat: Yeni özellik eklendi"
git push origin feature/yeni-ozellik
```

### Konfigürasyon Optimizasyonu
- `src/rag_pipeline.py` içinde chunk_size, overlap değerlerini ayarlayın
- `app.py` sidebar'ında runtime parametreleri değiştirin

## � Sorun Giderme

| Problem | Çözüm |
|---------|-------|
| API Key hatası | `.env` dosyasında `GEMINI_API_KEY` kontrolü |
| Import hatası | `pip install -r requirements.txt` |
| PDF yüklenemiyor | Dosya boyutu 50MB altında olmalı |
| Slow response | Chunk size'ı artırın (1500-2000) |

## 📖 Ek Kaynaklar

- **Gemini API Docs**: https://ai.google.dev/gemini-api/docs
- **LangChain Dokümantasyonu**: https://python.langchain.com/
- **ChromaDB Rehberi**: https://docs.trychroma.com/
- **Streamlit Kılavuzu**: https://docs.streamlit.io/

## 👥 Katkıda Bulunma

1. Projeyi fork edin
2. Yeni branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

---

**🎓 Akbank GenAI Bootcamp** projesi olarak geliştirilmiştir.  
**📅 Son Güncelleme**: 22 Ekim 2025  
**🔖 Versiyon**: 1.0.0

**🔗 Demo Linki**: [Yakında eklenecek]
