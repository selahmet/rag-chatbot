# 🤖 RAG Chatbot - Akıllı Doküman Asistanı

**Retrieval Augmented Generation (RAG)** teknolojisiyle güçlendirilmiş, PDF dokümanlarınız üzerinde doğal dil ile soru-cevap yapabilen gelişmiş chatbot sistemi.

## ✨ Neden Bu Proje?

- 📚 **Doküman Bilgisi**: PDF'lerinizdeki bilgilere anında erişim
- 🧠 **Akıllı Anlama**: Gemini AI ile güçlendirilmiş doğal dil anlama
- 🔍 **Hassas Arama**: Vector database ile hızlı ve doğru bilgi bulma
- 📖 **Kaynak Takibi**: Her yanıt için doğrudan kaynak referansları
- 🌐 **Kolay Kullanım**: Streamlit ile modern web arayüzü

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Repository'yi klonlayın
git clone <your-repo-url>
cd chatbot-genai

# Sanal ortam oluşturun
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Konfigürasyon

```bash
# .env dosyasını oluşturun
cp .env.example .env
# .env dosyasında GEMINI_API_KEY'inizi ayarlayın
```

### 3. Çalıştırma

```bash
# Streamlit uygulamasını başlatın
streamlit run app.py
```

## 📁 Proje Yapısı

```
chatbot-genai/
├── app.py                    # Ana Streamlit uygulaması
├── requirements.txt          # Python bağımlılıkları
├── .env.example             # Çevresel değişken şablonu
├── README.md                # Bu dosya
├── chatbot.prompt.md        # Detaylı proje dokümantasyonu
├── notebooks/               # Geliştirme notebook'ları
│   ├── rag_pipeline_test.ipynb
│   └── evaluation.ipynb
├── src/                     # Kaynak kod modülleri
│   ├── __init__.py
│   ├── rag_pipeline.py      # RAG pipeline implementasyonu
│   ├── document_processor.py # Doküman işleme
│   └── utils.py             # Yardımcı fonksiyonlar
├── data/                    # Örnek veri dosyaları
│   └── sample_documents/
├── tests/                   # Test dosyaları
│   └── test_rag_pipeline.py
└── chroma_db/              # ChromaDB veritabanı (otomatik oluşur)
```

## 🎯 Özellikler

- ✅ **PDF Doküman Yükleme**: Çoklu PDF dosyası desteği
- ✅ **Akıllı Chunking**: Optimal parça boyutları ile metin bölme
- ✅ **Semantic Search**: Anlamsal benzerlik bazlı arama
- ✅ **Source Citation**: Her yanıt için kaynak referansları
- ✅ **Real-time Chat**: Gerçek zamanlı soru-cevap arayüzü
- ✅ **Responsive UI**: Mobil ve desktop uyumlu tasarım

## 🧪 Test Senaryoları

### Functionality Tests
1. **Basic RAG Test**: Yüklenen dokümanlara dayalı soru-cevap
2. **Source Citation Test**: Kaynak atıflarının doğruluğu
3. **Out-of-Scope Test**: Kapsam dışı sorulara verilen yanıtlar

### Performance Metrics
- **Yanıt Süresi**: < 5 saniye
- **Relevance Score**: > 0.7
- **Source Accuracy**: %95+

## 📊 API Kullanım Rehberi

### Gemini API Key Alma
1. [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)'ya gidin
2. Google hesabınızla giriş yapın
3. "Get API Key" → "Create API Key" 
4. API key'i `.env` dosyasına kaydedin

## 🛠️ Geliştirme

### Yeni Özellik Ekleme
```bash
# Yeni branch oluşturun
git checkout -b feature/new-feature

# Değişikliklerinizi yapın ve commit edin
git add .
git commit -m "Add new feature"

# Push ve pull request oluşturun
git push origin feature/new-feature
```

### Test Çalıştırma
```bash
# Unit testler
python -m pytest tests/

# Notebook testleri
jupyter nbconvert --execute notebooks/rag_pipeline_test.ipynb
```

## 🚀 Deployment

### Streamlit Cloud
1. GitHub repository'yi Streamlit Cloud'a bağlayın
2. `secrets.toml` dosyasında API key'lerinizi ayarlayın
3. Otomatik deployment başlayacak

### Hugging Face Spaces
1. Repository'yi Hugging Face Spaces'e upload edin
2. `app.py` dosyasını Gradio versiyonu ile değiştirin
3. Secrets kısmında API key'leri ayarlayın

## 📖 Teknik Dokümantasyon

Detaylı teknik dokümantasyon için [`chatbot.prompt.md`](chatbot.prompt.md) dosyasını inceleyiniz.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request açın

## 📄 Lisans

Bu proje MIT lisansı ile lisanslanmıştır.

---

**Son Güncelleme**: 22 Ekim 2025  
**Versiyon**: 1.0.0