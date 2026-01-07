# 🔍 Vietnam Fake News Detector - PTIT

An AI-powered fake news detection system for TikTok videos, integrating Chrome Extension and Backend API with advanced Machine Learning technologies. Developed by Posts and Telecommunications Institute of Technology (PTIT).

## 📋 Overview

This project is a complete system for detecting fake news on TikTok platform, including:

- **Chrome Extension**: Browser extension "Kiểm Tin Giả" to analyze TikTok videos directly on the website
- **Backend API**: Python API server using FastAPI for analysis and prediction
- **Machine Learning Model**: HAN (Hierarchical Attention Network) model optimized with ONNX Runtime
- **RAG System**: Retrieval-Augmented Generation system to verify information with trusted news sources
- **Media Processing**: Video/image processing with OCR (Optical Character Recognition) and STT (Speech-to-Text)

## 🏗️ System Architecture

```
┌─────────────────┐
│ Chrome Extension│
│ "Kiểm Tin Giả"  │
│  (extension/)   │
└────────┬────────┘
         │ HTTP API
         ▼
┌─────────────────┐
│  FastAPI Server │
│   (backend/)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │        │
    ▼        ▼
┌────────┐ ┌──────────┐
│  HAN   │ │   RAG    │
│ Model  │ │  Service │
│ (GPU)  │ │  (GPU)   │
└────────┘ └────┬─────┘
                │
                ▼
         ┌──────────────┐
         │  Supabase DB │
         │  (PostgreSQL)│
         └──────────────┘
```

## 📁 Directory Structure

```
detect-fake-news/
├── backend/              # Python Backend API
│   ├── routers/         # API endpoints
│   ├── services/        # Business logic (GPU-accelerated)
│   ├── scripts/         # Utility scripts
│   └── main.py          # FastAPI app entry
│
├── extension/            # Chrome Extension "Kiểm Tin Giả"
│   ├── background/       # Service worker
│   ├── content/          # Content scripts
│   ├── popup/            # Extension popup UI (PTIT branding)
│   ├── icons/            # Extension icons + PTIT logo
│   └── manifest.json     # Extension manifest
│
├── crawl/               # Data crawling scripts
│   ├── crawl_video.py   # TikTok video crawler
│   └── *.ipynb          # Data processing notebooks
│
├── dataset/             # Datasets and analysis
│   ├── final_dataset_for_training.csv
│   ├── analysis_fake_real.py
│   └── data_analysis.ipynb
│
├── train/               # Model training & experiments
│   ├── train-baseline-phobert.py    # Experiment 1: Baseline PhoBERT
│   ├── train-author-embedding.py    # Experiment 2: PhoBERT + Author Embedding
│   ├── train-MLM_Prompt.py          # Experiment 3: Prompt-based MLM
│   └── RAG_HAN_v4.ipynb             # Experiment 4: HAN + RAG (Production)
│
└── models/              # Trained models
    └── han_rag_model.onnx
```

## 🚀 Installation and Setup

### System Requirements

- Python 3.8+
- Node.js 16+
- Chrome/Edge browser
- PostgreSQL with pgvector extension (or Supabase)
- FFmpeg (for media processing)
- **CUDA 12.x** (recommended) - NVIDIA GPU with compatible driver

### 1. Backend API Setup

```bash
cd backend
pip install -r requirement.txt
```

**Note:** Backend automatically detects CUDA. If NVIDIA GPU is available, all services will use GPU for acceleration.

Create `.env` file:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
MODEL_PATH=./models/han_rag_model.onnx
TOKENIZER_PATH=vinai/phobert-base-v2
EMBEDDING_MODEL=keepitreal/vietnamese-sbert
PORT=8000
HOST=0.0.0.0
```

Run server:
```bash
python main.py
```

Server will display CUDA info on startup:
```
✅ CUDA Available: NVIDIA GeForce RTX 3050 Ti Laptop GPU
✅ CUDA Version: 12.1
CUDA: ✅ GPU
```

### 2. Chrome Extension Setup

```bash
cd extension
npm install
```

Load extension into Chrome:
1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select `extension/` folder
5. Extension will appear as **"Kiểm Tin Giả - PTIT"**

### 3. Database Setup

Run SQL schema from `extension/database/supabase_schema.sql` on Supabase or PostgreSQL.

## 🎯 Key Features

### 1. TikTok Video Analysis

**Smart processing flow:**
- **Video URL** (`/video/`) → Uses **Whisper (STT)** to transcribe audio
- **Photo URL** (`/photo/`) → Uses **VietOCR** to extract text from images
- Automatically detects content type from URL
- Caches results for optimal performance

### 2. RAG Verification
- Searches for similar articles from trusted news sources
- Verifies information with similarity search
- Adjusts confidence based on evidence

### 3. GPU Acceleration
- **Whisper (STT)**: GPU-accelerated with `medium` model
- **VietOCR**: GPU support for text extraction
- **ONNX Model**: CUDA Execution Provider for fast inference
- **SentenceTransformer**: GPU for embedding generation
- Automatic fallback to CPU if GPU unavailable

### 4. User Reporting
- Users can report incorrect results
- System tracking to improve model

## 🔧 Technologies Used

### Backend
- **FastAPI**: Web framework
- **ONNX Runtime GPU**: Optimized model inference with CUDA
- **Supabase**: Database and vector search
- **Sentence Transformers**: Embedding generation (GPU)
- **VietOCR**: Vietnamese OCR (GPU)
- **Whisper**: Speech-to-Text (GPU)
- **yt-dlp**: Video download

### Frontend
- **Chrome Extension API**: Extension development
- **Vanilla JavaScript**: UI logic
- **Light Theme UI**: Light interface with PTIT logo

### ML/AI
- **HAN Model**: Hierarchical Attention Network
- **PhoBERT**: Vietnamese BERT tokenizer
- **Vietnamese SBERT**: Sentence embeddings
- **RAG**: Retrieval-Augmented Generation

## 📊 Model Architecture

### HAN Model
- **Input**: Title (caption) + Content (OCR or STT depending on URL type)
- **Tokenizer**: PhoBERT-base-v2
- **Architecture**: Hierarchical Attention with chunk selection
- **Output**: Binary classification (REAL/FAKE) with confidence score
- **Model on HuggingFace**: [vn_fake_news_v2](https://huggingface.co/jamus0702/vn_fake_news_v2/tree/main)

### RAG Pipeline
1. Chunk selection from content based on title similarity
2. Vector search in news corpus
3. Similarity threshold: Adaptive (0.5-0.7 for search, 0.6-0.85 for verification)
4. Confidence adjustment based on matching articles

## 📝 API Endpoints

### `/health`
Health check with CUDA info:

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "database": "connected",
  "cuda": {
    "available": true,
    "gpu": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
    "version": "12.1",
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  }
}
```

### `/api/v1/predict`
Predict fake/real news from TikTok video

**Request:**
```json
{
  "video_id": "1234567890",
  "video_url": "https://tiktok.com/@user/video/123",
  "caption": "Video caption...",
  "ocr_text": "Text from OCR...",
  "stt_text": "Text from STT...",
  "author_id": "username"
}
```

**Response:**
```json
{
  "video_id": "1234567890",
  "prediction": "FAKE",
  "confidence": 0.85,
  "method": "rag_enhanced",
  "rag_used": true,
  "probabilities": {
    "REAL": 0.15,
    "FAKE": 0.85
  },
  "processing_time_ms": 1234.5
}
```

### `/api/v1/process-media`
Process media (OCR or STT depending on URL type)

**Flow:**
- URL contains `/video/` → Only runs STT (Whisper)
- URL contains `/photo/` → Only runs OCR (VietOCR)

### `/api/v1/report`
Report incorrect results

### `/api/v1/predict-text`
Predict from text only (without media processing)

## 🧪 Testing

```bash
# Test API with CUDA info
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

## 📈 Performance

- **Prediction time**: ~1-3 seconds (no cache, GPU)
- **Cache hit**: <100ms
- **Media processing**: 
  - Video (STT): ~3-5 seconds (GPU)
  - Photo (OCR): ~2-4 seconds (GPU)
- **RAG search**: ~500ms-1s (GPU)

## 🎨 UI/UX

### Extension Popup
- **Name**: "Kiểm Tin Giả - PTIT"
- **Logo**: PTIT logo at top left
- **Theme**: Light theme with white background, black border
- **Color coding**:
  - 🟢 REAL: Green (#2e7d32)
  - 🔴 FAKE: Red (#d32f2f)
  - ⚪ UNCERTAIN: Orange (#f57c00)

## 🔒 Security

- Row Level Security (RLS) on Supabase
- Service role authentication
- Input validation and sanitization
- CORS middleware

## 📚 Additional Documentation

- [Backend API README](backend/README.md)
- [Chrome Extension README](extension/README.md)
- [Crawling Scripts README](crawl/README.md)
- [Training & Experiments Guide](train/README.md)

## 📄 License

This project is released under the MIT License.

## 👥 Authors

- *[Đặng Thị Bích Trâm](https://github.com/jj4002)*
- *[Đỗ Minh Bảo Huy](https://github.com/ddooxhuy09)*
- *[Trần Anh Tuấn](https://github.com/tuanhqv123)*

**Posts and Telecommunications Institute of Technology (PTIT)**

## 🙏 Acknowledgments

- PhoBERT team for Vietnamese BERT model
- VietOCR team for Vietnamese OCR
- OpenAI Whisper for STT
- Supabase for infrastructure
- Model published on [HuggingFace](https://huggingface.co/jamus0702/vn_fake_news_v2/tree/main)

## 📊 Datasets

The project uses the following datasets for training and evaluation:

- **[Vietnamese Fake News Detection](https://github.com/hiepnguyenduc2005/Vietnamese-Fake-News-Detection)**: Dataset from ReINTEL with nearly 10,000 labeled examples, used for training baseline models
- **[VFND Vietnamese Fake News Datasets](https://github.com/WhySchools/VFND-vietnamese-fake-news-datasets)**: Collection of Vietnamese articles and Facebook posts classified (228-254 articles), including both Article Contents and Social Contents
