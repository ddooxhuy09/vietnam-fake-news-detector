# 🔍 Kiểm Tin Giả - PTIT

Hệ thống phát hiện tin giả trên TikTok sử dụng AI, tích hợp Chrome Extension và Backend API với các công nghệ Machine Learning tiên tiến. Dự án được phát triển bởi Học viện Công nghệ Bưu chính Viễn thông (PTIT).

## 📋 Tổng quan

Dự án này là một hệ thống hoàn chỉnh để phát hiện tin giả trên nền tảng TikTok, bao gồm:

- **Chrome Extension**: Extension trình duyệt "Kiểm Tin Giả" để phân tích video TikTok trực tiếp trên trang web
- **Backend API**: API server Python sử dụng FastAPI để xử lý phân tích và dự đoán
- **Machine Learning Model**: Mô hình HAN (Hierarchical Attention Network) được tối ưu hóa với ONNX Runtime
- **RAG System**: Hệ thống Retrieval-Augmented Generation để xác minh thông tin với nguồn tin đáng tin cậy
- **Media Processing**: Xử lý video/ảnh với OCR (Optical Character Recognition) và STT (Speech-to-Text)

## 🏗️ Kiến trúc hệ thống

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

## 📁 Cấu trúc thư mục

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
└── train/               # Model training & experiments
    ├── train-baseline-phobert.py    # Experiment 1: Baseline PhoBERT
    ├── train-author-embedding.py    # Experiment 2: PhoBERT + Author Embedding
    ├── train-MLM_Prompt.py          # Experiment 3: Prompt-based MLM
    └── train-rag-han.ipynb          # Experiment 4: HAN + RAG (Production)
```

## 🚀 Cài đặt và Chạy

### Yêu cầu hệ thống

- Python 3.8+
- Node.js 16+
- Chrome/Edge browser
- PostgreSQL với pgvector extension (hoặc Supabase)
- FFmpeg (cho xử lý media)
- **CUDA 12.x** (khuyến nghị) - GPU NVIDIA với driver tương thích

### 1. Cài đặt Backend API

```bash
cd backend
pip install -r requirement.txt
```

**Lưu ý:** Backend tự động detect CUDA. Nếu có GPU NVIDIA, tất cả services sẽ dùng GPU để tăng tốc.

Tạo file `.env`:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
MODEL_PATH=./models/han_rag_model.onnx
TOKENIZER_PATH=vinai/phobert-base-v2
EMBEDDING_MODEL=keepitreal/vietnamese-sbert
PORT=8000
HOST=0.0.0.0
```

Chạy server:
```bash
python main.py
```

Server sẽ hiển thị CUDA info khi khởi động:
```
✅ CUDA Available: NVIDIA GeForce RTX 3050 Ti Laptop GPU
✅ CUDA Version: 12.1
CUDA: ✅ GPU
```

### 2. Cài đặt Chrome Extension

```bash
cd extension
npm install
```

Load extension vào Chrome:
1. Mở `chrome://extensions/`
2. Bật "Developer mode"
3. Click "Load unpacked"
4. Chọn thư mục `extension/`
5. Extension sẽ hiển thị với tên **"Kiểm Tin Giả - PTIT"**

### 3. Setup Database

Chạy SQL schema từ `extension/database/supabase_schema.sql` trên Supabase hoặc PostgreSQL.

## 🎯 Tính năng chính

### 1. Phân tích Video TikTok

**Flow xử lý thông minh:**
- **Video URL** (`/video/`) → Sử dụng **Whisper (STT)** để transcribe audio
- **Photo URL** (`/photo/`) → Sử dụng **VietOCR** để extract text từ hình ảnh
- Tự động detect loại content từ URL
- Cache kết quả để tối ưu hiệu suất

### 2. RAG Verification
- Tìm kiếm bài viết tương tự từ nguồn tin đáng tin cậy
- Xác minh thông tin với similarity search
- Điều chỉnh confidence dựa trên bằng chứng

### 3. GPU Acceleration
- **Whisper (STT)**: GPU-accelerated với model `medium`
- **VietOCR**: GPU support cho text extraction
- **ONNX Model**: CUDA Execution Provider cho inference nhanh
- **SentenceTransformer**: GPU cho embedding generation
- Tự động fallback về CPU nếu không có GPU

### 4. User Reporting
- Người dùng có thể báo cáo kết quả sai
- Hệ thống tracking để cải thiện model

## 🔧 Công nghệ sử dụng

### Backend
- **FastAPI**: Web framework
- **ONNX Runtime GPU**: Model inference tối ưu với CUDA
- **Supabase**: Database và vector search
- **Sentence Transformers**: Embedding generation (GPU)
- **VietOCR**: OCR tiếng Việt (GPU)
- **Whisper**: Speech-to-Text (GPU)
- **yt-dlp**: Video download

### Frontend
- **Chrome Extension API**: Extension development
- **Vanilla JavaScript**: UI logic
- **Light Theme UI**: Giao diện sáng với logo PTIT

### ML/AI
- **HAN Model**: Hierarchical Attention Network
- **PhoBERT**: Vietnamese BERT tokenizer
- **Vietnamese SBERT**: Sentence embeddings
- **RAG**: Retrieval-Augmented Generation

## 📊 Model Architecture

### HAN Model
- **Input**: Title (caption) + Content (OCR hoặc STT tùy loại URL)
- **Tokenizer**: PhoBERT-base-v2
- **Architecture**: Hierarchical Attention với chunk selection
- **Output**: Binary classification (REAL/FAKE) với confidence score
- **Model trên HuggingFace**: [vn_fake_news_v2](https://huggingface.co/jamus0702/vn_fake_news_v2/tree/main)

### RAG Pipeline
1. Chunk selection từ content dựa trên title similarity
2. Vector search trong news corpus
3. Similarity threshold: 0.75
4. Confidence adjustment dựa trên matching articles

## 📝 API Endpoints

### `/health`
Health check với CUDA info:

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
Dự đoán tin giả/thật từ video TikTok

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
Xử lý media (OCR hoặc STT tùy loại URL)

**Flow:**
- URL có `/video/` → Chỉ chạy STT (Whisper)
- URL có `/photo/` → Chỉ chạy OCR (VietOCR)

### `/api/v1/report`
Báo cáo kết quả sai

## 🧪 Testing

```bash
# Test API với CUDA info
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

## 📈 Performance

- **Prediction time**: ~1-3 giây (không cache, GPU)
- **Cache hit**: <100ms
- **Media processing**: 
  - Video (STT): ~3-5 giây (GPU)
  - Photo (OCR): ~2-4 giây (GPU)
- **RAG search**: ~500ms-1s (GPU)

## 🎨 UI/UX

### Extension Popup
- **Tên**: "Kiểm Tin Giả - PTIT"
- **Logo**: PTIT logo ở góc trái trên
- **Theme**: Light theme với nền trắng, viền đen
- **Color coding**:
  - 🟢 REAL: Green (#2e7d32)
  - 🔴 FAKE: Red (#d32f2f)
  - ⚪ UNCERTAIN: Orange (#f57c00)

## 🔒 Bảo mật

- Row Level Security (RLS) trên Supabase
- Service role authentication
- Input validation và sanitization
- CORS middleware

## 📚 Tài liệu thêm

- [Backend API README](backend/README.md)
- [Chrome Extension README](extension/README.md)
- [Crawling Scripts README](crawl/README.md)
- [Training & Experiments Guide](train/README.md)

## 📄 License

Dự án này được phát hành dưới giấy phép MIT.

## 👥 Tác giả

- *[Đặng Thị Bích Trâm](https://github.com/jj4002)*
- *[Đỗ Minh Bảo Huy](https://github.com/ddooxhuy09)*
- *[Trần Anh Tuấn](https://github.com/tuanhqv123)*

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**

## 🙏 Acknowledgments

- PhoBERT team cho Vietnamese BERT model
- VietOCR team cho OCR tiếng Việt
- OpenAI Whisper cho STT
- Supabase cho infrastructure
- Model được đăng tải trên [HuggingFace](https://huggingface.co/jamus0702/vn_fake_news_v2/tree/main)

## 📊 Datasets

Dự án sử dụng các datasets sau cho training và evaluation:

- **[Vietnamese Fake News Detection](https://github.com/hiepnguyenduc2005/Vietnamese-Fake-News-Detection)**: Dataset từ ReINTEL với gần 10,000 examples được gán nhãn, sử dụng cho training baseline models
- **[VFND Vietnamese Fake News Datasets](https://github.com/WhySchools/VFND-vietnamese-fake-news-datasets)**: Tập hợp các bài báo tiếng Việt và Facebook posts được phân loại (228-254 bài), bao gồm cả Article Contents và Social Contents
