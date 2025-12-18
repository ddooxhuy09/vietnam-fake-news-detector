# Backend API Server

FastAPI server cung cấp API để phát hiện tin giả trên TikTok với các tính năng ML/AI tiên tiến và GPU acceleration.

## 📋 Tổng quan

Backend này cung cấp:
- **Prediction API**: Dự đoán tin giả/thật từ video TikTok
- **Media Processing**: OCR và Speech-to-Text từ video với GPU support
- **RAG Verification**: Xác minh với nguồn tin đáng tin cậy
- **CUDA Detection**: Tự động detect và sử dụng GPU nếu có
- **Caching**: Lưu kết quả để tối ưu performance
- **Reporting**: Hệ thống báo cáo để cải thiện model

## 🏗️ Kiến trúc

```
┌──────────────┐
│   FastAPI    │
│   (main.py)  │
│  CUDA Detect │
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌──────┐ ┌──────────┐
│Router│ │ Services │
│      │ │  (GPU)   │
└──┬───┘ └────┬─────┘
   │          │
   │    ┌─────┴─────┐
   │    │           │
   ▼    ▼           ▼
┌────┐ ┌────┐ ┌──────────┐
│Pred│ │Med │ │   RAG    │
│ict │ │ia  │ │ Service  │
│    │ │    │ │  (GPU)   │
└──┬─┘ └──┬─┘ └─────┬─────┘
   │      │         │
   │      │         │
   ▼      ▼         ▼
┌──────┐ ┌──────┐ ┌──────────┐
│HAN   │ │OCR/  │ │ Supabase │
│Model │ │STT   │ │   DB     │
│(GPU) │ │(GPU) │ │          │
└──────┘ └──────┘ └──────────┘
```

## 📁 Cấu trúc thư mục

```
backend/
├── main.py                 # FastAPI app entry point (CUDA detection)
├── requirement.txt          # Python dependencies
│
├── routers/                # API endpoints
│   ├── predict.py          # Prediction endpoint
│   ├── media.py            # Media processing endpoint (smart routing)
│   └── reports.py          # Reporting endpoint
│
├── services/               # Business logic (GPU-accelerated)
│   ├── inference.py        # HAN model inference (ONNX + CUDA)
│   ├── rag_service.py      # RAG verification (GPU)
│   ├── media_processor.py  # Video/image processing (URL type detection)
│   ├── ocr_service.py     # OCR service (GPU)
│   ├── stt_service.py     # Speech-to-Text service (GPU)
│   └── supabase_client.py # Database client
│
└── scripts/                # Utility scripts
    ├── generate_embeddings.py
    └── regenerate_embeddings.py
```

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirement.txt
```

**Key dependencies:**
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `onnxruntime-gpu`: Model inference với CUDA support
- `sentence-transformers`: Embeddings (GPU)
- `supabase`: Database client
- `vietocr`: Vietnamese OCR (GPU)
- `openai-whisper`: Speech-to-Text (GPU)
- `yt-dlp`: Video download
- `opencv-python`: Image processing
- `moviepy`: Audio extraction
- `torch`: PyTorch cho CUDA detection

### 2. Cấu hình Environment Variables

Tạo file `.env`:

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Model paths
MODEL_PATH=./models/han_rag_model.onnx
TOKENIZER_PATH=vinai/phobert-base-v2
EMBEDDING_MODEL=keepitreal/vietnamese-sbert

# Server
PORT=8000
HOST=0.0.0.0
```

### 3. Setup Database

Chạy SQL schema từ `extension/database/supabase_schema.sql` trên Supabase.

### 4. Chạy server

```bash
python main.py
```

Server sẽ tự động detect CUDA khi khởi động:
```
✅ CUDA Available: NVIDIA GeForce RTX 3050 Ti Laptop GPU
✅ CUDA Version: 12.1
CUDA: ✅ GPU
```

Hoặc với uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

## 📝 API Endpoints

### 1. Health Check

```http
GET /health
```

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

### 2. Predict (`/api/v1/predict`)

Dự đoán tin giả/thật từ video TikTok.

**Request:**
```json
{
  "video_id": "1234567890",
  "video_url": "https://tiktok.com/@user/video/123",
  "caption": "Video caption text...",
  "ocr_text": "Text extracted from video frames...",
  "stt_text": "Transcribed audio text...",
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

**Prediction Methods:**
- `cached`: Kết quả từ cache
- `base_model`: Chỉ dùng HAN model
- `rag_enhanced`: Có sử dụng RAG verification

### 3. Process Media (`/api/v1/process-media`)

Xử lý media với smart routing dựa trên URL type.

**Flow logic:**
- URL chứa `/video/` → Chỉ chạy **Whisper (STT)**
- URL chứa `/photo/` → Chỉ chạy **VietOCR**

**Request:**
```json
{
  "video_id": "1234567890",
  "video_url": "https://tiktok.com/@user/video/123"
}
```

**Response (Video):**
```json
{
  "video_id": "1234567890",
  "ocr_text": "",
  "stt_text": "Transcribed audio text...",
  "processing_time_ms": 3456.7
}
```

**Response (Photo):**
```json
{
  "video_id": "1234567890",
  "ocr_text": "Text extracted from images...",
  "stt_text": "",
  "processing_time_ms": 2345.6
}
```

### 4. Report (`/api/v1/report`)

Báo cáo kết quả prediction sai.

**Request:**
```json
{
  "video_id": "1234567890",
  "reported_prediction": "FAKE",
  "reason": "Optional reason text..."
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Report saved successfully"
}
```

### 5. Get Pending Reports (`/api/v1/reports/pending`)

Lấy danh sách reports đang chờ review (admin).

**Query params:**
- `limit`: Số lượng reports (default: 50)

## 🔧 Services Chi tiết

### Inference Service (`services/inference.py`)

**HANONNXInference Class:**
- Load ONNX model với CUDA support
- Text normalization (Vietnamese)
- Chunk selection với RAG
- Model prediction

**GPU Configuration:**
- ONNX Runtime: `CUDAExecutionProvider` (nếu có CUDA)
- SentenceTransformer: `device='cuda'` (auto-detect)

**Methods:**
- `predict(title, content)`: Dự đoán với HAN model
- `_select_chunks_with_rag()`: Chọn chunks quan trọng

### RAG Service (`services/rag_service.py`)

**RAGService Class:**
- Vector similarity search (GPU)
- Verification với news corpus
- Confidence adjustment

**GPU Configuration:**
- SentenceTransformer: `device='cuda'` (auto-detect)

**Methods:**
- `should_use_rag()`: Quyết định có dùng RAG không
- `verify_with_sources()`: Tìm kiếm và verify

**RAG Triggers:**
- High confidence (>0.95)
- Clickbait patterns
- Sensitive topics
- Breaking news keywords
- Unknown source với high confidence

### Media Processor (`services/media_processor.py`)

**MediaProcessor Class:**
- Download video/image từ TikTok
- **Smart URL detection**: Detect `/video/` vs `/photo/`
- Extract frames cho OCR
- Extract audio cho STT

**Methods:**
- `detect_tiktok_type(url)`: Detect URL type
- `download_media()`: Download với yt-dlp
- `extract_frames()`: Extract frames từ video
- `extract_audio()`: Extract audio track

### OCR Service (`services/ocr_service.py`)

**OCRService Class:**
- Sử dụng VietOCR (Vietnamese optimized)
- Extract text từ frames/images
- GPU support với CUDA

**GPU Configuration:**
- Device: `cuda:0` (auto-detect)

**Methods:**
- `extract_text_from_frames()`: OCR từ video frames
- `extract_text_from_image()`: OCR từ image

### STT Service (`services/stt_service.py`)

**STTService Class:**
- Sử dụng OpenAI Whisper (`medium` model)
- Transcribe audio sang text
- GPU support với CUDA

**GPU Configuration:**
- Model: `medium` (tiết kiệm VRAM)
- Device: `cuda` (auto-detect)

**Methods:**
- `transcribe_audio()`: Speech-to-Text

### Supabase Client (`services/supabase_client.py`)

**SupabaseService Class:**
- Database operations
- Vector search
- Caching

**Methods:**
- `get_video()`: Lấy cached prediction
- `save_video()`: Lưu prediction
- `search_similar_news()`: Vector similarity search
- `save_report()`: Lưu user report

## 🖥️ GPU Support

### CUDA Detection

Backend tự động detect CUDA khi khởi động:
- Kiểm tra PyTorch CUDA availability
- Kiểm tra ONNX Runtime CUDA providers
- Log GPU information

### GPU Services

| Service | Device | Model |
|---------|--------|-------|
| **Whisper (STT)** | `cuda` | `medium` |
| **VietOCR** | `cuda:0` | `vgg_transformer` |
| **ONNX Model** | `CUDAExecutionProvider` | `han_rag_model.onnx` |
| **SentenceTransformer (inference)** | `cuda` | `keepitreal/vietnamese-sbert` |
| **SentenceTransformer (RAG)** | `cuda` | `keepitreal/vietnamese-sbert` |

### Fallback

Nếu không có CUDA, tất cả services tự động fallback về CPU.

## 🧪 Testing

### Test với curl

```bash
# Health check với CUDA info
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test123",
    "video_url": "https://tiktok.com/@test/video/123",
    "caption": "Test caption"
  }'
```

### Test với Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "video_id": "test123",
        "video_url": "https://tiktok.com/@test/video/123",
        "caption": "Test caption"
    }
)
print(response.json())
```

## 📊 Performance

### Benchmarks (với GPU)

- **Prediction (no cache)**: ~1-3 giây
- **Prediction (cached)**: <100ms
- **Media processing**:
  - Video (STT): ~3-5 giây (GPU)
  - Photo (OCR): ~2-4 giây (GPU)
- **RAG search**: ~500ms-1s (GPU)

### Optimization

1. **GPU Acceleration**: Tất cả ML services dùng GPU
2. **Caching**: Kết quả được cache trong database
3. **Smart Routing**: Video → STT, Photo → OCR
4. **Async operations**: FastAPI async support
5. **Model optimization**: ONNX Runtime cho inference nhanh

## 🐛 Troubleshooting

### CUDA không detect được

**Vấn đề:** `CUDA: ❌ CPU only` trong logs
- **Giải pháp:** 
  - Kiểm tra NVIDIA driver: `nvidia-smi`
  - Kiểm tra PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
  - Cài đặt `onnxruntime-gpu` thay vì `onnxruntime`

### Model không load

**Vấn đề:** `FileNotFoundError: Model not found`
- **Giải pháp:** Kiểm tra `MODEL_PATH` trong `.env`

### Database connection failed

**Vấn đề:** `Supabase connection failed`
- **Giải pháp:** Kiểm tra `SUPABASE_URL` và `SUPABASE_KEY`

### OCR/STT không hoạt động

**Vấn đề:** `VietOCR/Whisper not available`
- **Giải pháp:** 
  - Cài đặt dependencies: `pip install vietocr openai-whisper`
  - Kiểm tra FFmpeg đã cài đặt

### Memory issues (GPU)

**Vấn đề:** Out of memory khi process media
- **Giải pháp:**
  - Services chạy tuần tự nên không lo hết VRAM
  - Nếu vẫn lỗi, có thể giảm model size (Whisper: `medium` → `base`)

## 🔒 Security

- **CORS**: Configured cho extension origin
- **Input validation**: Pydantic models
- **SQL injection**: Supabase client tự động escape
- **RLS**: Row Level Security trên database

## 📈 Monitoring

### Logging

Server sử dụng Python logging:
- Level: INFO
- Format: Timestamp, level, message
- Output: Console
- CUDA info được log khi khởi động

### Metrics (có thể thêm)

- Request count
- Response time
- Error rate
- Cache hit rate
- GPU utilization

## 🔮 Future Improvements

- [ ] WebSocket support cho real-time updates
- [ ] Batch prediction API
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Prometheus metrics
- [ ] Distributed caching (Redis)
- [ ] Multi-GPU support

## 📄 License

MIT License
