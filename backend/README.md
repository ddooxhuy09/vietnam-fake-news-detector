# Backend API Server

FastAPI server providing API for fake news detection on TikTok with advanced ML/AI features and GPU acceleration.

## 📋 Overview

This backend provides:
- **Prediction API**: Predict fake/real news from TikTok videos
- **Media Processing**: OCR and Speech-to-Text from videos with GPU support
- **RAG Verification**: Verify with trusted news sources
- **CUDA Detection**: Automatically detect and use GPU if available
- **Caching**: Cache results for optimal performance
- **Reporting**: User reporting system to improve model

## 🏗️ Architecture

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

## 📁 Directory Structure

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
│   ├── ocr_service.py      # OCR service (GPU)
│   ├── stt_service.py      # Speech-to-Text service (GPU)
│   └── supabase_client.py  # Database client
│
└── scripts/                # Utility scripts
    ├── generate_embeddings.py
    └── regenerate_embeddings.py
```

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirement.txt
```

**Key dependencies:**
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `onnxruntime-gpu`: Model inference with CUDA support
- `sentence-transformers`: Embeddings (GPU)
- `supabase`: Database client
- `vietocr`: Vietnamese OCR (GPU)
- `openai-whisper`: Speech-to-Text (GPU)
- `yt-dlp`: Video download
- `opencv-python`: Image processing
- `moviepy`: Audio extraction
- `torch`: PyTorch for CUDA detection

### 2. Configure Environment Variables

Create `.env` file:

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

Run SQL schema from `extension/database/supabase_schema.sql` on Supabase.

### 4. Run Server

```bash
python main.py
```

Server will automatically detect CUDA on startup:
```
✅ CUDA Available: NVIDIA GeForce RTX 3050 Ti Laptop GPU
✅ CUDA Version: 12.1
CUDA: ✅ GPU
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server will run at: `http://localhost:8000`

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

Predict fake/real news from TikTok video.

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
- `cached`: Result from cache
- `base_model`: HAN model only
- `rag_enhanced`: RAG verification used

### 3. Process Media (`/api/v1/process-media`)

Process media with smart routing based on URL type.

**Flow logic:**
- URL contains `/video/` → Only runs **Whisper (STT)**
- URL contains `/photo/` → Only runs **VietOCR**

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

### 4. Predict Text (`/api/v1/predict-text`)

Predict from text only (without media processing).

**Request:**
```json
{
  "text": "News article text...",
  "author_id": "username"
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 0.82,
  "method": "base_model",
  "rag_used": false,
  "probabilities": {
    "REAL": 0.18,
    "FAKE": 0.82
  }
}
```

### 5. Report (`/api/v1/report`)

Report incorrect prediction results.

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

### 6. Get Pending Reports (`/api/v1/reports/pending`)

Get list of pending reports awaiting review (admin).

**Query params:**
- `limit`: Number of reports (default: 50)

## 🔧 Service Details

### Inference Service (`services/inference.py`)

**HANONNXInference Class:**
- Load ONNX model with CUDA support
- Text normalization (Vietnamese)
- Chunk selection with RAG
- Model prediction

**GPU Configuration:**
- ONNX Runtime: `CUDAExecutionProvider` (if CUDA available)
- SentenceTransformer: `device='cuda'` (auto-detect)

**Methods:**
- `predict(title, content)`: Predict with HAN model
- `_select_chunks_with_rag()`: Select important chunks

### RAG Service (`services/rag_service.py`)

**RAGService Class:**
- Vector similarity search (GPU)
- Verification with news corpus
- Confidence adjustment
- Adaptive threshold calculation

**GPU Configuration:**
- SentenceTransformer: `device='cuda'` (auto-detect)

**Methods:**
- `should_use_rag()`: Decide whether to use RAG
- `verify_with_sources()`: Search and verify
- `_calculate_adaptive_thresholds()`: Calculate adaptive thresholds based on content length

**RAG Triggers:**
- High confidence (>0.95)
- Clickbait patterns
- Sensitive topics
- Breaking news keywords
- Unknown source with high confidence

**Adaptive Thresholds:**
- Short text (<250 chars): Lower threshold (0.5 search, 0.6 verify)
- Long text (>1000 chars): Higher threshold (0.7 search, 0.85 verify)
- Normal text: Base threshold (0.65 search, 0.80 verify)

### Media Processor (`services/media_processor.py`)

**MediaProcessor Class:**
- Download video/image from TikTok
- **Smart URL detection**: Detect `/video/` vs `/photo/`
- Extract frames for OCR
- Extract audio for STT

**Methods:**
- `detect_tiktok_type(url)`: Detect URL type
- `download_media()`: Download with yt-dlp
- `extract_frames()`: Extract frames from video
- `extract_audio()`: Extract audio track

### OCR Service (`services/ocr_service.py`)

**OCRService Class:**
- Uses VietOCR (Vietnamese optimized)
- Extract text from frames/images
- GPU support with CUDA

**GPU Configuration:**
- Device: `cuda:0` (auto-detect)

**Methods:**
- `extract_text_from_frames()`: OCR from video frames
- `extract_text_from_image()`: OCR from image

### STT Service (`services/stt_service.py`)

**STTService Class:**
- Uses OpenAI Whisper (`medium` model)
- Transcribe audio to text
- GPU support with CUDA

**GPU Configuration:**
- Model: `medium` (VRAM efficient)
- Device: `cuda` (auto-detect)

**Methods:**
- `transcribe_audio()`: Speech-to-Text

### Supabase Client (`services/supabase_client.py`)

**SupabaseService Class:**
- Database operations
- Vector search
- Caching

**Methods:**
- `get_video()`: Get cached prediction
- `save_video()`: Save prediction
- `search_similar_news()`: Vector similarity search
- `save_report()`: Save user report

## 🖥️ GPU Support

### CUDA Detection

Backend automatically detects CUDA on startup:
- Check PyTorch CUDA availability
- Check ONNX Runtime CUDA providers
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

If CUDA is not available, all services automatically fallback to CPU.

## 🧪 Testing

### Test with curl

```bash
# Health check with CUDA info
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

### Test with Python

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

### Benchmarks (with GPU)

- **Prediction (no cache)**: ~1-3 seconds
- **Prediction (cached)**: <100ms
- **Media processing**:
  - Video (STT): ~3-5 seconds (GPU)
  - Photo (OCR): ~2-4 seconds (GPU)
- **RAG search**: ~500ms-1s (GPU)

### Optimization

1. **GPU Acceleration**: All ML services use GPU
2. **Caching**: Results cached in database
3. **Smart Routing**: Video → STT, Photo → OCR
4. **Async operations**: FastAPI async support
5. **Model optimization**: ONNX Runtime for fast inference

## 🐛 Troubleshooting

### CUDA not detected

**Issue:** `CUDA: ❌ CPU only` in logs
- **Solution:** 
  - Check NVIDIA driver: `nvidia-smi`
  - Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
  - Install `onnxruntime-gpu` instead of `onnxruntime`

### Model not loading

**Issue:** `FileNotFoundError: Model not found`
- **Solution:** Check `MODEL_PATH` in `.env`

### Database connection failed

**Issue:** `Supabase connection failed`
- **Solution:** Check `SUPABASE_URL` and `SUPABASE_KEY`

### OCR/STT not working

**Issue:** `VietOCR/Whisper not available`
- **Solution:** 
  - Install dependencies: `pip install vietocr openai-whisper`
  - Check FFmpeg is installed

### Memory issues (GPU)

**Issue:** Out of memory when processing media
- **Solution:**
  - Services run sequentially so no VRAM overflow
  - If still error, reduce model size (Whisper: `medium` → `base`)

## 🔒 Security

- **CORS**: Configured for extension origin
- **Input validation**: Pydantic models
- **SQL injection**: Supabase client auto-escapes
- **RLS**: Row Level Security on database

## 📈 Monitoring

### Logging

Server uses Python logging:
- Level: INFO
- Format: Timestamp, level, message
- Output: Console
- CUDA info logged on startup

### Metrics (can be added)

- Request count
- Response time
- Error rate
- Cache hit rate
- GPU utilization

## 🔮 Future Improvements

- [ ] WebSocket support for real-time updates
- [ ] Batch prediction API
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Prometheus metrics
- [ ] Distributed caching (Redis)
- [ ] Multi-GPU support

## 📄 License

MIT License
