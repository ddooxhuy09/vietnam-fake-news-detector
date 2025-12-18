# main.py
from dotenv import load_dotenv
import os

# ✅ LOAD .ENV TRƯỚC KHI IMPORT BẤT CỨ THỨ GÌ
load_dotenv()

# ============================================
# CUDA Detection
# ============================================
def detect_cuda():
    """Detect CUDA availability and return GPU info"""
    cuda_info = {
        "cuda_available": False,
        "gpu_name": None,
        "cuda_version": None,
        "onnx_providers": ["CPUExecutionProvider"]
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_info["cuda_available"] = True
            cuda_info["gpu_name"] = torch.cuda.get_device_name(0)
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["onnx_providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"✅ CUDA Available: {cuda_info['gpu_name']}")
            print(f"✅ CUDA Version: {cuda_info['cuda_version']}")
        else:
            print("⚠️ CUDA not available, using CPU")
    except ImportError:
        print("⚠️ PyTorch not installed, checking ONNX Runtime...")
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                cuda_info["cuda_available"] = True
                cuda_info["onnx_providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                print(f"✅ ONNX Runtime CUDA Provider available")
            else:
                print(f"⚠️ Available ONNX providers: {providers}")
        except ImportError:
            print("⚠️ ONNX Runtime not installed")
    
    return cuda_info

# Detect CUDA on startup
CUDA_INFO = detect_cuda()

# Debug: Print để check
print("="*50)
print("Environment Variables:")
print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL', 'NOT SET')[:30]}...")
print(f"SUPABASE_KEY: {os.getenv('SUPABASE_KEY', 'NOT SET')[:30]}...")
print(f"MODEL_PATH: {os.getenv('MODEL_PATH', 'NOT SET')}")
print(f"CUDA: {'✅ GPU' if CUDA_INFO['cuda_available'] else '❌ CPU only'}")
print("="*50)

# Sau đó mới import routers
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers import predict, media, reports
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TikTok Fake News Detection API",
    version="2.0.0",
    description="AI-powered fake news detection for TikTok videos"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(media.router, prefix="/api/v1", tags=["Media"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])

@app.get("/")
def root():
    return {
        "service": "TikTok Fake News Detection",
        "version": "2.2.2",
        "status": "online",
        "endpoints": {
            "predict": "/api/v1/predict",
            "process_media": "/api/v1/process-media",
            "report": "/api/v1/report",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    model_exists = os.path.exists(os.getenv("MODEL_PATH", "./models/han_rag_model.onnx"))
    supabase_connected = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))
    
    return {
        "status": "healthy",
        "model": "loaded" if model_exists else "missing",
        "database": "connected" if supabase_connected else "not configured",
        "cuda": {
            "available": CUDA_INFO["cuda_available"],
            "gpu": CUDA_INFO["gpu_name"],
            "version": CUDA_INFO["cuda_version"],
            "providers": CUDA_INFO["onnx_providers"]
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
