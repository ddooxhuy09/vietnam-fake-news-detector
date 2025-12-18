# routers/media.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from services.media_processor import MediaProcessor
from services.ocr_service import OCRService
from services.stt_service import STTService
from services.supabase_client import SupabaseService  # ← ADD THIS

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
media_processor = MediaProcessor()
ocr_service = OCRService()
stt_service = STTService()
db = SupabaseService()  # ← ADD THIS

class MediaRequest(BaseModel):
    video_id: str
    video_url: str

class MediaResponse(BaseModel):
    video_id: str
    ocr_text: str
    stt_text: str
    processing_time_ms: float

@router.post("/process-media", response_model=MediaResponse)
async def process_media(request: MediaRequest):
    import time
    start = time.time()

    file_path = None
    audio_path = None

    try:
        logger.info("="*70)
        logger.info(f"📹 Processing media: {request.video_id}")

        # ✅ DETECT TIKTOK TYPE FROM URL
        tiktok_type = media_processor.detect_tiktok_type(request.video_url)
        logger.info(f"🔍 TikTok type detected: {tiktok_type.upper()}")

        # ✅ CHECK CACHE FIRST
        logger.info("🔍 Checking cache before processing...")
        cached = db.get_video(request.video_id)

        if cached:
            logger.info(f"✅ Cache hit: {request.video_id}")
            logger.info(f"   Using cached OCR/STT data")

            processing_time = (time.time() - start) * 1000
            logger.info(f"✅ Returned from cache in {processing_time:.0f}ms")
            logger.info("="*70)

            return MediaResponse(
                video_id=request.video_id,
                ocr_text=cached.get("ocr_text", ""),
                stt_text=cached.get("stt_text", ""),
                processing_time_ms=processing_time,
            )

        logger.info("✅ No cache found, processing media...")

        # Download
        logger.info("⬇️ Downloading...")
        file_path, media_type = media_processor.download_media(
            request.video_url, request.video_id
        )

        logger.info(f"   Downloaded: {file_path}")
        logger.info(f"   File type: {media_type}")

        ocr_text = ""
        stt_text = ""

        # ============================================
        # NEW FLOW: Dựa vào URL type, không phải file type
        # - /video/ → STT (Whisper) only
        # - /photo/ → OCR (VietOCR) only
        # ============================================

        if tiktok_type == "video":
            # ========== VIDEO: Chỉ dùng WHISPER (STT) ==========
            logger.info("🎬 VIDEO URL → Using WHISPER (STT) only")

            # Extract audio for STT
            logger.info("🔊 Extracting audio...")
            audio_path = media_processor.extract_audio(file_path)

            if audio_path:
                logger.info(f"   Audio saved: {audio_path}")
                
                # Run STT with Whisper
                logger.info("🎤 Running Whisper Speech-to-Text...")
                stt_text = stt_service.transcribe_audio(audio_path, language="vi") or ""
                logger.info(f"   ✅ STT: {len(stt_text)} chars")
                if stt_text:
                    logger.info(f"   Preview: {stt_text[:150]}...")
            else:
                logger.warning("⚠️ No audio track found in video")

        elif tiktok_type == "photo":
            # ========== PHOTO: Chỉ dùng VietOCR ==========
            logger.info("🖼️ PHOTO URL → Using VietOCR only")

            if media_type == "video":
                # Photo slideshow được download như video
                logger.info("📸 Photo slideshow (video format) → Extracting frames...")
                frames = media_processor.extract_frames(file_path, max_frames=10)
                
                if frames:
                    logger.info(f"🔤 Running VietOCR on {len(frames)} frames...")
                    ocr_text = ocr_service.extract_text_from_frames(frames)
                    logger.info(f"   ✅ OCR: {len(ocr_text)} chars")
                    if ocr_text:
                        logger.info(f"   Preview: {ocr_text[:150]}...")

            elif media_type == "image":
                # Single image
                logger.info("📸 Single image → Running VietOCR...")
                ocr_text = ocr_service.extract_text_from_image(file_path)
                logger.info(f"   ✅ OCR: {len(ocr_text)} chars")
                if ocr_text:
                    logger.info(f"   Preview: {ocr_text[:150]}...")

            else:
                logger.warning(f"⚠️ Photo URL but unsupported file type: {media_type}")

        processing_time = (time.time() - start) * 1000

        logger.info("="*70)
        logger.info("✅ Media processing complete:")
        logger.info(f"   TikTok type: {tiktok_type.upper()}")
        logger.info(f"   OCR: {len(ocr_text)} chars")
        logger.info(f"   STT: {len(stt_text)} chars")
        logger.info(f"   Time: {processing_time:.0f}ms")
        logger.info("="*70)

        return MediaResponse(
            video_id=request.video_id,
            ocr_text=ocr_text,
            stt_text=stt_text,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"❌ Media processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 🧹 Luôn dọn file tạm (video + audio) sau khi xử lý xong
        try:
            media_processor.cleanup(file_path, audio_path)
        except Exception:
            # tránh làm vỡ response nếu cleanup lỗi
            pass

