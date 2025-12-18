# routers/predict.py - FULL CODE (Removed risk assessment)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import logging

from services.supabase_client import SupabaseService
from services.rag_service import RAGService
from services.inference import HANONNXInference

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
db = SupabaseService()
rag_service = RAGService()
model = HANONNXInference()


class PredictRequest(BaseModel):
    video_id: str
    video_url: str
    caption: str
    ocr_text: Optional[str] = ""
    stt_text: Optional[str] = ""
    author_id: Optional[str] = None


class PredictResponse(BaseModel):
    video_id: str
    prediction: str
    confidence: float
    method: str
    rag_used: bool
    probabilities: dict
    processing_time_ms: float


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start = time.time()
    
    try:
        logger.info("=" * 70)
        logger.info(f"📥 NEW REQUEST: {request.video_id}")
        
        # =========================
        # VALIDATE INPUT
        # =========================
        if not request.caption or len(request.caption.strip()) == 0:
            logger.error("❌ Invalid input: Empty caption")
            raise HTTPException(status_code=400, detail="Caption is required")
        
        # =========================
        # CHECK CACHE
        # =========================
        logger.info(f"🔍 Checking cache for video: {request.video_id}")
        cached = db.get_video(request.video_id)
        
        if cached:
            logger.info(f"✅ Cache hit: {request.video_id}")
            logger.info(f"   Cached prediction: {cached['prediction']} ({cached['confidence']:.4f})")
            logger.info(f"   Cached method: {cached.get('method', 'unknown')}")
            
            if cached["prediction"] == "FAKE":
                probs = {"FAKE": cached["confidence"], "REAL": 1 - cached["confidence"]}
            else:
                probs = {"REAL": cached["confidence"], "FAKE": 1 - cached["confidence"]}
            
            processing_time = (time.time() - start) * 1000
            logger.info(f"⚡ Completed in {processing_time:.0f}ms (cached)")
            logger.info("=" * 70)
            
            return PredictResponse(
                video_id=request.video_id,
                prediction=cached["prediction"],
                confidence=cached["confidence"],
                method="cached",
                rag_used=False,
                probabilities=probs,
                processing_time_ms=processing_time,
            )
        
        logger.info("✅ No cache found, running model...")
        
        # =========================
        # PREPARE INPUT
        # =========================
        title = request.caption.strip()
        
        content_parts = []
        if request.ocr_text and len(request.ocr_text.strip()) > 10:
            content_parts.append(request.ocr_text.strip())
            logger.info(f"   OCR: {len(request.ocr_text)} chars")
        
        if request.stt_text and len(request.stt_text.strip()) > 50:
            content_parts.append(request.stt_text.strip())
            logger.info(f"   STT: {len(request.stt_text)} chars")
        
        content = " ".join(content_parts)[:2000] if content_parts else title
        
        logger.info("📝 Input:")
        logger.info(f"   Title: {title[:100]}...")
        logger.info(f"   Content: {len(content)} chars")
        
        # =========================
        # BASE PREDICTION
        # =========================
        logger.info("🤖 Running base model...")
        base_result = model.predict(
            title=title,
            content=content
        )
        
        logger.info(f"   Base result: {base_result['prediction']} ({base_result['confidence']:.4f})")
        logger.info(f"   Probabilities: REAL={base_result['probabilities']['REAL']:.4f}, FAKE={base_result['probabilities']['FAKE']:.4f}")
        
        # =========================
        # RAG VERIFICATION
        # =========================
        rag_used = False
        method = "base_model"
        final_result = base_result
        
        if rag_service.should_use_rag(
            title, 
            content, 
            base_result["confidence"], 
            request.author_id or ""
        ):
            logger.info("🔍 Running RAG verification...")
            
            top_chunk = base_result.get('top_chunk', '')
            verification = rag_service.verify_with_sources(
                title=title,
                content=content,
                top_chunk=top_chunk
            )
            
            logger.info(f"   RAG result: {verification['recommendation']}")
            logger.info(f"   Similarity: {verification['similarity_score']:.2f}")
            
            if verification["matching_articles"]:
                rag_used = True
                top_article = verification['matching_articles'][0]
                logger.info(f"   Top match: {top_article['source']}")
                logger.info(f"   Title: {top_article['title'][:80]}...")
                
                similarity = verification["similarity_score"]
                recommendation = verification["recommendation"]
                
                if base_result["prediction"] == "REAL":
                    logger.info("✅ Base REAL, checking RAG confirmation...")
                    
                    if recommendation == "VERIFIED_REAL" and similarity >= 0.85:
                        logger.info(f"✅ VERIFIED_REAL (similarity {similarity:.2f} ≥ 0.85)")
                        method = "rag_enhanced"
                        boosted_conf = min(0.98, base_result["confidence"] * 1.15)
                        final_result = {
                            "prediction": "REAL",
                            "confidence": boosted_conf,
                            "probabilities": {
                                "REAL": boosted_conf,
                                "FAKE": 1 - boosted_conf,
                            },
                        }
                        logger.info(f"   Boosted: {base_result['confidence']:.4f} → {boosted_conf:.4f}")
                    else:
                        logger.info("   RAG did not strongly confirm → Keep base")
                        final_result = base_result
                
                elif base_result["prediction"] == "FAKE":
                    logger.info("🔍 Base FAKE, checking RAG evidence...")
                    
                    if recommendation == "VERIFIED_REAL" and similarity >= 0.85:
                        logger.warning("⚠️ VERIFIED_REAL (≥0.85) → Switching to REAL")
                        method = "rag_enhanced"
                        new_conf = max(0.7, min(0.95, 0.7 + (similarity - 0.85) * 3))
                        final_result = {
                            "prediction": "REAL",
                            "confidence": new_conf,
                            "probabilities": {
                                "REAL": new_conf,
                                "FAKE": 1 - new_conf,
                            },
                        }
                        logger.info(f"   Overridden to REAL: {new_conf:.4f}")
                    
                    elif recommendation == "NEEDS_REVIEW" and similarity >= 0.75:
                        logger.warning("⚠️ NEEDS_REVIEW → Reducing FAKE confidence")
                        adjusted_conf = max(0.55, base_result["confidence"] * 0.95)
                        final_result = {
                            "prediction": "FAKE",
                            "confidence": adjusted_conf,
                            "probabilities": {
                                "FAKE": adjusted_conf,
                                "REAL": 1 - adjusted_conf,
                            },
                        }
                        logger.info(f"   Adjusted: {adjusted_conf:.4f}")
                    else:
                        logger.info("   No strong evidence → Keep base")
                        final_result = base_result
            else:
                logger.info("   No matching articles found → Using base result")
                final_result = base_result
        
        # =========================
        # EXTRACT FINAL RESULT
        # =========================
        prediction = final_result["prediction"]
        confidence = final_result["confidence"]
        probabilities = final_result["probabilities"]
        
        logger.info(f"📊 Final result:")
        logger.info(f"   Prediction: {prediction}")
        logger.info(f"   Confidence: {confidence:.4f}")
        logger.info(f"   Method: {method}")
        logger.info(f"   RAG used: {rag_used}")
        
        # =========================
        # SAVE TO CACHE
        # =========================
        try:
            logger.info("💾 Saving to database...")
            save_data = {
                "video_id": request.video_id,
                "video_url": request.video_url,
                "caption": request.caption,
                "ocr_text": request.ocr_text,
                "stt_text": request.stt_text,
                "author_id": request.author_id,
                "prediction": prediction,
                "confidence": confidence,
                "method": method,
            }
            
            db.save_video(save_data)
            logger.info("✅ Saved to cache")
        
        except Exception as save_error:
            logger.error(f"❌ Cache save error: {save_error}")
            logger.warning("⚠️ Continuing without cache...")
        
        # =========================
        # RETURN RESPONSE
        # =========================
        processing_time = (time.time() - start) * 1000
        logger.info(f"✅ Completed in {processing_time:.0f}ms")
        logger.info("=" * 70)
        
        return PredictResponse(
            video_id=request.video_id,
            prediction=prediction,
            confidence=confidence,
            method=method,
            rag_used=rag_used,
            probabilities=probabilities,
            processing_time_ms=processing_time,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
