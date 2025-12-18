# services/ocr_service.py
import logging
from typing import List
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Auto-detect CUDA for OCR
def get_device():
    """Auto-detect CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ CUDA available for OCR: {gpu_name}")
            return "cuda:0"
    except ImportError:
        pass
    logger.info("⚠️ Using CPU for OCR")
    return "cpu"

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    HAS_VIETOCR = True
except ImportError:
    HAS_VIETOCR = False
    logger.warning("VietOCR not available")


class OCRService:
    def __init__(self):
        self.available = HAS_VIETOCR
        self.predictor = None
        self.device = get_device()
        
        if self.available:
            try:
                # Load VietOCR config
                config = Cfg.load_config_from_name('vgg_transformer')
                config['device'] = self.device  # Auto-detect GPU
                config['predictor']['beamsearch'] = False  # Tắt beamsearch cho nhanh hơn
                
                self.predictor = Predictor(config)
                logger.info(f"✅ VietOCR loaded (device={self.device})")
            except Exception as e:
                logger.error(f"Failed to load VietOCR: {e}")
                self.available = False
        else:
            logger.warning("VietOCR not available - OCR will be disabled")
    
    def extract_text_from_frames(self, frames: List[np.ndarray]) -> str:
        """Extract text from multiple frames using VietOCR"""
        
        if not self.available or not frames:
            logger.warning("VietOCR not available or no frames")
            return ""
        
        try:
            all_text = []
            seen_text = set()  # Deduplicate
            
            for i, frame in enumerate(frames):
                try:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # VietOCR predict
                    text = self.predictor.predict(pil_image)
                    
                    # Clean and deduplicate
                    text = text.strip()
                    if text and text not in seen_text:
                        all_text.append(text)
                        seen_text.add(text)
                        logger.debug(f"Frame {i}: {text[:50]}...")
                
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {e}")
                    continue
            
            # Combine all text
            combined = ' '.join(all_text)
            logger.info(f"✅ VietOCR extracted {len(combined)} chars from {len(frames)} frames")
            return combined
            
        except Exception as e:
            logger.error(f"VietOCR error: {e}")
            return ""

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from a single image file using VietOCR"""
        
        if not self.available:
            logger.warning("VietOCR not available")
            return ""
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot load image: {image_path}")
                return ""
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # VietOCR predict
            text = self.predictor.predict(pil_image)
            text = text.strip()
            
            logger.info(f"✅ VietOCR extracted {len(text)} chars from image")
            return text
            
        except Exception as e:
            logger.error(f"VietOCR error on image: {e}")
            return ""
