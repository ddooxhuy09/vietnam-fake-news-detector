# services/inference.py 

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from typing import Dict, List
import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)

# ===========================
# TEXT PREPROCESSING
# ===========================

class VietnameseTextNormalizer:
    """Text normalizer - EXACT MATCH với training"""
    
    def __init__(self):
        try:
            from underthesea import word_tokenize
            self.use_word_segment = True
            logger.info("✅ Underthesea available")
        except:
            self.use_word_segment = False
            logger.warning("⚠️ Underthesea not available")
    
    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)
    
    def clean_special_chars(self, text: str) -> str:
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(
            r'[^a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ0-9\s.,!?;:]',
            ' ', text
        )
        return text
    
    def word_segment(self, text: str) -> str:
        if not self.use_word_segment:
            return text
        try:
            from underthesea import word_tokenize
            return word_tokenize(text, format="text")
        except:
            return text
    
    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = self.normalize_unicode(text)
        text = text.strip()
        text = self.clean_special_chars(text)
        text = re.sub(r'\s+', ' ', text)
        text = self.word_segment(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class SemanticChunkRetriever:
    """Chunk retriever - EXACT MATCH với training"""
    
    def __init__(self, chunk_size=400):
        self.chunk_size = chunk_size
    
    def chunk_document(self, text: str) -> List[str]:
        """EXACT COPY từ training code"""
        if not text or len(text.strip()) == 0:
            return []
        
        # ✅ GIỐNG TRAINING: Split by [.!?-]
        sentences = re.split(r'[.!?\-]\s+', text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_len = len(sent)
            
            # ✅ Handle oversized chunks (GIỐNG TRAINING)
            if current_len + sent_len > self.chunk_size:
                if current_chunk:
                    chunks.append('. '.join(current_chunk))
                
                # ✅ Handle single long sentence
                if sent_len > self.chunk_size * 1.5:
                    words = sent.split()
                    temp_chunk = []
                    temp_len = 0
                    
                    for word in words:
                        if temp_len + len(word) > self.chunk_size:
                            if temp_chunk:
                                chunks.append(' '.join(temp_chunk))
                            temp_chunk = [word]
                            temp_len = len(word)
                        else:
                            temp_chunk.append(word)
                            temp_len += len(word) + 1
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                        current_len = temp_len
                else:
                    current_chunk = [sent]
                    current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks


# ===========================
# HAN ONNX INFERENCE
# ===========================

class HANONNXInference:
    """
    HAN Model Inference - CORRECTED VERSION
    EXACT MATCH với training architecture
    """
    
    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        retriever_model: str = "keepitreal/vietnamese-sbert",
        top_k: int = 5,
        chunk_size: int = 400,
        max_length: int = 256,
        min_chunks: int = 3,
        min_similarity: float = 0.15
    ):
        """
        Initialize HAN ONNX Inference
        
        Args:
            model_path: Path to ONNX model (han_rag_model.onnx)
            tokenizer_path: Local tokenizer dir or HF model name
            retriever_model: SentenceTransformer for RAG
            top_k: Number of chunks (default: 5, GIỐNG TRAINING)
            chunk_size: Max chars per chunk (default: 400)
            max_length: Max tokens per chunk (default: 256)
            min_chunks: Minimum chunks required (default: 3)
            min_similarity: Min cosine similarity (default: 0.15)
        """
        
        model_path = model_path or os.getenv("MODEL_PATH", "./models/han_rag_model.onnx")
        tokenizer_path = tokenizer_path or os.getenv("TOKENIZER_PATH", "vinai/phobert-base-v2")
        
        logger.info("=" * 70)
        logger.info("🔧 Initializing HAN ONNX Inference (CORRECTED)")
        logger.info("=" * 70)
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Tokenizer: {tokenizer_path}")
        logger.info(f"  Retriever: {retriever_model}")
        logger.info(f"  Config: top_k={top_k}, chunk_size={chunk_size}, max_length={max_length}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        
        # 1. Load ONNX model với GPU support
        logger.info("📦 Loading ONNX model...")
        # Auto-detect CUDA for ONNX Runtime
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers and torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger.info(f"✅ ONNX using GPU: {torch.cuda.get_device_name(0)}")
        else:
            providers = ['CPUExecutionProvider']
            logger.info("⚠️ ONNX using CPU (CUDA not available)")
        
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )
        logger.info(f"✅ ONNX model loaded (providers: {providers})")
        
        # 2. Load tokenizer
        logger.info("📦 Loading tokenizer...")
        if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        logger.info("✅ Tokenizer loaded")
        
        # 3. Load normalizer
        logger.info("📦 Initializing text normalizer...")
        self.normalizer = VietnameseTextNormalizer()
        
        # 4. Load retriever với GPU support
        logger.info("📦 Loading sentence retriever...")
        # Auto-detect CUDA for SentenceTransformer
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"✅ SentenceTransformer using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                logger.info("⚠️ SentenceTransformer using CPU (CUDA not available)")
        except:
            device = 'cpu'
            logger.info("⚠️ SentenceTransformer using CPU")
        
        self.retriever = SentenceTransformer(retriever_model, device=device)
        logger.info(f"✅ Sentence retriever loaded (device: {device})")
        
        # 5. Initialize chunker
        self.chunker = SemanticChunkRetriever(chunk_size=chunk_size)
        
        # 6. Config
        self.top_k = top_k
        self.max_length = max_length
        self.min_chunks = min_chunks
        self.min_similarity = min_similarity
        
        logger.info("=" * 70)
        logger.info("✅ HAN ONNX Inference initialized successfully!")
        logger.info("=" * 70)
    
    def _select_chunks_with_rag(self, title: str, content: str) -> List[str]:
        """
        RAG chunk selection - EXACT MATCH với training
        
        Args:
            title: Normalized title (used as query)
            content: Normalized content (to be chunked)
        
        Returns:
            List of selected chunks (length = self.top_k)
        """
        
        # 1. Chunk content
        raw_chunks = self.chunker.chunk_document(content)
        
        if not raw_chunks:
            logger.warning("  ⚠️ No chunks generated, returning empty list")
            # ✅ GIỐNG TRAINING: Duplicate empty or use title
            return [title if title else ""] * self.top_k
        
        logger.info(f"  Generated {len(raw_chunks)} chunks from content")
        
        # 2. ✅ VALIDATE MINIMUM CHUNKS (GIỐNG TRAINING)
        if len(raw_chunks) < self.min_chunks:
            logger.info(f"  Only {len(raw_chunks)} chunks < {self.min_chunks}, duplicating...")
            while len(raw_chunks) < self.min_chunks:
                raw_chunks.extend(raw_chunks[:self.min_chunks - len(raw_chunks)])
        
        # 3. Nếu ít chunks, lấy hết
        if len(raw_chunks) <= self.top_k:
            logger.info(f"  Using all {len(raw_chunks)} chunks (≤ top_k)")
            selected_chunks = raw_chunks[:]
        else:
            # 4. ✅ RAG: Use title as query (GIỐNG TRAINING)
            query = title if len(title) > 5 else raw_chunks[0]
            
            try:
                # Encode query và chunks
                query_emb = self.retriever.encode(query, convert_to_tensor=True)
                chunk_embs = self.retriever.encode(raw_chunks, convert_to_tensor=True)
                
                # ✅ Cosine similarity (GIỐNG TRAINING)
                from sentence_transformers import util
                scores = util.cos_sim(query_emb, chunk_embs)[0]
                
                # ✅ FILTER LOW-SIMILARITY CHUNKS (GIỐNG TRAINING)
                valid_indices = (scores >= self.min_similarity).nonzero(as_tuple=True)[0]
                
                if len(valid_indices) < self.top_k:
                    # Not enough valid chunks, take top-k anyway
                    top_indices = scores.argsort(descending=True)[:self.top_k]
                else:
                    # Sort valid indices by similarity
                    valid_sims = scores[valid_indices]
                    sorted_valid = valid_indices[valid_sims.argsort(descending=True)]
                    top_indices = sorted_valid[:self.top_k]
                
                selected_chunks = [raw_chunks[i] for i in top_indices.tolist()]
                
                logger.info(f"  ✅ RAG selected {len(selected_chunks)}/{len(raw_chunks)} chunks")
                
            except Exception as e:
                logger.warning(f"  ⚠️ RAG failed: {e}, using fallback")
                # Fallback: first + last chunks
                mid = self.top_k // 2
                selected_chunks = raw_chunks[:mid] + raw_chunks[-self.top_k + mid:]
                selected_chunks = selected_chunks[:self.top_k]
        
        # 5. ✅ NO EMPTY PADDING - DUPLICATE INSTEAD (GIỐNG TRAINING)
        while len(selected_chunks) < self.top_k:
            selected_chunks.append(
                selected_chunks[0] if selected_chunks else title
            )
        
        # Truncate if over
        selected_chunks = selected_chunks[:self.top_k]
        
        return selected_chunks
    
    def predict(self, title: str, content: str) -> Dict:
        """
        Predict with HAN model - CORRECTED VERSION
        
        Args:
            title: Video title/caption
            content: Video content (OCR + STT combined)
        
        Returns:
            {
                'prediction': 'FAKE' | 'REAL',
                'confidence': float,
                'probabilities': {'REAL': float, 'FAKE': float}
            }
        """
        
        try:
            logger.info("🤖 Running HAN model prediction...")
            
            # 1. Normalize
            title_norm = self.normalizer.normalize(title)
            content_norm = self.normalizer.normalize(content)
            
            logger.info(f"  Title: {title_norm[:100]}...")
            logger.info(f"  Content: {len(content_norm)} chars")
            
            # 2. ✅ RAG chunk selection (GIỐNG TRAINING)
            selected_chunks = self._select_chunks_with_rag(title_norm, content_norm)
            
            # 3. ✅ Tokenize chunks với max_length=256 (GIỐNG TRAINING!)
            chunk_encodings = self.tokenizer(
                selected_chunks,
                max_length=self.max_length,  # 256
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            
            # 4. ✅ Prepare ONNX inputs (ĐÚNG shape!)
            chunk_input_ids = np.expand_dims(
                chunk_encodings['input_ids'], axis=0
            ).astype(np.int64)  # [1, 5, 256]
            
            chunk_attention_masks = np.expand_dims(
                chunk_encodings['attention_mask'], axis=0
            ).astype(np.int64)  # [1, 5, 256]
            
            logger.info(f"  chunk_input_ids shape: {chunk_input_ids.shape}")
            logger.info(f"  chunk_attention_masks shape: {chunk_attention_masks.shape}")
            
            # 5. ✅ Run ONNX inference (CHỈ 2 INPUTS!)
            onnx_inputs = {
                'chunk_input_ids': chunk_input_ids,
                'chunk_attention_masks': chunk_attention_masks
            }
            
            onnx_outputs = self.session.run(None, onnx_inputs)
            
            # 6. Post-process
            logits = onnx_outputs[0][0]  # [2]
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            prediction_idx = int(np.argmax(probs))
            confidence = float(probs[prediction_idx])
            prediction = 'FAKE' if prediction_idx == 1 else 'REAL'
            
            logger.info(f"✅ Prediction: {prediction} ({confidence:.4f})")
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'REAL': float(probs[0]),
                    'FAKE': float(probs[1])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Inference error: {e}", exc_info=True)
            raise