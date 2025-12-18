# scripts/regenerate_embeddings.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from services.supabase_client import SupabaseService
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def regenerate_embeddings():
    """Generate embeddings only for articles with NULL embedding"""
    
    logger.info("🚀 Generating embeddings for NULL records...")
    
    # ✅ LOAD MODEL ONCE
    logger.info("Loading model...")
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    logger.info(f"✅ Model loaded: {model.max_seq_length} max_seq_length")
    
    # Connect to Supabase
    supabase = SupabaseService()
    
    # Get all articles
    logger.info("Fetching all articles...")
    response = supabase.client.table('news_corpus').select('*').execute()
    
    all_articles = response.data
    logger.info(f"Found {len(all_articles)} total articles")
    
    # ✅ FILTER: Only articles with NULL embedding
    articles_without_embedding = [
        article for article in all_articles 
        if article.get('embedding') is None
    ]
    
    logger.info(f"Found {len(articles_without_embedding)} articles without embedding")
    
    if len(articles_without_embedding) == 0:
        logger.info("✅ All articles already have embeddings!")
        return
    
    # ✅ GENERATE ONLY FOR NULL EMBEDDINGS
    success = 0
    for i, article in enumerate(articles_without_embedding, 1):
        try:
            text = f"{article['title']} {article['content']}"
            
            # Generate with SAME model
            embedding = model.encode(text, normalize_embeddings=True)  # ← NORMALIZE!
            
            # Update
            supabase.client.table('news_corpus').update({
                'embedding': embedding.tolist()
            }).eq('id', article['id']).execute()
            
            success += 1
            logger.info(f"✅ [{i}/{len(articles_without_embedding)}] {article['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"❌ [{i}/{len(articles_without_embedding)}] Error: {e}")
    
    logger.info(f"🎉 Done! {success}/{len(articles_without_embedding)} embeddings generated")

if __name__ == "__main__":
    regenerate_embeddings()
