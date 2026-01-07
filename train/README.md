# Model Training & Experiments

Directory containing scripts and notebooks to train and experiment with different model architectures for fake news detection on TikTok.

## 📋 Overview

This project includes **4 main experiments** with different approaches:

1. **Baseline PhoBERT** (`train-baseline-phobert.py`) - Simple Sequence Classification
2. **PhoBERT + Author Embedding** (`train-author-embedding.py`) - Multi-modal with author information
3. **Prompt-based MLM** (`train-MLM_Prompt.py`) - Masked Language Modeling with prompts
4. **HAN + RAG** (`RAG_HAN_v4.ipynb`) - Hierarchical Attention Network with RAG (Production)

## 📁 Files

```
train/
├── train-baseline-phobert.py    # Experiment 1: Baseline PhoBERT
├── train-author-embedding.py    # Experiment 2: PhoBERT + Author Embedding
├── train-MLM_Prompt.py          # Experiment 3: Prompt-based MLM
├── RAG_HAN_v3.ipynb              # Experiment 4: HAN + RAG (v3)
├── RAG_HAN_v4.ipynb              # Experiment 4: HAN + RAG (v4 - Production)
└── RAG_HAN_v4_1.ipynb            # Experiment 4: HAN + RAG (v4.1)
```

## 🔬 Experiments Overview

### Experiment 1: Baseline PhoBERT (`train-baseline-phobert.py`)

**Purpose:** Simple baseline with PhoBERT sequence classification

**Architecture:**
- **Model**: `RobertaForSequenceClassification`
- **Input**: Text only (title + content)
- **Output**: Binary classification (REAL/FAKE)

**Hyperparameters:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 5
- Max length: 256 tokens
- Optimizer: AdamW
- Loss: CrossEntropyLoss

**Results:** Baseline performance for comparison with other models

---

### Experiment 2: PhoBERT + Author Embedding (`train-author-embedding.py`)

**Purpose:** Leverage author information to improve accuracy

**Architecture:**
- **Backbone**: PhoBERT-base-v2
- **Author Embedding**: Embedding layer for each author
- **Adaptive Gating**: Automatically learn when to trust author, when to use text only
- **Dual Branch**: 
  - Text-only branch (for unknown authors)
  - Combined branch (text + author embedding)

**Features:**
- Author encoding with LabelEncoder
- Gating mechanism to adjust author importance
- Weighted Focal Loss with label smoothing
- Mixed precision training (FP16)

**Hyperparameters:**
- Learning rate: 2e-5 (different rates for each component)
- Batch size: 16
- Epochs: 8
- Author embedding dim: 64
- Dropout: 0.3
- Focal loss: alpha=0.7, gamma=2

**Results:** Significant improvement when author information available

---

### Experiment 3: Prompt-based MLM (`train-MLM_Prompt.py`)

**Purpose:** Fine-tune PhoBERT with Masked Language Modeling and prompt engineering

**Architecture:**
- **Model**: `AutoModelForMaskedLM` (PhoBERT MLM)
- **Prompt Format**: `"Bài viết này là <mask> . Tiêu_đề : {title} . Nội_dung : {content}"`
- **Verbalizer**: 
  - Label 0 (REAL) → token "thật"
  - Label 1 (FAKE) → token "giả"
- **Training**: Predict token at `<mask>` position

**Features:**
- Vietnamese text normalizer (no vinorm needed)
- Teencode handling
- Word segmentation with underthesea
- Class-weighted loss
- Gradient accumulation

**Hyperparameters:**
- Learning rate: 2e-5
- Batch size: 16
- Gradient accumulation: 2 steps
- Epochs: 4
- Max length: 256 tokens
- Warmup: 10% of total steps

**Results:** Better leverage of pre-trained knowledge with prompts

---

### Experiment 4: HAN + RAG (`RAG_HAN_v4.ipynb`) ⭐ **PRODUCTION**

**Purpose:** Hierarchical Attention Network with RAG verification (model used in production)

**Architecture:**
- **HAN Model**: 
  - Chunk content into segments
  - RAG-based chunk selection (top-k chunks based on title similarity)
  - Hierarchical attention (chunk-level → document-level)
- **RAG Integration**:
  - Vector search in news corpus
  - Adaptive similarity threshold (0.5-0.7 for search, 0.6-0.85 for verification)
  - Confidence adjustment based on matching articles

**Features:**
- Text normalization same as training
- Semantic chunk retriever with SentenceTransformer
- ONNX export for production
- Cache mechanism

**Hyperparameters:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 5-10
- Max length: 256 tokens
- Chunk size: 400 chars
- Top-k chunks: 5

**Results:** Best performance with RAG verification, deployed in production

**Versions:**
- `RAG_HAN_v3.ipynb`: Initial HAN + RAG implementation
- `RAG_HAN_v4.ipynb`: Production version with adaptive thresholds
- `RAG_HAN_v4_1.ipynb`: Refined version with improvements

---

## 📊 Experiment Comparison

| Experiment | Model | Input Features | Complexity | Performance | Use Case |
|------------|-------|----------------|------------|-------------|----------|
| 1. Baseline | PhoBERT SC | Text only | Low | Baseline | Quick test |
| 2. Author Embed | PhoBERT + Author | Text + Author | Medium | Good | When author info available |
| 3. Prompt MLM | PhoBERT MLM | Text + Prompt | Medium | Good | Leverage pre-trained knowledge |
| 4. HAN + RAG | HAN + RAG | Text + Chunks | High | **Best** | **Production** |

## 🚀 Training Pipeline (Common for all experiments)

### 1. Data Preparation

**Input:**
- Dataset from `crawl/` folder or `dataset/` folder
- Format: CSV with columns `title`, `content` (or `text`), `label`
- Optional: `author_id` (for Experiment 2)

**Preprocessing:**
- Text normalization (Vietnamese)
- Word segmentation with underthesea
- Chunking content into segments (for HAN)
- Train/val/test split (stratified)

### 2. Training Process

**Common steps:**
1. Load and preprocess data
2. Initialize model and tokenizer
3. Create DataLoaders
4. Setup optimizer and scheduler
5. Train with validation
6. Evaluate on test set
7. Export model (ONNX for production)

### 3. Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision/Recall**: Per-class metrics
- **F1-score**: Weighted F1
- **Confusion Matrix**: Visual representation
- **ROC-AUC**: (Optional) Area under curve

## 📝 Usage

### Setup Environment

```bash
# Core dependencies
pip install torch transformers sentence-transformers
pip install underthesea  # Vietnamese NLP
pip install onnx onnxruntime
pip install pandas numpy scikit-learn

# Additional for specific experiments
pip install ydata_profiling  # For data profiling (train-baseline-phobert.py)
pip install optimum[onnxruntime]  # For ONNX export (train-MLM_Prompt.py)
```

### Run Experiments

#### Experiment 1: Baseline PhoBERT

```bash
python train-baseline-phobert.py
```

**Input files:**
- `combined_train.csv` - Combined training data
- `val_clean.csv` - Validation set
- `test_clean.csv` - Test set

**Output:**
- `best_phobert_fake_news.pt` - Best model weights
- `phobert_fake_news_model/` - Saved model directory

#### Experiment 2: PhoBERT + Author Embedding

```bash
python train-author-embedding.py
```

**Input files:**
- `final_train_stratified.csv` - Training with author_id
- `final_val_stratified.csv` - Validation with author_id
- `final_test_stratified.csv` - Test with author_id

**Output:**
- `phobert_for_onnx/best_model_weights.pt` - Model weights
- `phobert_for_onnx/model_config.json` - Config
- `phobert_for_onnx/author_classes.json` - Author mappings
- `phobert_fake_news.onnx` - ONNX model

#### Experiment 3: Prompt-based MLM

```bash
python train-MLM_Prompt.py
```

**Input:**
- Merged dataset with `title`, `text`, `label` columns

**Output:**
- Trained MLM model
- Evaluation metrics

#### Experiment 4: HAN + RAG (Production)

1. Open notebook: `RAG_HAN_v4.ipynb`
2. Configure paths:
   - Dataset path
   - Model save path
   - Output path
3. Run cells in order

**Export to ONNX:**

```python
# Export HAN model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "han_rag_model.onnx",
    input_names=['chunk_input_ids', 'chunk_attention_masks'],
    output_names=['logits'],
    dynamic_axes={
        'chunk_input_ids': {0: 'batch_size'},
        'chunk_attention_masks': {0: 'batch_size'}
    }
)
```

## 🔧 Configuration

### Data Paths (Varies by experiment)

**Experiment 1:**
```python
TRAIN_CSV = "combined_train.csv"
VAL_CSV = "val_clean.csv"
TEST_CSV = "test_clean.csv"
```

**Experiment 2:**
```python
TRAIN_CSV = "final_train_stratified.csv"
VAL_CSV = "final_val_stratified.csv"
TEST_CSV = "final_test_stratified.csv"
```

**Experiment 4 (HAN):**
```python
TRAIN_CSV = "../dataset/final_dataset_for_training.csv"
VAL_CSV = "../crawl/val_data.csv"
TEST_CSV = "../crawl/test_data.csv"
```

### Model Config (Common)

```python
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
NUM_LABELS = 2
```

**HAN-specific:**
```python
CHUNK_SIZE = 400
TOP_K_CHUNKS = 5
RETRIEVER_MODEL = "keepitreal/vietnamese-sbert"
```

**Author Embedding (Exp 2):**
```python
AUTHOR_EMBED_DIM = 64
DROPOUT_RATE = 0.3
```

### Training Config (Common)

```python
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5-8  # Varies by experiment
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01-0.02
```

**Experiment-specific:**
- **Exp 2**: Different learning rates for each component
- **Exp 3**: Gradient accumulation = 2
- **Exp 4**: Chunk-based processing

## 📊 Dataset Requirements

### Format

CSV with columns:
- `title`: Video caption/title
- `content`: OCR + STT text (or just caption if not available)
- `label`: `FAKE` or `REAL`

### Size Recommendations

- **Minimum**: 1000 samples per class
- **Recommended**: 5000+ samples per class
- **Ideal**: 10000+ samples per class

### Data Balance

- Balance between FAKE and REAL
- If imbalanced, use class weights

## 🧪 Evaluation

### Metrics

```python
# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
```

### Validation

- Validate on held-out set
- Early stopping if validation loss doesn't decrease
- Save best model based on F1-score

## 🐛 Troubleshooting

### Out of Memory

**Issue:** CUDA out of memory
- **Solution:**
  - Reduce batch size
  - Reduce max_length
  - Use gradient accumulation

### Training not converging

**Issue:** Loss not decreasing
- **Solution:**
  - Check learning rate
  - Check data quality
  - Try different optimizers
  - Add warmup steps

### Overfitting

**Issue:** Train accuracy high but val low
- **Solution:**
  - Add dropout
  - Increase weight decay
  - Add more data
  - Early stopping

## 📈 Best Practices

1. **Data Quality**: Clean and validate data thoroughly
2. **Cross-validation**: Use k-fold if dataset is small
3. **Hyperparameter tuning**: Grid search or random search
4. **Model checkpointing**: Save model each epoch
5. **Logging**: Log metrics and losses
6. **Reproducibility**: Set random seeds

## 🔒 Model Security

- **Model validation**: Test model on edge cases
- **Bias checking**: Check bias on different groups
- **Adversarial testing**: Test with adversarial examples

## 🔮 Future Improvements

- [ ] Multi-task learning
- [ ] Transfer learning from other models
- [ ] Ensemble methods
- [ ] Hyperparameter optimization with Optuna
- [ ] Model distillation
- [ ] Quantization for mobile deployment

## 📚 References

### Papers & Models

- **HAN**: [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
- **PhoBERT**: [PhoBERT: Pre-trained language models for Vietnamese](https://arxiv.org/abs/2003.00744)
- **Prompt Learning**: [GPT-3 Paper](https://arxiv.org/abs/2005.14165) (inspiration)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

### Technical Docs

- **ONNX Export**: [PyTorch to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- **Transformers**: [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- **Sentence Transformers**: [Sentence-BERT](https://www.sbert.net/)

### Datasets

- **[Vietnamese Fake News Detection](https://github.com/hiepnguyenduc2005/Vietnamese-Fake-News-Detection)**: Dataset from ReINTEL with nearly 10,000 labeled examples. This dataset is primarily used for training baseline models and experiments.
- **[VFND Vietnamese Fake News Datasets](https://github.com/WhySchools/VFND-vietnamese-fake-news-datasets)**: Collection of Vietnamese articles and Facebook posts classified (228-254 articles), including both Article Contents and Social Contents. This dataset is used to supplement and diversify training data.

## 📄 License

MIT License
