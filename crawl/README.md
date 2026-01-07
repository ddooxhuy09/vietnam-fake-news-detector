# Data Crawling Scripts

Scripts to crawl and process data from TikTok and other sources to build datasets for model training.

## 📋 Overview

This directory contains scripts and notebooks to:
- Crawl TikTok videos by channel/keyword
- Extract text from videos (STT)
- Clean and merge datasets
- Prepare data for training

## 📁 Directory Structure

```
crawl/
├── crawl_video.py              # Basic TikTok video crawler
├── crawl_video_by_channel.py   # Crawl by channel
├── create_fake_news.py         # Create fake news dataset
├── clean_data_tiktok.ipynb      # Clean TikTok data
├── merge_dataset_fb,pp.ipynb    # Merge datasets from Facebook, etc.
├── stt_fake_2.ipynb            # STT for fake videos
├── stt_real.ipynb              # STT for real videos
├── keyword_fake.txt            # Keywords to find fake news
├── list_channel_real.txt        # List of trusted channels
├── fake_all.csv                # Fake news dataset
└── tiktok_videos_all_keywords_real.csv  # Real news dataset
```

## 🚀 Usage

### 1. Crawl TikTok Videos

#### Basic Crawl (`crawl_video.py`)

```bash
python crawl_video.py
```

**Functions:**
- Read URLs from CSV file
- Download videos from TikTok
- Transcribe audio with Whisper
- Save results to CSV

**Input:** `fake2_1.csv` (contains URLs)
**Output:** `output2_1.csv` (contains URLs + transcribed text)

#### Crawl by Channel (`crawl_video_by_channel.py`)

```bash
python crawl_video_by_channel.py
```

**Functions:**
- Crawl videos from TikTok channels
- Filter by keywords
- Extract metadata and transcript

### 2. Data Processing Notebooks

#### Clean Data (`clean_data_tiktok.ipynb`)

**Functions:**
- Remove duplicates
- Clean text (remove special chars, normalize)
- Filter invalid entries
- Export cleaned dataset

#### Merge Datasets (`merge_dataset_fb,pp.ipynb`)

**Functions:**
- Merge datasets from multiple sources (Facebook, TikTok, etc.)
- Standardize format
- Balance classes (fake/real)

#### STT Processing (`stt_fake_2.ipynb`, `stt_real.ipynb`)

**Functions:**
- Batch process videos to extract STT
- Handle errors and retries
- Save progress to resume

#### Create Fake News (`create_fake_news.py`)

**Functions:**
- Generate fake news using LLM
- Create synthetic dataset for training
- Export to CSV format

## 📝 Script Details

### crawl_video.py

**Dependencies:**
- `yt-dlp`: Download video
- `whisper`: Speech-to-Text
- `torch`: PyTorch for Whisper

**Functions:**
- `download_and_transcribe()`: Download and transcribe video
- `read_urls_from_csv()`: Read URLs from CSV
- `save_result_to_csv()`: Save results
- `process_videos_from_csv()`: Main processing function

**Usage:**
```python
# Edit input/output files in main()
input_csv = "fake2_1.csv"
output_csv = "output2_1.csv"

python crawl_video.py
```

### crawl_video_by_channel.py

**Functions:**
- Crawl videos from TikTok channels
- Filter by keywords from `keyword_fake.txt`
- Extract metadata (caption, author, views, etc.)
- Save to CSV

**Usage:**
```bash
# Configure channels and keywords in script
python crawl_video_by_channel.py
```

### create_fake_news.py

**Functions:**
- Generate fake news articles using LLM
- Create synthetic dataset for training
- Export to CSV format

**Usage:**
```bash
python create_fake_news.py
```

## 📊 Data Format

### Input CSV Format

```csv
url,text
https://tiktok.com/@user/video/123,
https://tiktok.com/@user/video/456,
```

### Output CSV Format

```csv
url,text
https://tiktok.com/@user/video/123,Transcribed text from video...
https://tiktok.com/@user/video/456,Another transcribed text...
```

### Dataset Format (for training)

```csv
title,content,label
Video caption,OCR text + STT text,FAKE
Another caption,More text content,REAL
```

## 🔧 Configuration

### Keywords (`keyword_fake.txt`)

List of keywords to find fake news:
```
tặng tiền
phát tiền
nhận tiền ngay
virus mới
bệnh lạ
...
```

### Real Channels (`list_channel_real.txt`)

List of trusted channels:
```
@vnexpress
@vtv24
@vovtv
@60giay
...
```

## 🧪 Testing

### Test crawl single video

```python
from crawl_video import download_and_transcribe

video_url = "https://tiktok.com/@user/video/123"
text = download_and_transcribe(video_url, "test123")
print(text)
```

### Test with sample data

1. Create `test_urls.csv` with a few URLs
2. Run script
3. Check output

## 🐛 Troubleshooting

### Download failed

**Issue:** `yt-dlp` cannot download
- **Solution:** 
  - Update yt-dlp: `pip install --upgrade yt-dlp`
  - Check TikTok URL format
  - May need VPN if blocked

### STT failed

**Issue:** Whisper cannot transcribe
- **Solution:**
  - Check audio file exists
  - Check FFmpeg is installed
  - Try smaller model (base, small)

### Memory issues

**Issue:** Out of memory when processing many videos
- **Solution:**
  - Process videos one at a time
  - Cleanup files after each video
  - Use batch processing with limit

### Rate limiting

**Issue:** TikTok blocks requests
- **Solution:**
  - Add delays between requests
  - Use proxies
  - Rotate user agents

## 📈 Best Practices

1. **Incremental processing**: Save progress to resume
2. **Error handling**: Catch and log errors
3. **Rate limiting**: Don't spam requests
4. **Data validation**: Validate data before saving
5. **Backup**: Backup datasets regularly

## 🔒 Legal & Ethics

⚠️ **Important Notes:**

- Comply with TikTok Terms of Service
- Don't crawl too much to avoid rate limits
- Respect privacy and copyright
- Only use data for research/training
- Don't redistribute crawled data

## 📚 Related Files

- Training notebooks: `../train/`
- Dataset files: `*.csv` in this directory
- Keywords/channels: `*.txt` files
- Final dataset: `../dataset/final_dataset_for_training.csv`

## 🔮 Future Improvements

- [ ] Async crawling with aiohttp
- [ ] Database storage instead of CSV
- [ ] Automatic retry with exponential backoff
- [ ] Progress tracking with tqdm
- [ ] Parallel processing
- [ ] Data validation pipeline

## 📄 License

MIT License - For research/training purposes only
