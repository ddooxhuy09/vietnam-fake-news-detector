# Chrome Extension - Kiểm Tin Giả - PTIT

Chrome/Edge browser extension to detect fake news on TikTok directly on the website. Extension developed by Posts and Telecommunications Institute of Technology (PTIT).

## 📋 Overview

This extension allows users to:
- Analyze TikTok videos directly on the website
- View fake/real prediction results in popup
- Report incorrect results to improve model
- Simple interface with PTIT branding

## 🏗️ Architecture

```
┌─────────────┐
│   Popup     │  ← UI displaying results (PTIT branding)
│  (popup/)   │
└──────┬──────┘
       │
       │ chrome.runtime.sendMessage
       ▼
┌─────────────┐
│ Background  │  ← Service worker
│(background/)│
└──────┬──────┘
       │
       │ chrome.tabs.sendMessage
       ▼
┌─────────────┐
│  Content    │  ← Injected into TikTok page
│ (content/)  │
└──────┬──────┘
       │
       │ Scrape data from DOM
       ▼
   TikTok Page
```

## 📁 Directory Structure

```
extension/
├── manifest.json          # Extension manifest (v3)
├── background/
│   └── background.js      # Service worker
├── content/
│   ├── content.js         # Content script (scraping)
│   └── content.css        # Styles for injected UI
├── popup/
│   ├── popup.html         # Popup UI (PTIT branding)
│   ├── popup.js           # Popup logic
│   └── popup.css          # Popup styles (light theme)
├── icons/                 # Extension icons + PTIT logo
│   └── logo-ptit.png      # PTIT logo
├── database/              # Database schema
│   └── supabase_schema.sql
├── package.json           # Dependencies
└── node_modules/          # npm packages
```

## 🚀 Installation

### 1. Install Dependencies

```bash
npm install
```

Dependencies:
- `@huggingface/tokenizers`: Tokenizer for Vietnamese text
- `onnxruntime-web`: ONNX Runtime for browser (optional)

### 2. Load Extension into Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top right corner)
3. Click **Load unpacked**
4. Select `extension/` folder
5. Extension will appear as **"Kiểm Tin Giả - PTIT"**

### 3. Configure API URL

By default, extension connects to `http://localhost:8000`. To change:

1. Open `popup/popup.js`
2. Edit `API_BASE_URL`:
```javascript
const API_BASE_URL = 'http://your-api-url:8000/api/v1';
```

## 📝 Component Details

### manifest.json

Extension manifest version 3 with permissions:
- `activeTab`: Access current tab
- `storage`: Local storage
- `scripting`: Inject scripts
- `contextMenus`: Context menu support
- Host permissions: `https://www.tiktok.com/*`, `http://localhost:8000/*`

**Extension name:** "Kiểm Tin Giả - PTIT"
**Version:** 2.2.2

### Content Script (`content/content.js`)

**Functions:**
- Scrape data from TikTok page
- Listen for URL changes (TikTok SPA)
- Return video data when popup requests

**Data extraction methods:**
1. **SIGI_STATE** (Priority): Parse from `<script id="SIGI_STATE">`
2. **UNIVERSAL_DATA**: Parse from `__UNIVERSAL_DATA_FOR_REHYDRATION__`
3. **DOM scraping** (Fallback): Query DOM elements

**Data structure:**
```javascript
{
  video_id: "1234567890",
  video_url: "https://tiktok.com/@user/video/123",
  caption: "Video caption text...",
  author_id: "username"
}
```

### Popup (`popup/popup.html`, `popup/popup.js`)

**UI Design:**
- **Theme**: Light theme with white background, black border
- **Logo**: PTIT logo at top left
- **Name**: "Kiểm Tin Giả"
- **Subtitle**: "Phát hiện tin giả TikTok bằng AI"

**Functions:**
- UI to trigger analysis
- Call backend API
- Display results with styling

**Flow:**
1. User clicks "Phân tích video"
2. Check if on TikTok page
3. Inject content script if needed
4. Get video data from content script
5. Call `/api/v1/process-media` (OCR or STT depending on URL type)
6. Call `/api/v1/predict` (prediction)
7. Display results

**UI States:**
- Loading: Display spinner
- Success: Display prediction + confidence
  - 🟢 REAL: Green (#2e7d32)
  - 🔴 FAKE: Red (#d32f2f)
  - ⚪ UNCERTAIN: Orange (#f57c00)
- Error: Display error message

### Background Script (`background/background.js`)

**Functions:**
- Service worker (Manifest v3)
- Message routing between popup and content script
- Context menu setup
- Currently simple, can be extended for offline support

## 🎨 UI/UX

### Popup Design

**Theme:**
- Background: White (#ffffff)
- Border: Black (#1a1a1a, 2px)
- Text: Black (#1a1a1a)
- Button: Red PTIT color (#d32f2f)

**Layout:**
- Header with PTIT logo (48x48px) and title
- Analyze button (full width)
- Result area with confidence bar
- Report button (shown when result available)
- Footer with PTIT credit and version

**Color Coding:**
- 🟢 REAL: Green (#2e7d32)
- 🔴 FAKE: Red (#d32f2f)
- ⚪ UNCERTAIN: Orange (#f57c00)

### Accessibility

- Keyboard navigation support
- Screen reader friendly
- High contrast colors

## 🔧 Development

### Debugging

**Content Script:**
- Open DevTools on TikTok page
- Console will display logs from content script

**Popup:**
- Right-click extension icon → "Inspect popup"
- DevTools will open for popup window

**Background:**
- Go to `chrome://extensions/`
- Click "service worker" link under extension

### Testing

1. Open TikTok page: `https://www.tiktok.com/@user/video/123`
2. Click extension icon
3. Click "Phân tích video"
4. Check console logs and network requests

## 🐛 Troubleshooting

### Extension not working

**Issue:** Content script not injecting
- **Solution:** Reload TikTok page (F5)

**Issue:** Cannot get video data
- **Solution:** TikTok may have changed DOM structure, need to update selectors

**Issue:** API connection failed
- **Solution:** 
  - Check backend server is running
  - Check CORS settings
  - Check API_BASE_URL in popup.js

### Logo not displaying

**Issue:** PTIT logo not loading
- **Solution:**
  - Check file `icons/logo-ptit.png` exists
  - Check path in `popup.html`: `../icons/logo-ptit.png`
  - Reload extension

### Scraping not accurate

TikTok frequently changes DOM structure. If scraping fails:

1. Check console logs in DevTools
2. Inspect DOM structure of TikTok page
3. Update selectors in `content.js`

## 📦 Build & Deploy

### Development
```bash
# Just load unpacked in Chrome
# No build step needed
```

### Production (if minification needed)
```bash
# Can use webpack/rollup to bundle
npm run build
```

### Publish to Chrome Web Store

1. Create ZIP file:
```bash
zip -r extension.zip . -x "node_modules/*" "*.md" ".git/*"
```

2. Upload to Chrome Web Store Developer Dashboard
3. Fill information and submit for review

## 🔒 Permissions

Extension only requests necessary permissions:
- `activeTab`: Only when user clicks extension
- `storage`: Store user preferences (future)
- `scripting`: Inject content script
- `contextMenus`: Context menu support
- Host: Only TikTok and localhost API

## 📚 API Integration

Extension communicates with backend via REST API:

### Endpoints used:
- `POST /api/v1/process-media`: Process OCR or STT (depending on URL type)
  - Video URL (`/video/`) → STT only
  - Photo URL (`/photo/`) → OCR only
- `POST /api/v1/predict`: Predict fake/real news
- `POST /api/v1/report`: Report incorrect results

See details in [backend/README.md](../backend/README.md)

## 📊 Media Processing Flow

Backend automatically detects URL type and selects processing method:

```
┌──────────────┐
│  TikTok URL  │
└──────┬───────┘
       │
       ▼
┌─────────────────┐
│ Detect URL Type │
└──────┬──────────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│Video│ │Photo│
│/video│ │/photo│
└──┬──┘ └──┬──┘
   │       │
   ▼       ▼
┌─────┐ ┌─────┐
│ STT │ │ OCR │
│(GPU)│ │(GPU)│
└─────┘ └─────┘
```

## 🔮 Future Improvements

- [ ] Offline mode with ONNX Runtime Web
- [ ] Prediction history
- [ ] Settings page
- [ ] Batch analysis
- [ ] Export results
- [ ] Dark mode toggle
- [ ] Multi-language support

## 📄 License

MIT License

## 👥 Credits

**Posts and Telecommunications Institute of Technology (PTIT)**

This extension is developed as part of a research project on fake news detection on social media.
