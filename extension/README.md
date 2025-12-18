# Chrome Extension - Kiểm Tin Giả - PTIT

Extension trình duyệt Chrome/Edge để phát hiện tin giả trên TikTok trực tiếp trên trang web. Extension được phát triển bởi Học viện Công nghệ Bưu chính Viễn thông (PTIT).

## 📋 Tổng quan

Extension này cho phép người dùng:
- Phân tích video TikTok ngay trên trang web
- Xem kết quả dự đoán tin giả/thật trong popup
- Báo cáo kết quả sai để cải thiện model
- Giao diện đơn giản với branding PTIT

## 🏗️ Kiến trúc

```
┌─────────────┐
│   Popup     │  ← UI hiển thị kết quả (PTIT branding)
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
│  Content    │  ← Inject vào TikTok page
│ (content/)  │
└──────┬──────┘
       │
       │ Scrape data từ DOM
       ▼
   TikTok Page
```

## 📁 Cấu trúc thư mục

```
extension/
├── manifest.json          # Extension manifest (v3)
├── background/
│   └── background.js      # Service worker
├── content/
│   ├── content.js         # Content script (scraping)
│   └── content.css        # Styles cho injected UI
├── popup/
│   ├── popup.html         # Popup UI (PTIT branding)
│   ├── popup.js           # Popup logic
│   └── popup.css          # Popup styles (light theme)
├── icons/                 # Extension icons + PTIT logo
│   └── logo-ptit.png      # PTIT logo
├── database/              # Database schema
│   └── supabase_schema.sql
├── model-loader.js        # ONNX model loader (optional)
├── tokenizer.js           # Tokenizer (optional)
└── package.json           # Dependencies
```

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
npm install
```

Dependencies:
- `@huggingface/tokenizers`: Tokenizer cho Vietnamese text
- `onnxruntime-web`: ONNX Runtime cho browser (optional)

### 2. Load Extension vào Chrome

1. Mở Chrome và vào `chrome://extensions/`
2. Bật **Developer mode** (góc trên bên phải)
3. Click **Load unpacked**
4. Chọn thư mục `extension/`
5. Extension sẽ xuất hiện với tên **"Kiểm Tin Giả - PTIT"**

### 3. Cấu hình API URL

Mặc định extension kết nối đến `http://localhost:8000`. Để thay đổi:

1. Mở `popup/popup.js`
2. Sửa `API_BASE_URL`:
```javascript
const API_BASE_URL = 'http://your-api-url:8000/api/v1';
```

## 📝 Chi tiết các thành phần

### manifest.json

Extension manifest version 3 với các permissions:
- `activeTab`: Truy cập tab hiện tại
- `storage`: Lưu trữ local
- `scripting`: Inject scripts
- Host permissions: `https://www.tiktok.com/*`, `http://localhost:8000/*`

**Extension name:** "Kiểm Tin Giả - PTIT"

### Content Script (`content/content.js`)

**Chức năng:**
- Scrape dữ liệu từ TikTok page
- Lắng nghe URL changes (TikTok SPA)
- Trả về video data khi popup request

**Data extraction methods:**
1. **SIGI_STATE** (Priority): Parse từ `<script id="SIGI_STATE">`
2. **UNIVERSAL_DATA**: Parse từ `__UNIVERSAL_DATA_FOR_REHYDRATION__`
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
- **Theme**: Light theme với nền trắng, viền đen
- **Logo**: PTIT logo ở góc trái trên
- **Tên**: "Kiểm Tin Giả"
- **Subtitle**: "Phát hiện tin giả TikTok bằng AI"

**Chức năng:**
- UI để trigger phân tích
- Gọi API backend
- Hiển thị kết quả với styling

**Flow:**
1. User click "Phân tích video"
2. Check nếu đang ở TikTok page
3. Inject content script nếu cần
4. Lấy video data từ content script
5. Gọi `/api/v1/process-media` (OCR hoặc STT tùy URL type)
6. Gọi `/api/v1/predict` (prediction)
7. Hiển thị kết quả

**UI States:**
- Loading: Hiển thị spinner
- Success: Hiển thị prediction + confidence
  - 🟢 REAL: Green (#2e7d32)
  - 🔴 FAKE: Red (#d32f2f)
  - ⚪ UNCERTAIN: Orange (#f57c00)
- Error: Hiển thị error message

### Background Script (`background/background.js`)

**Chức năng:**
- Service worker (Manifest v3)
- Message routing giữa popup và content script
- Hiện tại đơn giản, có thể mở rộng cho offline support

## 🎨 UI/UX

### Popup Design

**Theme:**
- Background: White (#ffffff)
- Border: Black (#1a1a1a, 2px)
- Text: Black (#1a1a1a)
- Button: Red PTIT color (#d32f2f)

**Layout:**
- Header với PTIT logo (48x48px) và title
- Analyze button (full width)
- Result area với confidence bar
- Report button (hiện khi có result)
- Footer với PTIT credit và version

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
- Mở DevTools trên TikTok page
- Console sẽ hiển thị logs từ content script

**Popup:**
- Right-click extension icon → "Inspect popup"
- DevTools sẽ mở cho popup window

**Background:**
- Vào `chrome://extensions/`
- Click "service worker" link dưới extension

### Testing

1. Mở TikTok page: `https://www.tiktok.com/@user/video/123`
2. Click extension icon
3. Click "Phân tích video"
4. Kiểm tra console logs và network requests

## 🐛 Troubleshooting

### Extension không hoạt động

**Vấn đề:** Content script không inject
- **Giải pháp:** Reload TikTok page (F5)

**Vấn đề:** Không lấy được video data
- **Giải pháp:** TikTok có thể đã thay đổi DOM structure, cần update selectors

**Vấn đề:** API connection failed
- **Giải pháp:** 
  - Kiểm tra backend server đang chạy
  - Kiểm tra CORS settings
  - Kiểm tra API_BASE_URL trong popup.js

### Logo không hiển thị

**Vấn đề:** PTIT logo không load
- **Giải pháp:**
  - Kiểm tra file `icons/logo-ptit.png` tồn tại
  - Kiểm tra path trong `popup.html`: `../icons/logo-ptit.png`
  - Reload extension

### Scraping không chính xác

TikTok thường xuyên thay đổi DOM structure. Nếu scraping fail:

1. Check console logs trong DevTools
2. Inspect DOM structure của TikTok page
3. Update selectors trong `content.js`

## 📦 Build & Deploy

### Development
```bash
# Chỉ cần load unpacked trong Chrome
# Không cần build step
```

### Production (nếu cần minify)
```bash
# Có thể dùng webpack/rollup để bundle
npm run build
```

### Publish to Chrome Web Store

1. Tạo ZIP file:
```bash
zip -r extension.zip . -x "node_modules/*" "*.md" ".git/*"
```

2. Upload lên Chrome Web Store Developer Dashboard
3. Điền thông tin và submit for review

## 🔒 Permissions

Extension chỉ request permissions cần thiết:
- `activeTab`: Chỉ khi user click extension
- `storage`: Lưu user preferences (future)
- `scripting`: Inject content script
- Host: Chỉ TikTok và localhost API

## 📚 API Integration

Extension giao tiếp với backend qua REST API:

### Endpoints sử dụng:
- `POST /api/v1/process-media`: Xử lý OCR hoặc STT (tùy URL type)
  - Video URL (`/video/`) → STT only
  - Photo URL (`/photo/`) → OCR only
- `POST /api/v1/predict`: Dự đoán tin giả/thật
- `POST /api/v1/report`: Báo cáo kết quả sai

Xem chi tiết trong [backend/README.md](../backend/README.md)

## 📊 Media Processing Flow

Backend tự động detect URL type và chọn phương pháp xử lý:

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

- [ ] Offline mode với ONNX Runtime Web
- [ ] History của predictions
- [ ] Settings page
- [ ] Batch analysis
- [ ] Export results
- [ ] Dark mode toggle
- [ ] Multi-language support

## 📄 License

MIT License

## 👥 Credits

**Học viện Công nghệ Bưu chính Viễn thông (PTIT)**

Extension này được phát triển như một phần của dự án nghiên cứu về phát hiện tin giả trên mạng xã hội.
