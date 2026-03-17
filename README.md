# Book Video Generator

A Flask API that turns a book title and summary into a cinematic Instagram Video — complete with AI-generated scene images, text-to-speech narration, subtitles, and ambient background music.

---

## 🗂️ Project Structure

```
book-video-generator/
├── app.py                   # Full pipeline + Flask API
├── ai_video_generator.ipynb # Python notebook
└── README.md
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/book-video-generator.git
cd book-video-generator
```

### 2. Install system dependencies
```bash
# FFmpeg (required for video encoding)
sudo apt-get install -y ffmpeg

# ImageMagick (optional — not required, subtitles use Pillow instead)
sudo apt-get install -y imagemagick
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

>  **GPU strongly recommended.** On CPU, each image takes ~2–5 minutes. On a T4 GPU (Google Colab), ~15 seconds per image.

### 4. Run the API
```bash
python app.py
```

API will be live at `http://localhost:5000`

---

## API Reference

### `GET /health`
Check if the API is running.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "jobs_total": 3
}
```

---

### `POST /generate`
Submit a new video generation job.

**Request body:**
```json
{
  "title": "The Forgotten Magic",
  "summary": "In a world where magic has been forgotten, a young orphan named Elena..."
}
```

**Constraints:**
- `title` — required, non-empty string
- `summary` — required, 50–2000 characters

**Response `202`:**
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "queued",
  "message": "Job started. Poll GET /status/<job_id> for updates."
}
```

---

### `GET /status/<job_id>`
Poll job progress.

**Response:**
```json
{
  "job_id": "f47ac10b-...",
  "title": "The Forgotten Magic",
  "genre": "fantasy",
  "status": "processing",
  "log": [
    "[12:00:01] Genre detected: FANTASY  |  Scenes: 7",
    "[12:00:02] Generating image 1/7 [title]",
    "[12:01:45] Generating image 2/7 [scene]"
  ],
  "video_path": null,
  "error": null,
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:01:45"
}
```

**Status values:** `queued` → `processing` → `completed` | `failed`

---

### `GET /download/<job_id>`
Download the finished `.mp4` once status is `completed`.

Returns the video file as `video/mp4`.

---

### `GET /jobs`
List all jobs. Optional filter: `?status=completed`

**Response:**
```json
{
  "total": 2,
  "jobs": [...]
}
```

---

### `DELETE /jobs/<job_id>`
Delete a job record and its video file from disk.

---

## 🧪 Testing

```bash
python test_client.py
```

This will:
1. Hit `/health`
2. Submit a sample job
3. Poll `/status` every 10 seconds
4. Download the video once complete

---

## ⚙️ Configuration

All settings are in the `Config` class inside `app.py`:

| Setting | Default | Description |
|---|---|---|
| `VIDEO_WIDTH` | `608` | Output width (px) |
| `VIDEO_HEIGHT` | `1080` | Output height (px) |
| `FPS` | `24` | Frames per second |
| `IMAGE_STEPS` | `20` | SD inference steps (higher = better quality, slower) |
| `MUSIC_VOLUME` | `0.25` | Background music volume (0.0–1.0) |
| `MODEL_ID` | `SG161222/Realistic_Vision_V5.1_noVAE` | HuggingFace model |

### Reels resolution
Change in `Config`:
```python
VIDEO_WIDTH  = 1080
VIDEO_HEIGHT = 1920
```

---

## 🎭 Supported Genres

Auto-detected from keywords in the summary:

| Genre | Visual Style |
|---|---|
| Fantasy | Epic landscapes, magical, ethereal |
| Sci-Fi | Futuristic, cyberpunk, neon lights |
| Romance | Soft lighting, dreamy, warm tones |
| Thriller | Dark, dramatic shadows, suspenseful |
| Mystery | Noir, moody, atmospheric fog |
| Horror | Dark, eerie, gothic |
| Adventure | Vibrant, dynamic, epic journeys |
| Literary | Artistic, elegant, sophisticated |

---

## 📋 Requirements

- Python 3.10+
- CUDA GPU recommended (NVIDIA T4 or better)
- ~5 GB disk space for model weights (auto-downloaded on first run)
- FFmpeg installed on the system

---

## 📄 License

MIT License — free to use, modify, and distribute.
