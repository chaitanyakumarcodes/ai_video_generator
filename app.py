import os
import re
import uuid
import threading
import warnings
import numpy as np
from datetime import datetime
from typing import List, Dict

warnings.filterwarnings("ignore")

# ── Flask
from flask import Flask, request, jsonify, send_file

# ── Image / Audio / Video 
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from gtts import gTTS
from moviepy.editor import (
    ImageClip, concatenate_videoclips, CompositeVideoClip,
    AudioFileClip, CompositeAudioClip,
)

# ── Diffusion model
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

#  CONFIG

class Config:
    OUTPUT_DIR   = "outputs/videos"
    TEMP_DIR     = "outputs/temp"
    VIDEO_WIDTH  = 608          # 9:16 portrait at half-res (fast)
    VIDEO_HEIGHT = 1080
    FPS          = 24
    IMAGE_STEPS  = 20           # lower = faster; raise to 30-50 for quality
    MUSIC_VOLUME = 0.25
    MAX_SUMMARY  = 2000
    MIN_SUMMARY  = 50
    MODEL_ID     = "SG161222/Realistic_Vision_V5.1_noVAE"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.TEMP_DIR,   exist_ok=True)

#  IN-MEMORY JOB STORE

jobs: Dict[str, dict] = {}

#  STABLE DIFFUSION — lazy-loaded once on first request

_pipe       = None
_pipe_lock  = threading.Lock()

def get_pipe():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype  = torch.float16 if device == "cuda" else torch.float32
                print(f"[model] Loading Stable Diffusion on {device} …")
                pipe = StableDiffusionPipeline.from_pretrained(
                    Config.MODEL_ID,
                    torch_dtype=dtype,
                    safety_checker=None,
                )
                pipe = pipe.to(device)
                if device == "cuda":
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                _pipe = pipe
                print("[model] Ready.")
    return _pipe


#  GENRE DETECTION

GENRE_STYLES = {
    "fantasy":   {
        "visual": "epic fantasy landscape, magical, ethereal, cinematic lighting, detailed",
        "colors": "purple and blue mystical tones",
    },
    "sci-fi":    {
        "visual": "futuristic, cyberpunk aesthetic, neon lights, high-tech, space",
        "colors": "cyan and dark blue tech tones",
    },
    "romance":   {
        "visual": "romantic, soft lighting, beautiful scenery, emotional, dreamy",
        "colors": "warm pink and gold tones",
    },
    "thriller":  {
        "visual": "dark, mysterious, dramatic shadows, suspenseful, tense",
        "colors": "dark red and black tones",
    },
    "mystery":   {
        "visual": "noir style, moody, atmospheric fog, mysterious",
        "colors": "dark blue and grey tones",
    },
    "horror":    {
        "visual": "dark, eerie, ominous atmosphere, gothic",
        "colors": "blood red and pitch black",
    },
    "adventure": {
        "visual": "dynamic action, exciting, vibrant landscapes, epic journey",
        "colors": "bright and colorful adventure tones",
    },
    "literary":  {
        "visual": "artistic, thoughtful, elegant composition, refined",
        "colors": "muted sophisticated tones",
    },
}

GENRE_KEYWORDS = {
    "fantasy":   ["magic", "wizard", "dragon", "fantasy", "kingdom", "quest", "spell"],
    "sci-fi":    ["space", "future", "robot", "ai", "alien", "technology", "cyber", "planet"],
    "romance":   ["love", "heart", "relationship", "romance", "passion", "wedding"],
    "thriller":  ["danger", "chase", "escape", "threat", "suspense", "hunt"],
    "mystery":   ["detective", "mystery", "clue", "investigation", "murder", "solve"],
    "horror":    ["horror", "terror", "fear", "ghost", "monster", "haunted", "nightmare"],
    "adventure": ["journey", "adventure", "explore", "discovery", "voyage", "expedition"],
}

def detect_genre(summary: str) -> str:
    lower = summary.lower()
    for genre, words in GENRE_KEYWORDS.items():
        if any(w in lower for w in words):
            return genre
    return "literary"

#  PROMPT BUILDER

def split_into_scenes(summary: str, num_scenes: int = 6) -> List[str]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", summary) if s.strip()]
    if len(sentences) <= num_scenes:
        return sentences
    step = len(sentences) // num_scenes
    return [
        " ".join(sentences[i * step: (i + 1) * step if i < num_scenes - 1 else len(sentences)])
        for i in range(num_scenes)
    ]

def build_prompts(summary: str, genre: str) -> List[dict]:
    style    = GENRE_STYLES.get(genre, GENRE_STYLES["literary"])
    scenes   = split_into_scenes(summary)
    negative = (
        "cartoon, 3d, blurry, deformed, ugly, bad anatomy, "
        "bad hands, distorted face, low quality, watermark"
    )
    prompts = []

    # Title card
    prompts.append({
        "type":     "title",
        "text":     "A New Story Awaits",
        "prompt":   f"elegant book cover design, {style['visual']}, {style['colors']}, professional, high quality",
        "negative": negative,
    })

    # Scene cards
    for scene in scenes:
        prompts.append({
            "type":     "scene",
            "text":     scene[:100],
            "prompt":   f"{scene}, {style['visual']}, {style['colors']}, cinematic, highly detailed, masterpiece",
            "negative": negative,
        })

    # End card
    prompts.append({
        "type":     "ending",
        "text":     "Read More…",
        "prompt":   f"inspiring book finale, {style['visual']}, {style['colors']}, hopeful, beautiful",
        "negative": negative,
    })

    return prompts


#  IMAGE GENERATION

def generate_image(prompt: str, negative: str = "") -> Image.Image:
    pipe  = get_pipe()
    w     = (Config.VIDEO_WIDTH  // 8) * 8
    h     = (Config.VIDEO_HEIGHT // 8) * 8
    return pipe(
        prompt,
        negative_prompt=negative,
        num_inference_steps=Config.IMAGE_STEPS,
        guidance_scale=7.5,
        width=w,
        height=h,
    ).images[0]

def add_overlay_text(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except Exception:
        font = ImageFont.load_default()
    bbox   = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y   = (w - tw) / 2, (h - th) / 2
    draw.text((x + 3, y + 3), text, font=font, fill=(0,   0,   0,   200))
    draw.text((x,     y    ), text, font=font, fill=(255, 255, 255, 255))
    return image


#  AUDIO

def create_narration(summary: str, out_path: str) -> List[str]:
    """Save TTS audio and return 4-word subtitle chunks."""
    text   = f"Discover a captivating story. {summary}"
    tts    = gTTS(text=text, lang="en", slow=False)
    tts.save(out_path)
    words  = text.split()
    chunks = [" ".join(words[i: i + 4]) for i in range(0, len(words), 4)]
    return chunks

def create_background_music(duration: float, out_path: str):
    sr  = 44100
    t   = np.linspace(0, duration, int(sr * duration))
    sig = sum(np.sin(2 * np.pi * f * t) * 0.25 / 4 for f in [220, 277, 330, 440])
    sig *= 1 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
    fade = int(sr * 3)
    sig[:fade]  *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    sig = (sig / np.max(np.abs(sig)) * 0.4 * 32767).astype(np.int16)
    wavfile.write(out_path, sr, sig)


#  SUBTITLE CLIP (Pillow — no ImageMagick)

def make_subtitle_clip(text: str, duration: float) -> ImageClip:
    img  = Image.new("RGBA", (Config.VIDEO_WIDTH, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x    = (Config.VIDEO_WIDTH - (bbox[2] - bbox[0])) / 2
    draw.text((x + 2, 42), text, font=font, fill=(0,   0,   0,   220))
    draw.text((x,     40), text, font=font, fill=(255, 255, 255, 255))
    return (
        ImageClip(np.array(img), ismask=False)
        .set_duration(duration)
        .set_position(("center", Config.VIDEO_HEIGHT - 180))
    )


#  VIDEO COMPOSITION

def compose_video(
    image_paths:     List[str],
    narration_path:  str,
    music_path:      str,
    output_path:     str,
    subtitle_chunks: List[str],
):
    # Duration follows narration length
    narration_audio = AudioFileClip(narration_path)
    total_duration  = narration_audio.duration + 1.0
    dur_per_image   = total_duration / len(image_paths)

    # Image clips with Ken-Burns zoom
    clips = []
    for i, path in enumerate(image_paths):
        clip = ImageClip(path, duration=dur_per_image)
        clip = clip.resize(lambda t: 1 + 0.03 * t / dur_per_image)
        if i > 0:
            clip = clip.crossfadein(0.5)
        if i < len(image_paths) - 1:
            clip = clip.crossfadeout(0.5)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")

    # Subtitle clips
    chunk_dur = total_duration / len(subtitle_chunks)
    sub_clips = [
        make_subtitle_clip(chunk, chunk_dur).set_start(i * chunk_dur)
        for i, chunk in enumerate(subtitle_chunks)
    ]

    video = CompositeVideoClip(
        [video] + sub_clips,
        size=(Config.VIDEO_WIDTH, Config.VIDEO_HEIGHT),
    )

    # Music
    music = AudioFileClip(music_path).volumex(Config.MUSIC_VOLUME)
    music = (
        music.audio_loop(duration=total_duration)
        if music.duration < total_duration
        else music.subclip(0, total_duration)
    )

    video = video.set_audio(CompositeAudioClip([narration_audio, music]))

    video.write_videofile(
        output_path,
        fps=Config.FPS,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="medium",
        logger=None,
    )


#  MAIN PIPELINE

def run_pipeline(job_id: str, summary: str, title: str):
    """Full video generation pipeline — runs in a background thread."""
    def log(msg):
        jobs[job_id]["log"].append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")

    try:
        jobs[job_id]["status"]     = "processing"
        jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

        # Working directories per job
        job_tmp = os.path.join(Config.TEMP_DIR, job_id)
        img_dir = os.path.join(job_tmp, "images")
        os.makedirs(img_dir, exist_ok=True)

        # ── Genre + prompts 
        genre   = detect_genre(summary)
        prompts = build_prompts(summary, genre)
        log(f"Genre detected: {genre.upper()}  |  Scenes: {len(prompts)}")
        jobs[job_id]["genre"] = genre

        # ── Images 
        image_paths = []
        for i, p in enumerate(prompts):
            log(f"Generating image {i + 1}/{len(prompts)} [{p['type']}]")
            img  = generate_image(p["prompt"], p.get("negative", ""))
            if p["type"] in ("title", "ending"):
                img = add_overlay_text(img, p["text"])
            path = os.path.join(img_dir, f"scene_{i:03d}.png")
            img.save(path)
            image_paths.append(path)

        # ── Audio 
        log("Creating narration …")
        narration_path = os.path.join(job_tmp, "narration.mp3")
        chunks         = create_narration(summary, narration_path)

        log("Creating background music …")
        music_path     = os.path.join(job_tmp, "music.wav")
        tmp_audio      = AudioFileClip(narration_path)
        actual_dur     = tmp_audio.duration + 1.0
        tmp_audio.close()
        create_background_music(actual_dur, music_path)

        # ── Video
        log("Composing video …")
        safe_title  = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
        output_path = os.path.join(Config.OUTPUT_DIR, f"{safe_title}_{job_id[:8]}_reel.mp4")
        compose_video(image_paths, narration_path, music_path, output_path, chunks)

        # ── Done
        jobs[job_id]["status"]     = "completed"
        jobs[job_id]["video_path"] = output_path
        jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        log("Done ✓")

    except Exception as exc:
        jobs[job_id]["status"]     = "failed"
        jobs[job_id]["error"]      = str(exc)
        jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        log(f"ERROR: {exc}")


#  FLASK ROUTES

def _new_job(title: str, summary: str) -> dict:
    return {
        "job_id":     None,
        "title":      title,
        "summary":    summary,
        "genre":      None,
        "status":     "queued",   # queued → processing → completed | failed
        "video_path": None,
        "error":      None,
        "log":        [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


@app.route("/health", methods=["GET"])
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return jsonify({
        "status": "ok",
        "device": device,
        "jobs_total": len(jobs),
    }), 200


@app.route("/generate", methods=["POST"])
def generate():
    """
    Start a video generation job.

    Request JSON:
        { "title": "My Book", "summary": "Once upon a time…" }

    Response 202:
        { "job_id": "…", "status": "queued", "message": "…" }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    title   = (data.get("title")   or "").strip()
    summary = (data.get("summary") or "").strip()

    if not title:
        return jsonify({"error": "'title' is required"}), 400
    if not summary:
        return jsonify({"error": "'summary' is required"}), 400
    if len(summary) < Config.MIN_SUMMARY:
        return jsonify({"error": f"'summary' must be at least {Config.MIN_SUMMARY} characters"}), 400
    if len(summary) > Config.MAX_SUMMARY:
        return jsonify({"error": f"'summary' must be under {Config.MAX_SUMMARY} characters"}), 400

    job_id        = str(uuid.uuid4())
    job           = _new_job(title, summary)
    job["job_id"] = job_id
    jobs[job_id]  = job

    thread = threading.Thread(target=run_pipeline, args=(job_id, summary, title), daemon=True)
    thread.start()

    return jsonify({
        "job_id":   job_id,
        "status":   "queued",
        "message":  f"Job started. Poll GET /status/{job_id} for updates.",
    }), 202


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    """
    Poll job status.

    Response:
        {
            "job_id": "…",
            "status": "processing",   # queued|processing|completed|failed
            "genre":  "fantasy",
            "log":    ["[12:00:01] Generating image 1/7 …", …],
            "error":  null
        }
    """
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404

    # Don't expose full summary in status
    return jsonify({k: v for k, v in job.items() if k != "summary"}), 200


@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    """Download the generated .mp4 once status is 'completed'."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404
    if job["status"] != "completed":
        return jsonify({"error": f"Not ready. Status: {job['status']}"}), 400

    path = job["video_path"]
    if not path or not os.path.exists(path):
        return jsonify({"error": "Video file missing on disk"}), 500

    return send_file(
        path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=os.path.basename(path),
    )


@app.route("/jobs", methods=["GET"])
def list_jobs():
    """
    List all jobs.
    Optional filter: GET /jobs?status=completed
    """
    filter_status = request.args.get("status")
    result = [
        {k: v for k, v in j.items() if k not in ("summary", "log")}
        for j in jobs.values()
        if not filter_status or j["status"] == filter_status
    ]
    return jsonify({"total": len(result), "jobs": result}), 200


@app.route("/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id):
    """Delete a job record (and its video file if present)."""
    job = jobs.pop(job_id, None)
    if not job:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404

    path = job.get("video_path")
    if path and os.path.exists(path):
        os.remove(path)

    return jsonify({"message": f"Job '{job_id}' deleted"}), 200


if __name__ == "__main__":
    print("=" * 60)
    print("  Book Video Generator API")
    print("  http://0.0.0.0:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)