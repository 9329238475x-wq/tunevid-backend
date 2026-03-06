"""
TuneVid.com — FastAPI Application Factory
Master entry point that mounts all routers and middleware.

Backend Architecture:
  - auth.py       → JWT verification, user sync
  - middleware.py  → Usage limits, abuse detection
  - payments.py   → Razorpay integration
  - marketing.py  → Referrals, auto-promo
  - dashboard.py   → User profile & stats
  - main.py       → App factory + all tool endpoints (preserved)
"""
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import traceback
import uuid
from pathlib import Path
from typing import Callable, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Depends, Request
from typing import Iterable
import librosa
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import uvicorn

# ── Import modular routers ────────────────────────────────────────────────────
from payments import router as payments_router
from marketing import router as marketing_router, inject_auto_promo, qualify_referral_on_upload
from dashboard import router as dashboard_router
from middleware import (
    UsageLimitChecker,
    check_youtube_upload_limit,
    increment_youtube_upload_count,
    check_file_size,
    detect_abuse,
)
from auth import get_current_user, register_device
from database import init_db, close_db, get_db
from models import User


# ── App Lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB. Shutdown: close connections."""
    try:
        await init_db()
        logging.info("✅ Database initialized")
    except Exception as e:
        logging.warning(f"⚠️ Database init skipped (not connected): {e}")
    yield
    try:
        await close_db()
    except Exception:
        pass


app = FastAPI(
    title="TuneVid API",
    version="4.0.0",
    description="Production SaaS API for TuneVid.com — AI Audio Studio & YouTube Uploader",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tunevid-final-di29.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Modular Routers ─────────────────────────────────────────────────────
app.include_router(payments_router)
app.include_router(marketing_router)
app.include_router(dashboard_router)

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
VIDEOS_DIR = BASE_DIR / "videos"
STATUS_DIR = BASE_DIR / "status"
TOOLS_DIR = BASE_DIR / "tools"
DOWNLOADS_DIR = BASE_DIR / "downloads"

for d in (UPLOAD_DIR, VIDEOS_DIR, STATUS_DIR, TOOLS_DIR, DOWNLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Real-time progress tracking (task_id -> progress percentage)
progress_store = {}

app.mount("/static", StaticFiles(directory=DOWNLOADS_DIR), name="static")


# ── Status helpers ────────────────────────────────────────────────────────────
def _write_status(task_id: str, payload: dict) -> None:
    (STATUS_DIR / f"{task_id}.json").write_text(json.dumps(payload))


def _read_status(task_id: str) -> dict:
    p = STATUS_DIR / f"{task_id}.json"
    if not p.exists():
        return {"step": 1, "message": "Starting…", "progress": 0}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"step": 1, "message": "Starting…", "progress": 0}


def _cancel_file(task_id: str) -> Path:
    return STATUS_DIR / f"{task_id}.cancel"


def _cleanup_paths(*paths: Path) -> None:
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        except Exception:
            pass


def cleanup_files(paths: list) -> None:
    """Best-effort cleanup for files/folders without crashing request lifecycle."""
    for raw_path in paths:
        try:
            p = Path(raw_path)
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                os.remove(p)
        except Exception as e:
            logging.warning(f"Cleanup skipped for {raw_path}: {e}")


def _run_with_progress(cmd: list, task_id: str, step_name: str) -> subprocess.CompletedProcess:
    """Run subprocess with real-time progress tracking."""
    _write_status(task_id, {"step": 2, "message": f"{step_name}...", "progress": 0})

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    stderr_output = []
    stdout_output = []
    last_progress = 0

    def read_stderr():
        nonlocal last_progress
        for line in iter(process.stderr.readline, ''):
            if not line:
                break
            stderr_output.append(line)
            percent_match = re.search(r'(\d+)%', line)
            if percent_match:
                progress = min(int(percent_match.group(1)), 100)
                if progress > last_progress:
                    last_progress = progress
                    _write_status(task_id, {
                        "step": 2,
                        "message": f"{step_name}: {progress}%",
                        "progress": progress
                    })

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            stdout_output.append(line)

    process.wait()
    stderr_thread.join(timeout=1)

    _write_status(task_id, {"step": 3, "message": "Complete!", "progress": 100})

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, cmd, ''.join(stdout_output), ''.join(stderr_output)
        )

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=process.returncode,
        stdout=''.join(stdout_output),
        stderr=''.join(stderr_output)
    )


# ── FFmpeg ────────────────────────────────────────────────────────────────────
def _get_image_size(image: Path) -> tuple:
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0", str(image)],
            capture_output=True, text=True, timeout=10,
        )
        parts = p.stdout.strip().split("x")
        w, h = int(parts[0]), int(parts[1])
        w = w if w % 2 == 0 else w - 1
        h = h if h % 2 == 0 else h - 1
        return w, h
    except Exception:
        return 1280, 720


def _run_ffmpeg(image: Path, audio: Path, output: Path, task_id: str) -> None:
    _write_status(task_id, {"step": 1, "message": "Generating video…", "progress": 10})
    width, height = _get_image_size(image)

    duration = 1.0
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
            capture_output=True, text=True, timeout=10,
        )
        duration = float(p.stdout.strip())
    except Exception:
        pass

    cmd = [
        "ffmpeg", "-progress", "pipe:1", "-nostats", "-y",
        "-loop", "1",
        "-i", str(image),
        "-i", str(audio),
        "-vf", f"scale={width}:{height},format=yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage",
        "-crf", "28", "-r", "1",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        "-shortest", "-movflags", "+faststart",
        str(output),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if proc.stdout:
        for line in proc.stdout:
            if "out_time_ms=" in line:
                try:
                    ms = int(line.split("=")[1])
                    pc = min(int((ms / (duration * 1_000_000)) * 100), 99)
                    _write_status(task_id, {
                        "step": 1,
                        "message": f"Generating video… {pc}%",
                        "progress": 10 + int(pc * 0.6),
                    })
                except Exception:
                    pass
    proc.wait()


# ── YouTube upload ────────────────────────────────────────────────────────────
def _upload_to_youtube(
    video: Path,
    title: str,
    desc: str,
    privacy: str,
    token: str,
    task_id: str,
    refresh_token: str = "",
    tags: list[str] | None = None,
    category_id: str = "10",
    made_for_kids: bool = False,
) -> str:
    _write_status(task_id, {"step": 2, "message": "Uploading to YouTube…", "progress": 75})

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")

    creds = Credentials(
        token=token,
        refresh_token=refresh_token or None,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id or None,
        client_secret=client_secret or None,
    )
    yt = build("youtube", "v3", credentials=creds)
    snippet = {"title": title, "description": desc, "categoryId": category_id}
    if tags:
        snippet["tags"] = tags
    body = {
        "snippet": snippet,
        "status": {"privacyStatus": privacy, "madeForKids": made_for_kids},
    }
    media = MediaFileUpload(str(video), mimetype="video/mp4", resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    resp = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            _write_status(task_id, {
                "step": 2,
                "message": f"Uploading to YouTube… {pct}%",
                "progress": 75 + int(pct * 0.24),
            })

    return f"https://www.youtube.com/watch?v={resp['id']}"


# ── Background job ────────────────────────────────────────────────────────────
def _process_job(
    task_id: str,
    audio: Path,
    image: Path,
    output: Path,
    title: str,
    desc: str,
    privacy: str,
    token: str,
    refresh_token: str = "",
    tags: list[str] | None = None,
    category_id: str = "10",
    made_for_kids: bool = False,
    cleanup_on_finish: list | None = None,
) -> None:
    try:
        _run_ffmpeg(image, audio, output, task_id)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("FFmpeg failed — output video is empty or missing.")

        if _cancel_file(task_id).exists():
            _write_status(task_id, {"step": 3, "message": "Cancelled.", "progress": 100})
            return

        url = _upload_to_youtube(
            output, title, desc, privacy, token, task_id,
            refresh_token, tags=tags, category_id=category_id, made_for_kids=made_for_kids,
        )

        _write_status(task_id, {"step": 3, "message": "Done! Video is live 🎉", "progress": 100, "youtube_url": url})

    except Exception as e:
        err_detail = traceback.format_exc()
        logging.error(f"[{task_id}] FAILED:\n{err_detail}")
        _write_status(task_id, {"step": 3, "message": f"Error: {e}", "progress": 100})
    finally:
        # Self-cleanup when running inside a thread (batch uploads).
        # When running via BackgroundTasks, cleanup_on_finish is None —
        # a separate cleanup task handles it instead.
        if cleanup_on_finish:
            cleanup_files(cleanup_on_finish)


# ── Shared upload handler ─────────────────────────────────────────────────────
async def _handle_upload(
    audio_file: UploadFile,
    image_file: UploadFile,
    background_tasks: Optional[BackgroundTasks],
    title: str,
    desc: str,
    privacy: str,
    token: str,
    refresh_token: str = "",
    tags: list[str] | None = None,
    category_id: str = "10",
    made_for_kids: bool = False,
    plan_type: str = "free",
) -> dict:
    task_id = uuid.uuid4().hex
    audio_ext = Path(audio_file.filename or "a.mp3").suffix or ".mp3"
    image_ext = Path(image_file.filename or "i.jpg").suffix or ".jpg"

    a_p = UPLOAD_DIR / f"{task_id}_a{audio_ext}"
    i_p = UPLOAD_DIR / f"{task_id}_i{image_ext}"
    o_p = VIDEOS_DIR / f"{task_id}_o.mp4"

    _write_status(task_id, {"step": 1, "message": "Files received. Starting…", "progress": 5})

    with a_p.open("wb") as f:
        f.write(await audio_file.read())
    with i_p.open("wb") as f:
        f.write(await image_file.read())

    # Auto-promo injection for free users
    desc = inject_auto_promo(desc, plan_type)

    # Common cleanup targets: input files, output video, Demucs dirs, status files
    demucs_dirs = [
        BASE_DIR / "separated" / "htdemucs" / a_p.stem,
        BASE_DIR / "separated" / "htdemucs_ft" / a_p.stem,
    ]
    status_files = [
        STATUS_DIR / f"{task_id}.json",
        STATUS_DIR / f"{task_id}.cancel",
    ]
    all_cleanup = [a_p, i_p, o_p, *demucs_dirs, *status_files]

    if background_tasks is not None:
        # BackgroundTasks path: tasks run sequentially —
        # process first, then cleanup fires after it finishes.
        background_tasks.add_task(
            _process_job,
            task_id, a_p, i_p, o_p,
            title or (audio_file.filename or "Untitled"),
            desc, privacy, token, refresh_token,
            tags, category_id, made_for_kids,
        )
        background_tasks.add_task(cleanup_files, all_cleanup)
    else:
        # Thread-based path (batch uploads) — thread self-cleans
        # via cleanup_on_finish after success/failure.
        threading.Thread(
            target=_process_job,
            args=(
                task_id, a_p, i_p, o_p,
                title or (audio_file.filename or "Untitled"),
                desc, privacy, token, refresh_token,
                tags, category_id, made_for_kids,
            ),
            kwargs={"cleanup_on_finish": all_cleanup},
            daemon=True,
        ).start()

    return {"task_id": task_id, "job_id": task_id}


# ── Cleanup Helper ────────────────────────────────────────────────────────────
async def _schedule_cleanup(background: BackgroundTasks, *paths: Path, delay_seconds: int = 1800) -> None:
    async def _delayed_delete() -> None:
        await asyncio.sleep(delay_seconds)
        _cleanup_paths(*paths)
    background.add_task(_delayed_delete)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL ENDPOINTS (with usage limit protection)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/tools/vocal-remover")
async def vocal_remover(
    request: Request,
    audio_file: UploadFile = File(...),
    mode: str = Form("2stems"),
    model: str = Form("htdemucs_ft"),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("vocal_remover")),
    _abuse: None = Depends(detect_abuse),
):
    """Vocal Remover using Demucs (Meta). Supports 2-stem and 4-stem separation with model selection."""
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate model choice
    allowed_models = {"htdemucs", "htdemucs_ft"}
    if model not in allowed_models:
        model = "htdemucs_ft"

    task_id = uuid.uuid4().hex
    audio_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_input{audio_ext}"
    output_dir = TOOLS_DIR / f"demucs_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    try:
        if mode == "2stems":
            cmd = ["demucs", "--two-stems", "vocals", "-n", model, "-o", str(output_dir), str(input_path)]
        else:
            cmd = ["demucs", "-n", model, "-o", str(output_dir), str(input_path)]

        _write_status(task_id, {"step": 1, "message": "Initializing Demucs...", "progress": 0})
        result = _run_with_progress(cmd, task_id, "Separating Vocals")
        _write_status(task_id, {"step": 3, "message": "Processing complete!", "progress": 100})

        file_stem = input_path.stem
        demucs_output_folder = output_dir / model / file_stem

        if not demucs_output_folder.exists():
            raise HTTPException(status_code=500, detail="Demucs failed to generate output folder")

        if mode == "2stems":
            available_stems = {
                "vocals": demucs_output_folder / "vocals.wav",
                "no_vocals": demucs_output_folder / "no_vocals.wav",
            }
        else:
            available_stems = {
                "vocals": demucs_output_folder / "vocals.wav",
                "drums": demucs_output_folder / "drums.wav",
                "bass": demucs_output_folder / "bass.wav",
                "other": demucs_output_folder / "other.wav",
            }

        available_stems = {k: v for k, v in available_stems.items() if v.exists()}
        if not available_stems:
            raise HTTPException(status_code=500, detail="Demucs failed to generate output stems")

        stem_urls: dict[str, str] = {}
        download_paths: list[Path] = []

        for stem_name, stem_path in available_stems.items():
            output_name = f"{task_id}_{stem_name}.wav"
            output_dest = DOWNLOADS_DIR / output_name
            shutil.move(str(stem_path), str(output_dest))
            stem_urls[stem_name] = f"/static/{output_name}"
            download_paths.append(output_dest)

        await _schedule_cleanup(background, input_path, output_dir, *download_paths, delay_seconds=1800)
        return {"task_id": task_id, "stems": stem_urls}

    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_dir)
        raise HTTPException(status_code=500, detail=f"Demucs processing failed: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_dir)
        raise HTTPException(status_code=500, detail=str(e))


# ── Audio Trimmer ─────────────────────────────────────────────────────────────
def _trim_with_ffmpeg(input_path: Path, output_path: Path, start_time: float, end_time: float) -> None:
    base_cmd = ["ffmpeg", "-y", "-i", str(input_path), "-ss", f"{start_time:.3f}", "-to", f"{end_time:.3f}", "-c", "copy", str(output_path)]
    result = subprocess.run(base_cmd, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.unlink(missing_ok=True)
    reencode_cmd = ["ffmpeg", "-y", "-i", str(input_path), "-ss", f"{start_time:.3f}", "-to", f"{end_time:.3f}", "-vn", "-acodec", "libmp3lame", "-b:a", "320k", str(output_path.with_suffix(".mp3"))]
    subprocess.run(reencode_cmd, check=True, capture_output=True, text=True)


@app.post("/api/tools/trim-audio")
async def trim_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("trim_audio")),
    _abuse: None = Depends(detect_abuse),
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if end_time <= start_time:
        raise HTTPException(status_code=400, detail="End time must be greater than start time")

    task_id = uuid.uuid4().hex
    audio_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_trim{audio_ext}"

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    output_name = f"{task_id}_trimmed{audio_ext}"
    output_path = DOWNLOADS_DIR / output_name

    try:
        _trim_with_ffmpeg(input_path, output_path, start_time, end_time)
        if not output_path.exists():
            output_path = DOWNLOADS_DIR / f"{task_id}_trimmed.mp3"
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Trimming failed")
        await _schedule_cleanup(background, input_path, output_path, delay_seconds=1200)
        return {"download_url": f"/static/{output_path.name}"}
    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── Slowed + Reverb ───────────────────────────────────────────────────────────
def _build_slowed_reverb_filter(speed: float, reverb: float) -> str:
    speed = max(0.5, min(1.0, speed))
    reverb = max(0.0, min(100.0, reverb))
    decay = 0.2 + (reverb / 100.0) * 0.6
    delay = 600 + int(reverb * 8)
    return f"asetrate=44100*{speed:.3f},aresample=44100,aecho=0.8:0.9:{delay}:{decay:.2f}"


@app.post("/api/tools/slowed-reverb")
async def slowed_reverb(
    request: Request,
    audio_file: UploadFile = File(...),
    speed: float = Form(0.85),
    reverb: float = Form(40),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("slowed_reverb")),
    _abuse: None = Depends(detect_abuse),
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    task_id = uuid.uuid4().hex
    audio_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_slowed{audio_ext}"

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    output_path = DOWNLOADS_DIR / f"{task_id}_slowed_reverb.mp3"
    filter_chain = _build_slowed_reverb_filter(speed, reverb)

    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", filter_chain, "-b:a", "320k", str(output_path)], check=True, capture_output=True, text=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Processing failed")
        await _schedule_cleanup(background, input_path, output_path, delay_seconds=1800)
        return {"download_url": f"/static/{output_path.name}"}
    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── Audio Converter ───────────────────────────────────────────────────────────
def _build_conversion_command(input_path: Path, output_path: Path, target_format: str, bitrate: str) -> list[str]:
    target_format = target_format.lower()
    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-ar", "44100", "-ac", "2"]
    if target_format in {"mp3", "m4a", "ogg"}:
        cmd += ["-b:a", bitrate]
    elif target_format == "flac":
        cmd += ["-compression_level", "5"]
    cmd.append(str(output_path))
    return cmd


@app.post("/api/tools/convert-audio")
async def convert_audio(
    request: Request,
    file: UploadFile = File(...),
    target_format: str = Form(...),
    bitrate: str = Form("320k"),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("convert_audio")),
    _abuse: None = Depends(detect_abuse),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    task_id = uuid.uuid4().hex
    input_ext = Path(file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_convert{input_ext}"

    with input_path.open("wb") as f:
        f.write(await file.read())

    output_ext = target_format.lower()
    output_path = DOWNLOADS_DIR / f"{task_id}_converted.{output_ext}"

    try:
        cmd = _build_conversion_command(input_path, output_path, target_format, bitrate)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Conversion failed")
        size_bytes = output_path.stat().st_size
        await _schedule_cleanup(background, input_path, output_path, delay_seconds=1800)
        return {"download_url": f"/static/{output_path.name}", "size_bytes": size_bytes}
    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── Bass Booster ──────────────────────────────────────────────────────────────
def _build_bass_boost_filter(bass_gain: float, treble_gain: float, volume_gain: float) -> str:
    bass_gain = max(-10.0, min(20.0, bass_gain))
    treble_gain = max(-10.0, min(10.0, treble_gain))
    volume_gain = max(0.0, min(10.0, volume_gain))
    return ",".join([
        f"equalizer=f=60:width_type=h:width=100:g={bass_gain:.1f}",
        f"equalizer=f=15000:width_type=h:width=1000:g={treble_gain:.1f}",
        f"volume={volume_gain:.1f}dB",
        "alimiter=limit=0.95",
    ])


@app.post("/api/tools/bass-boost")
async def bass_boost(
    request: Request,
    audio_file: UploadFile = File(...),
    bass_gain: float = Form(10.0),
    treble_gain: float = Form(0.0),
    volume_gain: float = Form(0.0),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("bass_boost")),
    _abuse: None = Depends(detect_abuse),
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    task_id = uuid.uuid4().hex
    audio_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_bass{audio_ext}"

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    output_path = DOWNLOADS_DIR / f"{task_id}_bass_boosted.mp3"
    filter_chain = _build_bass_boost_filter(bass_gain, treble_gain, volume_gain)

    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", filter_chain, "-b:a", "320k", str(output_path)], check=True, capture_output=True, text=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Processing failed")
        await _schedule_cleanup(background, input_path, output_path, delay_seconds=1800)
        return {"download_url": f"/static/{output_path.name}"}
    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── 8D Audio ──────────────────────────────────────────────────────────────────
def _build_8d_filter(speed_seconds: float, enable_reverb: bool, pattern: str = "circle") -> str:
    speed_seconds = max(2.0, min(20.0, speed_seconds))
    hz = 1.0 / speed_seconds

    if pattern == "figure8":
        # Figure-8 uses a sine pulsator with offset for complex movement
        base = f"apulsator=mode=sine:hz={hz:.4f}:amount=1:offset_l=0.25:offset_r=0.75"
    elif pattern == "bounce":
        # Bounce uses a square-like pulsator for ping-pong effect
        base = f"apulsator=mode=triangle:hz={hz:.4f}:amount=0.9"
    elif pattern == "random":
        # Random uses faster sine with slight frequency modulation
        fast_hz = hz * 1.5
        base = f"apulsator=mode=sine:hz={fast_hz:.4f}:amount=1,apulsator=mode=triangle:hz={hz * 0.7:.4f}:amount=0.5"
    else:
        # Default circle: smooth sine pulsator
        base = f"apulsator=mode=sine:hz={hz:.4f}:amount=1"

    if not enable_reverb:
        return base
    return f"{base},aecho=0.8:0.9:1000:0.3"


@app.post("/api/tools/8d-audio")
async def eight_d_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    speed: float = Form(10.0),
    reverb: int = Form(1),
    pattern: str = Form("circle"),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("8d_audio")),
    _abuse: None = Depends(detect_abuse),
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate pattern
    allowed_patterns = {"circle", "figure8", "bounce", "random"}
    if pattern not in allowed_patterns:
        pattern = "circle"

    task_id = uuid.uuid4().hex
    audio_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_8d{audio_ext}"

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    output_path = DOWNLOADS_DIR / f"{task_id}_8d.mp3"
    filter_chain = _build_8d_filter(speed, bool(reverb), pattern)

    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(input_path), "-filter:a", filter_chain, "-b:a", "320k", str(output_path)], check=True, capture_output=True, text=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Processing failed")
        await _schedule_cleanup(background, input_path, output_path, delay_seconds=1800)
        return {"download_url": f"/static/{output_path.name}"}
    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── Audio Merger ──────────────────────────────────────────────────────────────
def _build_crossfade_filter(count: int, duration: int = 3) -> str:
    if count < 2:
        return ""
    filter_parts = []
    for idx in range(count - 1):
        left = f"[{idx}:a]" if idx == 0 else f"[a{idx}]"
        right = f"[{idx + 1}:a]"
        output = f"[a{idx + 1}]"
        filter_parts.append(f"{left}{right}acrossfade=d={duration}:c1=tri:c2=tri{output}")
    return ";".join(filter_parts)


@app.post("/api/tools/merge-audio")
async def merge_audio(
    request: Request,
    files: list[UploadFile] = File(...),
    crossfade: int = Form(0),
    format: str = Form("mp3"),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("merge_audio")),
    _abuse: None = Depends(detect_abuse),
):
    if not files or len(files) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least two audio files")

    task_id = uuid.uuid4().hex
    temp_dir = TOOLS_DIR / f"merge_{task_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_paths: list[Path] = []
    standardized_paths: list[Path] = []

    try:
        for idx, upload in enumerate(files):
            ext = Path(upload.filename or "track").suffix or ".mp3"
            input_path = temp_dir / f"input_{idx}{ext}"
            with input_path.open("wb") as f:
                f.write(await upload.read())
            input_paths.append(input_path)

            standard_path = temp_dir / f"standard_{idx}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path), "-ar", "44100", "-ac", "2", "-f", "wav", str(standard_path)],
                check=True, capture_output=True, text=True,
            )
            standardized_paths.append(standard_path)

        output_ext = format.lower()
        output_path = DOWNLOADS_DIR / f"{task_id}_merged.{output_ext}"

        if crossfade:
            filter_complex = _build_crossfade_filter(len(standardized_paths))
            inputs = []
            for path in standardized_paths:
                inputs.extend(["-i", str(path)])
            cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", f"[a{len(standardized_paths) - 1}]", "-b:a", "320k", str(output_path)]
        else:
            list_path = temp_dir / "concat.txt"
            list_path.write_text("\n".join([f"file '{path.as_posix()}'" for path in standardized_paths]))
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path)]
            if output_ext == "mp3":
                cmd += ["-c:a", "libmp3lame", "-b:a", "320k"]
            else:
                cmd += ["-c:a", "pcm_s16le"]
            cmd.append(str(output_path))

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Merge failed")

        await _schedule_cleanup(background, temp_dir, output_path, delay_seconds=1800)
        return {"download_url": f"/static/{output_path.name}"}

    except subprocess.CalledProcessError as e:
        _cleanup_paths(temp_dir)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))


# ── Audio Compressor ──────────────────────────────────────────────────────────
def _build_compress_command(input_path: Path, output_path: Path, bitrate: int, format: str) -> list[str]:
    bitrate = max(32, min(320, bitrate))
    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-map", "0:a:0"]
    if bitrate <= 64:
        cmd += ["-ar", "22050", "-ac", "1"]
    else:
        cmd += ["-ar", "44100", "-ac", "2"]
    cmd += ["-b:a", f"{bitrate}k", str(output_path.with_suffix(f".{format}"))]
    return cmd


@app.post("/api/tools/compress-audio")
async def compress_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    bitrate_kbps: int = Form(128),
    format: str = Form("mp3"),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("compress_audio")),
    _abuse: None = Depends(detect_abuse),
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    task_id = uuid.uuid4().hex
    input_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_compress{input_ext}"

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    output_path = DOWNLOADS_DIR / f"{task_id}_compressed"

    try:
        cmd = _build_compress_command(input_path, output_path, bitrate_kbps, format)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        final_path = output_path.with_suffix(f".{format}")
        if not final_path.exists() or final_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Compression failed")
        original_size = input_path.stat().st_size
        new_size = final_path.stat().st_size
        await _schedule_cleanup(background, input_path, final_path, delay_seconds=1800)
        return {"download_url": f"/static/{final_path.name}", "original_size": original_size, "new_size": new_size, "compression_ratio": 1 - (new_size / original_size)}
    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── Noise Reducer ─────────────────────────────────────────────────────────────
def _build_denoise_filter(reduction_amount: int) -> str:
    reduction_amount = max(10, min(30, reduction_amount))
    # Use anlmdn (non-local means denoising) for AI-like quality at higher levels
    if reduction_amount >= 25:
        # Heavy: anlmdn for superior noise reduction + highpass to cut rumble
        strength = 0.00001 * (2 ** (reduction_amount / 5))
        return f"anlmdn=s={strength:.8f}:p=0.002:r=0.002:m=15,highpass=f=80"
    elif reduction_amount >= 15:
        # Medium: Use afftdn with optimized parameters
        return f"afftdn=nr={reduction_amount}:nf=-25:tn=1:om=o"
    else:
        # Light: Gentle afftdn
        return f"afftdn=nr={reduction_amount}:nf=-20:tn=1"


@app.post("/api/tools/denoise-audio")
async def denoise_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    reduction_amount: int = Form(20),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("denoise_audio")),
    _abuse: None = Depends(detect_abuse),
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    task_id = uuid.uuid4().hex
    input_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_denoise{input_ext}"

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    output_path = DOWNLOADS_DIR / f"{task_id}_denoised.mp3"
    filter_chain = _build_denoise_filter(reduction_amount)

    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(input_path), "-af", filter_chain, "-b:a", "320k", str(output_path)], check=True, capture_output=True, text=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Denoising failed")
        await _schedule_cleanup(background, input_path, output_path, delay_seconds=1800)
        return {"download_url": f"/static/{output_path.name}"}
    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── Silence Remover ───────────────────────────────────────────────────────────
def _get_audio_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


@app.post("/api/tools/remove-silence")
async def remove_silence(
    request: Request,
    audio_file: UploadFile = File(...),
    threshold: int = Form(-40),
    duration: float = Form(0.5),
    pad_ms: int = Form(0),
    background: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("remove_silence")),
    _abuse: None = Depends(detect_abuse),
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    task_id = uuid.uuid4().hex
    input_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_silence{input_ext}"

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    output_path = DOWNLOADS_DIR / f"{task_id}_trimmed.mp3"

    try:
        old_duration = _get_audio_duration(input_path)

        # Build filter with optional padding for smoother cuts
        pad_ms = max(0, min(500, pad_ms))
        pad_sec = pad_ms / 1000.0
        if pad_sec > 0:
            # Add small silence padding at cut points for smoother transitions
            filter_chain = f"silenceremove=stop_periods=-1:stop_duration={duration}:stop_threshold={threshold}dB,apad=pad_dur={pad_sec}"
        else:
            filter_chain = f"silenceremove=stop_periods=-1:stop_duration={duration}:stop_threshold={threshold}dB"

        subprocess.run(["ffmpeg", "-y", "-i", str(input_path), "-af", filter_chain, "-b:a", "320k", str(output_path)], check=True, capture_output=True, text=True)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Silence removal failed")
        new_duration = _get_audio_duration(output_path)
        await _schedule_cleanup(background, input_path, output_path, delay_seconds=1800)
        return {
            "download_url": f"/static/{output_path.name}",
            "old_duration": old_duration,
            "new_duration": new_duration,
            "time_saved": max(0.0, old_duration - new_duration),
            "pad_ms": pad_ms,
        }
    except subprocess.CalledProcessError as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e.stderr or e.stdout}")
    except Exception as e:
        _cleanup_paths(input_path, output_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── BPM & Key Finder ─────────────────────────────────────────────────────────
def _detect_key(y: np.ndarray, sr: int) -> tuple[str, float]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.mean(chroma, axis=1)
    chroma_vals = chroma_vals / np.sum(chroma_vals)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile /= np.sum(major_profile)
    minor_profile /= np.sum(minor_profile)

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    max_corr = -1
    detected_key = "Unknown"

    for i in range(12):
        major_corr = np.corrcoef(chroma_vals, np.roll(major_profile, i))[0, 1]
        minor_corr = np.corrcoef(chroma_vals, np.roll(minor_profile, i))[0, 1]
        if major_corr > max_corr:
            max_corr = major_corr
            detected_key = f"{notes[i]} Major"
        if minor_corr > max_corr:
            max_corr = minor_corr
            detected_key = f"{notes[i]} Minor"

    return detected_key, max(0.0, min(1.0, max_corr))


def _estimate_tempo(onset_env: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    if hasattr(librosa, "feature") and hasattr(librosa.feature, "rhythm") and hasattr(librosa.feature.rhythm, "tempo"):
        return librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None)
    return librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None)


@app.post("/api/tools/analyze-bpm")
async def analyze_bpm(
    request: Request,
    audio_file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    _limit: None = Depends(UsageLimitChecker("analyze_bpm")),
    _abuse: None = Depends(detect_abuse),
):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    task_id = uuid.uuid4().hex
    audio_ext = Path(audio_file.filename).suffix or ".mp3"
    input_path = UPLOAD_DIR / f"{task_id}_bpm{audio_ext}"

    with input_path.open("wb") as f:
        f.write(await audio_file.read())

    try:
        y, sr = librosa.load(str(input_path), sr=None, mono=True, duration=120)
        y, _ = librosa.effects.trim(y, top_db=25)

        hop_length = 512
        tempo_candidates = []
        confidence_scores = []
        win_length = int(20 * sr)
        win_hop = int(10 * sr)

        for start in range(0, max(len(y) - win_length, 0) + 1, win_hop):
            segment = y[start:start + win_length]
            if segment.size < sr * 5:
                continue
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=hop_length)
            tempos = _estimate_tempo(onset_env, sr, hop_length)
            if isinstance(tempos, np.ndarray) and tempos.size:
                tempos = tempos[(tempos > 40) & (tempos < 220)]
                if tempos.size:
                    tempo_candidates.append(float(np.median(tempos)))
                    confidence_scores.append(float(np.std(tempos)))

        if tempo_candidates:
            tempo_value = float(np.median(tempo_candidates))
            dispersion = float(np.median(confidence_scores)) if confidence_scores else 0.0
        else:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempos = _estimate_tempo(onset_env, sr, hop_length)
            if isinstance(tempos, np.ndarray) and tempos.size:
                tempos = tempos[(tempos > 40) & (tempos < 220)]
                tempo_value = float(np.median(tempos)) if tempos.size else 0.0
                dispersion = float(np.std(tempos)) if tempos.size else 0.0
            else:
                tempo_value, dispersion = 0.0, 0.0

        if np.isnan(tempo_value) or tempo_value <= 0:
            fallback_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_value = float(fallback_tempo[0]) if isinstance(fallback_tempo, np.ndarray) and fallback_tempo.size else float(fallback_tempo)

        if np.isnan(tempo_value):
            tempo_value = 0.0
        bpm = int(round(tempo_value)) if tempo_value else 0

        bpm_confidence = max(0.0, min(1.0, 1.0 - (dispersion / 20.0))) if tempo_candidates else (0.6 if bpm else 0.0)
        key, key_confidence = _detect_key(y, sr)

        _cleanup_paths(input_path)
        return {"bpm": bpm, "key": key, "confidence": bpm_confidence, "key_confidence": key_confidence}

    except Exception as e:
        _cleanup_paths(input_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# YOUTUBE UPLOAD ENDPOINTS (with auto-promo & upload count)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload_and_publish")
async def upload_and_publish(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
    title: str = Form(""),
    description: str = Form(""),
    privacy_status: str = Form("private"),
    made_for_kids: str = Form("no"),
    tags: str = Form(""),
    category_id: str = Form("10"),
    youtube_access_token: str = Form(...),
    youtube_refresh_token: str = Form(""),
    plan_type: str = Form("free"),
):
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]
    return await _handle_upload(
        audio_file, image_file, background_tasks, title, description, privacy_status,
        youtube_access_token, youtube_refresh_token,
        tags=parsed_tags, category_id=category_id,
        made_for_kids=made_for_kids.lower() == "yes",
        plan_type=plan_type,
    )


@app.post("/upload_to_youtube")
async def upload_to_youtube(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
    title: str = Form(""),
    description: str = Form(""),
    privacy_status: str = Form("private"),
    made_for_kids: str = Form("no"),
    tags: str = Form(""),
    category_id: str = Form("10"),
    youtube_access_token: str = Form(...),
    youtube_refresh_token: str = Form(""),
    plan_type: str = Form("free"),
):
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]
    return await _handle_upload(
        audio_file, image_file, background_tasks, title, description, privacy_status,
        youtube_access_token, youtube_refresh_token,
        tags=parsed_tags, category_id=category_id,
        made_for_kids=made_for_kids.lower() == "yes",
        plan_type=plan_type,
    )


# ── Batch Upload (Pro/Max only) ──────────────────────────────────────────────
@app.post("/api/batch-upload")
async def batch_upload(
    image_file: UploadFile = File(...),
    audio_files: list[UploadFile] = File(...),
    titles: str = Form(""),
    description: str = Form(""),
    privacy_status: str = Form("private"),
    tags: str = Form(""),
    category_id: str = Form("10"),
    youtube_access_token: str = Form(...),
    youtube_refresh_token: str = Form(""),
    plan_type: str = Form("free"),
):
    """
    Batch Upload: 1 Image + multiple Audio files.
    Only available for Pro and Max plans.
    """
    if plan_type == "free":
        raise HTTPException(
            status_code=403,
            detail="Batch upload is only available for Pro and Max plans. Upgrade at /pricing",
        )

    if len(audio_files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 audio files per batch")

    title_list = [t.strip() for t in titles.split("|")] if titles else []
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]

    results = []
    for idx, audio_file in enumerate(audio_files):
        title = title_list[idx] if idx < len(title_list) else (audio_file.filename or f"Track {idx + 1}")
        result = await _handle_upload(
            audio_file, image_file, None, title, description, privacy_status,
            youtube_access_token, youtube_refresh_token,
            tags=parsed_tags, category_id=category_id,
            plan_type=plan_type,
        )
        results.append(result)

    return {"batch_results": results, "total": len(results)}


# ── Progress & Cancel ─────────────────────────────────────────────────────────
@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    async def event_generator():
        last = None
        while True:
            cur = _read_status(task_id)
            if cur != last:
                yield f"data: {json.dumps(cur)}\n\n"
                last = cur
                if cur.get("progress", 0) >= 100:
                    break
            await asyncio.sleep(0.4)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/cancel/{task_id}")
async def cancel_job(task_id: str):
    _cancel_file(task_id).touch()
    return {"cancelled": True}


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "4.0.0", "service": "TuneVid API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
