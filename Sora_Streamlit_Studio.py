#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sora Streamlit Studio — Budget, Rate-Limits, Batch Queue, Duration Slider

Features:
 - Images API for generating reference images (gpt-image-1)
 - Multiple reference image uploads with automatic montage combining
 - Configurable montage layout (Auto, 1×N, N×1, 2×2, 3×2, 3×3, 3×N, Custom)
 - Row-wise, top-aligned prompt inputs for multi-generation
 - Safer reference upload: passes an open file handle to the Videos API
 - Improved polling: refreshes status and progress from retrieve()
"""
import os
import io
import time
import math
import base64
import tempfile
import shutil
import subprocess
import threading
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Any, Dict

import streamlit as st

try:
    from PIL import Image, ImageOps
    PILLOW_AVAILABLE = True
except ImportError:
    Image = None
    ImageOps = None
    PILLOW_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

APP_NAME = "Sora Streamlit Studio"
DEFAULT_OUTPUT_DIR = Path("outputs")

PRICE_PER_SECOND = {
    "sora-2": 0.10,
    "sora-2-pro": 0.30,
}

CURRENT_SUPPORTED_DURATIONS = [4, 8, 12]

RPM_TIERS = {
    "Free (unsupported)": 0,
    "Tier 1": 5,
    "Tier 2": 10,
    "Tier 3": 25,
    "Tier 4": 40,
    "Tier 5": 75,
}

# -------------------------
# Utilities & Budget
# -------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_bytes(path: Path, data: bytes) -> Path:
    ensure_dir(path.parent)
    path.write_bytes(data)
    return path

def parse_size(size_str: str, fallback: Tuple[int, int] = (720, 1280)) -> Tuple[int, int]:
    try:
        w, h = size_str.lower().split("x")
        return int(w), int(h)
    except Exception:
        return fallback

def pil_resize_letterbox(img: "Image.Image", canvas_size: Tuple[int, int]) -> "Image.Image":
    if not PILLOW_AVAILABLE:
        return img
    cw, ch = canvas_size
    canvas = Image.new("RGB", (cw, ch), color=(0, 0, 0))
    thumb = ImageOps.contain(img, (cw, ch))
    ox = (cw - thumb.width) // 2
    oy = (ch - thumb.height) // 2
    canvas.paste(thumb, (ox, oy))
    return canvas

def pil_save_image(
    b64_png: str,
    out: Path,
    size: Optional[Tuple[int, int]] = (720, 1280),
    letterbox: bool = True
) -> Optional[Path]:
    try:
        raw = base64.b64decode(b64_png)
        if not PILLOW_AVAILABLE:
            return write_bytes(out, raw)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if size:
            img = pil_resize_letterbox(img, size) if letterbox else img.resize(size)
        ensure_dir(out.parent)
        img.save(out)
        return out
    except Exception as e:
        st.warning(f"Failed to save image: {e}")
        return None

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def price_rate(model: str) -> float:
    return PRICE_PER_SECOND.get(model, 0.10)

def estimate_cost(num_videos: int, seconds: int, model: str) -> float:
    return num_videos * seconds * price_rate(model)

def init_budget_state():
    st.session_state.setdefault("spent_usd", 0.0)
    st.session_state.setdefault("counted_ids", set())
    st.session_state.setdefault("budget_enabled", False)
    st.session_state.setdefault("budget_limit", 10.0)

def add_spend_if_new(video_id: str, seconds: int, model: str):
    if video_id and video_id not in st.session_state.counted_ids:
        st.session_state.counted_ids.add(video_id)
        st.session_state.spent_usd += seconds * price_rate(model)

# -------------------------
# Rate Limiter
# -------------------------

class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(0, int(rpm))
        self.tokens = self.rpm
        self.lock = threading.Lock()
        self.last_refill = time.monotonic()

    def _maybe_refill(self):
        now = time.monotonic()
        if now - self.last_refill >= 60.0:
            self.tokens = self.rpm
            self.last_refill = now

    def acquire(self, calls: int = 1):
        if self.rpm <= 0:
            return
        while True:
            with self.lock:
                self._maybe_refill()
                if self.tokens >= calls:
                    self.tokens -= calls
                    return
                wait_s = max(0.1, 60.0 - (time.monotonic() - self.last_refill))
            time.sleep(wait_s)

def get_rate_limiter() -> Optional[RateLimiter]:
    return st.session_state.get("rate_limiter")

def rl_acquire(n: int = 1):
    rl = get_rate_limiter()
    if rl:
        rl.acquire(n)

# -------------------------
# OpenAI helpers
# -------------------------

def init_openai_client(api_key: Optional[str]) -> Optional[OpenAI]:
    if not OPENAI_AVAILABLE:
        st.error("OpenAI SDK not installed. Try: `pip install openai`")
        return None
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        st.warning("Provide an OpenAI API key in the sidebar.")
        return None
    os.environ["OPENAI_API_KEY"] = key
    try:
        return OpenAI()
    except Exception as e:
        st.error(f"Failed to init OpenAI client: {e}")
        return None

def _classify_error(e: Exception) -> Dict[str, Any]:
    s = str(e)
    status = None
    if hasattr(e, "status_code"):
        status = getattr(e, "status_code")
    elif "Error code:" in s:
        try:
            status = int(s.split("Error code:")[1].split("-")[0].strip())
        except Exception:
            status = None
    quota = "insufficient_quota" in s.lower() or "quota" in s.lower()
    return {"status": status, "is_quota": quota, "text": s}

def with_retries(
    func: Callable[[], Any],
    attempts: int = 2,
    base_delay: float = 1.5,
    max_delay: float = 6.0
) -> Any:
    last_exc = None
    for i in range(attempts):
        try:
            return func()
        except Exception as e:
            info = _classify_error(e)
            last_exc = e
            if info["is_quota"]:
                st.error("Quota exceeded. Details: " + info["text"])
                raise
            if info["status"] == 429 or (info["status"] and 500 <= info["status"] < 600):
                if i < attempts - 1:
                    delay = min(max_delay, base_delay * (2 ** i))
                    time.sleep(delay)
                    continue
            break
    if last_exc:
        raise last_exc
    return None

def call_api_with_rl(fn: Callable[[], Any]) -> Any:
    rl_acquire(1)
    return with_retries(fn, attempts=2)

# -------------------------
# Poll & Download, Prompt Enhance
# -------------------------

def poll_and_download_video(
    client: OpenAI,
    video,
    out_dir: Path,
    name_prefix: str,
    seconds: int,
    model: str
) -> Tuple[Optional[Path], Optional[str]]:
    status = getattr(video, "status", None)
    progress = getattr(video, "progress", 0) or 0
    ph = st.empty()
    bar = st.progress(0)

    while status not in ("completed", "failed", "cancelled"):
        try:
            video = client.videos.retrieve(video.id)
            status = getattr(video, "status", status)
            progress = getattr(video, "progress", progress) or progress
        except Exception:
            # keep last known values
            pass
        pct = int(max(0, min(100, progress)))
        bar.progress(pct)
        ph.info(f"Status: {status} • {pct}%")
        time.sleep(1.0)

    if getattr(video, "status", "failed") == "failed":
        err = getattr(getattr(video, "error", None), "message", "Video generation failed")
        ph.error(err)
        return None, None

    ph.success("Completed. Downloading...")
    try:
        vid_id = getattr(video, "id", "sora_video")
        content = client.videos.download_content(vid_id, variant="video")
        out_path = out_dir / f"{name_prefix}_{vid_id}.mp4"
        content.write_to_file(out_path)
        ph.success(f"Saved: {out_path.name}")
        add_spend_if_new(vid_id, seconds, model)
        return out_path, vid_id
    except Exception as e:
        ph.error(f"Download failed: {e}")
        return None, None

def enhance_prompt(client: OpenAI, prompt: str, style: str = "director") -> str:
    TEMPLATE_MAP = {
        "director": "You are a sharp film director. Rewrite the following video prompt for clarity, specificity, and cinematic detail (camera, lighting, lens, movement, environment). Return only the revised prompt.",
        "clean": "Rewrite clearly and concisely without losing meaning. Return only the revised text.",
        "pixar": "Rewrite as a warm, detailed Pixar-style prompt focusing on character, wardrobe, lensing, lighting, and blocking. Return only the revised prompt.",
    }
    ins = TEMPLATE_MAP.get(style, TEMPLATE_MAP["director"])
    try:
        resp = call_api_with_rl(
            lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ins},
                    {"role": "user", "content": prompt}
                ]
            )
        )
        return resp.choices[0].message.content or prompt
    except Exception as e:
        st.warning(f"Enhancement failed, using original prompt. ({e})")
        return prompt

# -------------------------
# Reference Images
# -------------------------

def make_reference_image(
    client: OpenAI,
    text: str,
    out_dir: Path,
    name: str,
    canvas_size: Tuple[int, int] = (720, 1280)
) -> Optional[Path]:
    """Generate a square image then letterbox to target canvas."""
    try:
        resp = call_api_with_rl(
            lambda: client.images.generate(
                model="gpt-image-1",       # patched from "dall-e-3"
                prompt=text,
                size="1024x1024",
                response_format="b64_json"
            )
        )
        b64 = resp.data[0].b64_json
        out_path = out_dir / f"{name}.png"
        return pil_save_image(b64, out_path, size=canvas_size, letterbox=True)
    except Exception as e:
        st.error(f"Reference image generation failed: {e}")
        return None

def _decide_grid(
    n: int,
    layout_mode: str,
    custom_cols: Optional[int],
    custom_rows: Optional[int]
) -> Tuple[int, int]:
    if layout_mode == "Auto":
        if n <= 2:
            return (1, n)
        elif n <= 4:
            return (2, 2)
        elif n <= 6:
            return (3, 2)
        elif n <= 9:
            return (3, 3)
        else:
            cols = 3
            rows = math.ceil(n / cols)
            return (cols, rows)
    if layout_mode == "1×N (vertical stack)":
        return (1, n)
    if layout_mode == "N×1 (horizontal strip)":
        return (n, 1)
    if layout_mode == "2×2 grid":
        return (2, 2)
    if layout_mode == "3×2 grid":
        return (3, 2)
    if layout_mode == "3×3 grid":
        return (3, 3)
    if layout_mode == "3×N (3 cols, rows as needed)":
        cols = 3
        rows = math.ceil(n / cols)
        return (cols, rows)
    c = max(1, int(custom_cols or 1))
    r = max(1, int(custom_rows or 1))
    return (c, r)

def combine_reference_images(
    paths: List[Path],
    out_path: Path,
    canvas_size: Tuple[int, int],
    *,
    layout_mode: str = "Auto",
    custom_cols: Optional[int] = None,
    custom_rows: Optional[int] = None,
    padding: int = 0,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> Optional[Path]:
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    if not PILLOW_AVAILABLE:
        st.warning("Pillow is not installed; using only the first uploaded reference.")
        return paths[0]

    cw, ch = canvas_size
    cols, rows = _decide_grid(len(paths), layout_mode, custom_cols, custom_rows)
    capacity = cols * rows
    imgs = paths[:capacity]
    if len(paths) > capacity:
        st.warning(f"Montage capacity is {capacity} cells ({cols}×{rows}). Using the first {capacity} image(s) out of {len(paths)}.")

    p = max(0, int(padding))
    inner_w = max(1, cw - (cols + 1) * p)
    inner_h = max(1, ch - (rows + 1) * p)
    cell_w = max(1, inner_w // cols)
    cell_h = max(1, inner_h // rows)

    sheet = Image.new("RGB", (cw, ch), color=bg_color)

    def load_img(pth: Path) -> Optional["Image.Image"]:
        try:
            im = Image.open(pth).convert("RGB")
            return im
        except Exception:
            return None

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(imgs):
                break
            im = load_img(imgs[idx])
            idx += 1
            if im is None:
                continue
            thumb = ImageOps.contain(im, (cell_w, cell_h))
            x = p + c * (cell_w + p)
            y = p + r * (cell_h + p)
            ox = x + (cell_w - thumb.width) // 2
            oy = y + (cell_h - thumb.height) // 2
            sheet.paste(thumb, (ox, oy))

    ensure_dir(out_path.parent)
    sheet.save(out_path)
    return out_path

def build_single_reference(
    ref_paths: List[Path],
    out_root: Path,
    canvas_size: Tuple[int, int],
    *,
    montage_layout: str = "Auto",
    montage_cols: Optional[int] = None,
    montage_rows: Optional[int] = None,
    montage_padding: int = 0,
) -> Optional[Path]:
    if not ref_paths:
        return None
    if len(ref_paths) == 1:
        return ref_paths[0]
    combined = out_root / "refs" / f"combined_{int(time.time())}.png"
    return combine_reference_images(
        ref_paths,
        combined,
        canvas_size,
        layout_mode=montage_layout,
        custom_cols=montage_cols,
        custom_rows=montage_rows,
        padding=montage_padding,
    )

# -------------------------
# Data models & Flows
# -------------------------

@dataclass
class GenJob:
    idx: int
    prompt: str
    enhanced_prompt: Optional[str] = None
    references: Optional[List[Path]] = None
    result_path: Optional[Path] = None

def _create_video_with_optional_ref(client: OpenAI, *, model: str, prompt: str,
                                    size: str, seconds: int, ref_path: Optional[Path]):
    """Create a video, safely passing a single reference as an open file handle if present."""
    kwargs: Dict[str, Any] = dict(model=model, prompt=prompt, size=size, seconds=int(seconds))
    if ref_path and ref_path.exists():
        with ref_path.open("rb") as f:
            kwargs["input_reference"] = f  # adjust key if your account uses a different name
            return call_api_with_rl(lambda: client.videos.create(**kwargs))
    return call_api_with_rl(lambda: client.videos.create(**kwargs))

def generate_single(
    client: OpenAI,
    prompt: str,
    size: str,
    seconds: int,
    references: Optional[List[Path]],
    model: str
) -> Optional[Path]:
    ref_path = Path(references[0]) if references else None
    vid = _create_video_with_optional_ref(client, model=model, prompt=prompt, size=size, seconds=seconds, ref_path=ref_path)
    path, _ = poll_and_download_video(client, vid, st.session_state.output_dir, "single", seconds, model)
    return path

def generate_and_remix(
    client: OpenAI,
    base_prompt: str,
    remix_prompts: List[str],
    size: str,
    seconds: int,
    references: Optional[List[Path]],
    model: str
) -> List[Optional[Path]]:
    outputs: List[Optional[Path]] = []
    ref_path = Path(references[0]) if references else None
    base = _create_video_with_optional_ref(client, model=model, prompt=base_prompt, size=size, seconds=seconds, ref_path=ref_path)
    base_path, _ = poll_and_download_video(client, base, st.session_state.output_dir, "remix_base", seconds, model)
    outputs.append(base_path)

    current = base
    for i, rprompt in enumerate(remix_prompts, start=1):
        st.write(f"Remix step {i}…")
        remix_job = call_api_with_rl(lambda: client.videos.remix(video_id=current.id, prompt=rprompt))
        out_path, _ = poll_and_download_video(client, remix_job, st.session_state.output_dir, f"remix_{i}", seconds, model)
        outputs.append(out_path)
        current = remix_job
    return outputs

def generate_multiple(
    client: OpenAI,
    jobs: List[GenJob],
    size: str,
    seconds: int,
    model: str,
    concurrent_workers: int = 2
) -> List[GenJob]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(job: GenJob):
        ref_path = Path(job.references[0]) if job.references else None
        try:
            vid = _create_video_with_optional_ref(
                client, model=model,
                prompt=(job.enhanced_prompt or job.prompt),
                size=size, seconds=seconds, ref_path=ref_path
            )
            out_path, _ = poll_and_download_video(client, vid, st.session_state.output_dir, f"multi_{job.idx+1}", seconds, model)
            return job.idx, out_path
        except Exception as e:
            st.error(f"Job {job.idx+1} failed: {e}")
            return job.idx, None

    with ThreadPoolExecutor(max_workers=max(1, int(concurrent_workers))) as ex:
        futures = [ex.submit(worker, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                idx, path = fut.result()
                jobs[idx].result_path = path
            except Exception as e:
                st.error(f"A job failed: {e}")
    return jobs

# -------------------------
# Stitcher
# -------------------------

def stitch_videos(files: List[Path], out_path: Path) -> Optional[Path]:
    if not files:
        st.warning("No files to stitch.")
        return None
    if not check_ffmpeg():
        st.error("ffmpeg not found in PATH.")
        return None
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
        for f in files:
            tf.write(f"file '{f.as_posix()}'\n")
        tf.flush()
        concat_list = tf.name
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c", "copy", out_path.as_posix(), "-y"],
            check=True
        )
        return out_path
    except Exception as e:
        st.error(f"Stitch failed: {e}")
        return None
    finally:
        try:
            os.unlink(concat_list)
        except Exception:
            pass

# -------------------------
# UI Helper
# -------------------------

def render_multi_prompt_inputs(
    n: int,
    *,
    cols_per_row: int = 2,
    key_prefix: str = "multi_prompt_",
    gap: str = "large",
    height: int = 120,
    placeholder: str = "Describe your video..."
) -> List[str]:
    prompts: List[str] = []
    supports_va = "vertical_alignment" in inspect.signature(st.columns).parameters
    total_rows = math.ceil(n / cols_per_row)
    idx = 0
    for _ in range(total_rows):
        kwargs = {"gap": gap}
        if supports_va:
            kwargs["vertical_alignment"] = "top"
        cols = st.columns(cols_per_row, **kwargs)
        for c in range(cols_per_row):
            if idx >= n:
                with cols[c]:
                    st.empty()
                continue
            with cols[c]:
                prompts.append(
                    st.text_area(
                        f"Prompt #{idx+1}",
                        key=f"{key_prefix}{idx}",
                        height=height,
                        placeholder=placeholder,
                        label_visibility="visible",
                    )
                )
            idx += 1
    return prompts

# -------------------------
# UI — App
# -------------------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Web UI for Sora video generation with prompt enhancement, references, multi-gen concurrency, remix, batch queue, and budget/rate-limit guardrails.")

with st.sidebar:
    st.header("API & Limits")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = st.selectbox("Sora Model", ["sora-2", "sora-2-pro"], index=0)

    seconds_slider = st.slider("Duration (seconds)", min_value=1, max_value=60, value=4, step=1)
    strict_durations = st.checkbox("Strict supported durations", value=True, help="When on, snaps to commonly supported values to avoid API errors.")
    st.caption(f"Currently supported durations: {', '.join(map(str, CURRENT_SUPPORTED_DURATIONS))} seconds.")
    seconds = min(CURRENT_SUPPORTED_DURATIONS, key=lambda x: abs(x - seconds_slider)) if strict_durations else seconds_slider

    size_choice = st.selectbox("Frame Size", ["720x1280 (portrait)", "1280x720 (landscape)", "Custom"], index=0)
    size = st.text_input("Custom size (e.g., 720x1280)", "720x1280") if size_choice == "Custom" else size_choice.split()[0]
    canvas_w, canvas_h = parse_size(size)

    tier = st.selectbox("Rate limit tier", list(RPM_TIERS.keys()), index=1)
    rpm = RPM_TIERS[tier]
    if "rate_limiter" not in st.session_state or getattr(st.session_state.get("rate_limiter"), "rpm", None) != rpm:
        st.session_state["rate_limiter"] = RateLimiter(rpm)

    st.subheader("Prompt Enhancement")
    use_enhance = st.checkbox("Enhance prompts before generating", value=False)
    enhance_style = st.selectbox("Enhancement style", ["director", "pixar", "clean"], index=0)

    st.subheader("Reference Montage")
    st.caption("Combine multiple uploaded images into one reference using these settings.")
    montage_layout = st.selectbox(
        "Montage layout",
        ["Auto", "1×N (vertical stack)", "N×1 (horizontal strip)", "2×2 grid", "3×2 grid", "3×3 grid", "3×N (3 cols, rows as needed)", "Custom"],
        index=0,
        help="Auto chooses a grid to fit your count. For Custom, set columns/rows below."
    )
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    with col_m1:
        montage_padding = st.number_input("Padding (px)", min_value=0, max_value=40, value=0, step=1)
    with col_m2:
        montage_cols = st.number_input("Custom columns", min_value=1, max_value=8, value=3, step=1, disabled=(montage_layout != "Custom"))
    with col_m3:
        montage_rows = st.number_input("Custom rows", min_value=1, max_value=8, value=2, step=1, disabled=(montage_layout != "Custom"))

    init_budget_state()
    st.subheader("Budget")
    st.session_state.budget_enabled = st.checkbox("Enable budget cap", value=st.session_state.budget_enabled)
    st.session_state.budget_limit = st.number_input(
        "Max budget for this session ($)",
        min_value=1.0,
        value=float(st.session_state.budget_limit),
        step=1.0,
        disabled=not st.session_state.budget_enabled
    )

    spent = float(st.session_state.spent_usd)
    if st.session_state.budget_enabled:
        remaining = max(0.0, st.session_state.budget_limit - spent)
        pct = 0.0 if st.session_state.budget_limit <= 0 else min(1.0, spent / st.session_state.budget_limit)
        st.progress(pct, text=f"Spent ${spent:.2f} / ${st.session_state.budget_limit:.2f} — Remaining ${remaining:.2f}")
    else:
        st.info(f"Session spend (est.): ${spent:.2f}")
    if st.button("Reset spend counter"):
        st.session_state.spent_usd = 0.0
        st.session_state.counted_ids = set()
        st.success("Spend counter reset.")

    st.subheader("Outputs")
    out_root = st.text_input("Output folder", value=str(DEFAULT_OUTPUT_DIR))
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = ensure_dir(Path(out_root) / time.strftime("%Y%m%d_%H%M%S"))
        st.session_state.last_out_root = out_root
    if st.session_state.get("last_out_root") != out_root:
        st.session_state.output_dir = ensure_dir(Path(out_root) / time.strftime("%Y%m%d_%H%M%S"))
        st.session_state.last_out_root = out_root

    colX, colY, colZ = st.columns([1, 1, 1])
    with colX:
        if st.button("New session folder"):
            st.session_state.output_dir = ensure_dir(Path(out_root) / time.strftime("%Y%m%d_%H%M%S"))
    with colY:
        if st.button("Zip session"):
            try:
                base = st.session_state.output_dir.with_suffix("")
                zip_path = shutil.make_archive(base.as_posix(), 'zip', st.session_state.output_dir.as_posix())
                z = Path(zip_path)
                if z.exists():
                    st.success(f"Created: {z.name}")
                    st.download_button("Download ZIP", data=z.read_bytes(), file_name=z.name, mime="application/zip")
            except Exception as e:
                st.error(f"Zip failed: {e}")
    with colZ:
        if st.button("Clear session folder"):
            try:
                shutil.rmtree(st.session_state.output_dir)
                st.session_state.output_dir = ensure_dir(Path(out_root) / time.strftime("%Y%m%d_%H%M%S"))
                st.success("Cleared session outputs.")
            except Exception as e:
                st.error(f"Clear failed: {e}")

    st.write(f"Session folder: `{st.session_state.output_dir}`")

client = init_openai_client(api_key_input)

tabs = st.tabs(["Single", "Multiple", "Remix", "Batch", "Assets"])

# ---------- SINGLE ----------
with tabs[0]:
    st.subheader("Single Generation")
    colA, colB = st.columns([1, 1], gap="large")
    with colA:
        user_prompt = st.text_area("Prompt", height=150, placeholder="Describe your video...")
        gen_refs_from_prompt = st.text_input("Optional: Generate reference image from a prompt (leave blank to skip)")
        st.caption("Upload multiple images to combine them into a single reference montage.")
        ref_uploaded = st.file_uploader("Or upload reference image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        est_cost = estimate_cost(1, int(seconds), model)
        st.info(f"**Estimated cost:** ~${est_cost:.2f} • **API calls:** {1 + (1 if use_enhance else 0)}")

        if st.session_state.budget_enabled and (spent + est_cost) > st.session_state.budget_limit:
            st.warning(f"This job (~${est_cost:.2f}) may exceed your remaining budget.")

    with colB:
        st.write("Preview & Actions")
        if client and st.button("Generate (Single)"):
            if st.session_state.budget_enabled and (spent + est_cost) > st.session_state.budget_limit:
                st.error("Budget cap would be exceeded. Cannot proceed.")
                st.stop()

            refs: List[Path] = []

            if gen_refs_from_prompt.strip():
                out_img = make_reference_image(
                    client, gen_refs_from_prompt,
                    st.session_state.output_dir / "refs", "ref_1",
                    canvas_size=(canvas_w, canvas_h)
                )
                if out_img:
                    refs.append(out_img)
                    st.image(str(out_img), caption="Generated reference", use_container_width=True)

            if ref_uploaded:
                refs_dir = ensure_dir(st.session_state.output_dir / "refs")
                for i, uf in enumerate(ref_uploaded, start=1):
                    p = refs_dir / f"upload_{i}.png"
                    write_bytes(p, uf.getvalue())
                    refs.append(p)

            single_ref_path = build_single_reference(
                refs, st.session_state.output_dir, (canvas_w, canvas_h),
                montage_layout=montage_layout,
                montage_cols=int(montage_cols),
                montage_rows=int(montage_rows),
                montage_padding=int(montage_padding),
            )
            refs = [single_ref_path] if single_ref_path else []

            if refs:
                st.success("Using 1 reference image.")
                st.image(str(refs[0]), caption=Path(refs[0]).name, use_container_width=True)

            final_prompt = user_prompt
            enhanced_txt = None
            if use_enhance and user_prompt.strip():
                enhanced_txt = enhance_prompt(client, user_prompt, enhance_style)
                final_prompt = enhanced_txt

            if not user_prompt.strip() and not gen_refs_from_prompt.strip() and not ref_uploaded:
                st.warning("Please provide a video prompt or a reference image prompt or upload a reference image.")
            else:
                try:
                    with st.spinner("Generating video..."):
                        out_path = generate_single(client, final_prompt or user_prompt, size, int(seconds), refs, model)
                    if out_path and out_path.exists():
                        st.success("Done.")
                        st.video(str(out_path))
                        st.download_button("Download video", data=out_path.read_bytes(), file_name=out_path.name, mime="video/mp4")
                        if enhanced_txt:
                            st.markdown("**Enhanced prompt**")
                            st.code(enhanced_txt)
                except Exception as e:
                    st.exception(e)

# ---------- MULTIPLE ----------
with tabs[1]:
    st.subheader("Multiple / Concurrent Generation")
    n = st.number_input("Number of generations", min_value=2, max_value=24, value=3, step=1)
    workers = st.number_input("Concurrent workers", min_value=1, max_value=24, value=2, step=1)
    st.caption("Run up to 'Concurrent workers' jobs in parallel. Rate limiter controls API call pacing globally.")

    m_est_cost = estimate_cost(int(n), int(seconds), model)
    m_calls_per_job = 1 + (1 if use_enhance else 0)
    m_total_calls = int(n) * m_calls_per_job
    st.info(f"**Estimated cost:** ~${m_est_cost:.2f} • **Total API calls:** ~{m_total_calls}")

    if st.session_state.budget_enabled and (spent + m_est_cost) > st.session_state.budget_limit:
        st.warning(f"This run (~${m_est_cost:.2f}) may exceed your remaining budget.")

    st.write("Prompts")
    multi_prompts: List[str] = render_multi_prompt_inputs(int(n))

    st.write("References (optional)")
    m_gen_ref_prompt = st.text_input("Generate a shared reference image from prompt (optional)", key="multi_ref_gen")
    m_ref_uploaded = st.file_uploader("Or upload reference image(s) shared by all jobs", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="multi_ref_upload")

    if client and st.button("Generate All"):
        if st.session_state.budget_enabled and (spent + m_est_cost) > st.session_state.budget_limit:
            st.error("Budget cap would be exceeded. Cannot proceed.")
            st.stop()

        refs: List[Path] = []
        if m_gen_ref_prompt.strip():
            out_img = make_reference_image(
                client, m_gen_ref_prompt,
                st.session_state.output_dir / "refs", "shared_ref",
                canvas_size=(canvas_w, canvas_h)
            )
            if out_img:
                refs.append(out_img)
        if m_ref_uploaded:
            refs_dir = ensure_dir(st.session_state.output_dir / "refs")
            for i, uf in enumerate(m_ref_uploaded, start=1):
                p = refs_dir / f"shared_upload_{i}.png"
                write_bytes(p, uf.getvalue())
                refs.append(p)

        single_ref_path = build_single_reference(
            refs, st.session_state.output_dir, (canvas_w, canvas_h),
            montage_layout=montage_layout,
            montage_cols=int(montage_cols),
            montage_rows=int(montage_rows),
            montage_padding=int(montage_padding),
        )
        refs = [single_ref_path] if single_ref_path else []

        if refs:
            st.image(str(refs[0]), caption=f"Shared reference used: {Path(refs[0]).name}", use_container_width=True)

        jobs: List[GenJob] = []
        for i, ptxt in enumerate(multi_prompts):
            if not ptxt or not ptxt.strip():
                continue
            enhanced_txt = enhance_prompt(client, ptxt, enhance_style) if use_enhance else None
            jobs.append(GenJob(idx=i, prompt=ptxt, enhanced_prompt=enhanced_txt, references=refs))

        if not jobs:
            st.warning("Please enter at least one prompt.")
        else:
            try:
                with st.spinner("Running jobs..."):
                    jobs = generate_multiple(client, jobs, size, int(seconds), model, concurrent_workers=int(workers))
                st.success("All jobs completed.")
                for job in sorted(jobs, key=lambda j: j.idx):
                    with st.expander(f"Result #{job.idx+1}"):
                        if job.result_path and job.result_path.exists():
                            st.video(str(job.result_path))
                            st.download_button("Download video", data=job.result_path.read_bytes(), file_name=job.result_path.name, mime="video/mp4", key=f"dl_multi_{job.idx}")
                        if job.enhanced_prompt:
                            st.markdown("**Enhanced prompt**")
                            st.code(job.enhanced_prompt)
            except Exception as e:
                st.exception(e)

# ---------- REMIX ----------
with tabs[2]:
    st.subheader("Remix Sequence")
    base_prompt = st.text_area("Base Shot Prompt", height=140, placeholder="Describe the initial shot...")
    remix_count = st.number_input("Number of remix steps", min_value=1, max_value=10, value=2, step=1)
    remix_prompts: List[str] = []
    for i in range(int(remix_count)):
        remix_prompts.append(st.text_input(f"Remix Prompt #{i+1}", key=f"remix_prompt_{i}", placeholder=f"Describe remix step {i+1}..."))

    r_videos = 1 + int(remix_count)
    r_cost = estimate_cost(r_videos, int(seconds), model)
    st.info(f"**Estimated cost:** ~${r_cost:.2f} • **Clips:** {r_videos}")

    if st.session_state.budget_enabled and (spent + r_cost) > st.session_state.budget_limit:
        st.warning(f"This remix run (~${r_cost:.2f}) may exceed your remaining budget.")

    rr_gen_refs_from_prompt = st.text_input("Optional: Generate reference image from a prompt (for base shot)")
    rr_ref_uploaded = st.file_uploader("Or upload reference image(s) for base shot", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="remix_refs")

    if client and st.button("Generate Remix Sequence"):
        if st.session_state.budget_enabled and (spent + r_cost) > st.session_state.budget_limit:
            st.error("Budget cap would be exceeded. Cannot proceed.")
            st.stop()

        if not base_prompt.strip():
            st.warning("Please enter a base prompt.")
        else:
            refs: List[Path] = []
            if rr_gen_refs_from_prompt.strip():
                out_img = make_reference_image(
                    client, rr_gen_refs_from_prompt,
                    st.session_state.output_dir / "refs", "remix_base_ref",
                    canvas_size=(canvas_w, canvas_h)
                )
                if out_img:
                    refs.append(out_img)
            if rr_ref_uploaded:
                refs_dir = ensure_dir(st.session_state.output_dir / "refs")
                for i, uf in enumerate(rr_ref_uploaded, start=1):
                    p = refs_dir / f"remix_upload_{i}.png"
                    write_bytes(p, uf.getvalue())
                    refs.append(p)

            single_ref_path = build_single_reference(
                refs, st.session_state.output_dir, (canvas_w, canvas_h),
                montage_layout=montage_layout,
                montage_cols=int(montage_cols),
                montage_rows=int(montage_rows),
                montage_padding=int(montage_padding),
            )
            refs = [single_ref_path] if single_ref_path else []

            if refs:
                st.image(str(refs[0]), caption="Reference used for base", use_container_width=True)

            base_final = enhance_prompt(client, base_prompt, enhance_style) if use_enhance else base_prompt
            remix_final = [enhance_prompt(client, rp, enhance_style) if use_enhance else rp for rp in remix_prompts]

            try:
                with st.spinner("Generating remix sequence..."):
                    outs = generate_and_remix(client, base_final, remix_final, size, int(seconds), refs, model)
                st.success("Remix sequence completed.")
                valid_paths = [p for p in outs if p and p.exists()]
                for i, p in enumerate(valid_paths):
                    with st.expander(f"Clip {i} ({p.name})"):
                        st.video(str(p))
                        st.download_button("Download video", data=p.read_bytes(), file_name=p.name, mime="video/mp4", key=f"dl_remix_{i}")
                if len(valid_paths) >= 2 and check_ffmpeg():
                    st.divider()
                    stitch_name = st.text_input("Stitched file name", value="sequence.mp4")
                    if st.button("Stitch all clips"):
                        outp = stitch_videos(valid_paths, st.session_state.output_dir / stitch_name)
                        if outp and outp.exists():
                            st.success(f"Stitched: {outp.name}")
                            st.video(str(outp))
                            st.download_button("Download stitched video", data=outp.read_bytes(), file_name=outp.name, mime="video/mp4", key="dl_stitched")
            except Exception as e:
                st.exception(e)

# ---------- BATCH ----------
with tabs[3]:
    st.subheader("Batch Queue")
    st.caption("Queue jobs and let the rate limiter handle pacing automatically.")

    if "batch_jobs" not in st.session_state:
        st.session_state.batch_jobs = []

    new_prompt = st.text_area("Add a prompt to the queue", height=100, key="batch_add_prompt")
    colQ1, colQ2 = st.columns([1, 1])
    with colQ1:
        if st.button("Add to queue"):
            if new_prompt.strip():
                st.session_state.batch_jobs.append({"prompt": new_prompt.strip(), "status": "queued", "path": None})
                st.success("Added to queue.")
    with colQ2:
        if st.button("Clear queue"):
            st.session_state.batch_jobs = []
            st.success("Cleared.")

    st.write("Shared references for all queued jobs (optional)")
    b_gen_ref_prompt = st.text_input("Generate a shared reference image from prompt (optional)", key="batch_ref_gen")
    b_ref_uploaded = st.file_uploader("Or upload reference image(s) shared by all batch jobs", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch_ref_upload")

    if st.session_state.batch_jobs:
        st.write("Current queue:")
        for i, job in enumerate(st.session_state.batch_jobs):
            st.write(f"{i+1}. **{job['status'].upper()}** — {job['prompt'][:80]}{'...' if len(job['prompt'])>80 else ''}")

    colBq1, _ = st.columns([1, 1])
    with colBq1:
        batch_enhance = st.checkbox("Enhance prompts in batch", value=False, key="batch_enhance")

    if client and st.button("Start queue run"):
        refs: List[Path] = []
        if b_gen_ref_prompt.strip():
            out_img = make_reference_image(
                client, b_gen_ref_prompt,
                st.session_state.output_dir / "refs", "batch_shared_ref",
                canvas_size=(canvas_w, canvas_h)
            )
            if out_img:
                refs.append(out_img)
        if b_ref_uploaded:
            refs_dir = ensure_dir(st.session_state.output_dir / "refs")
            for i, uf in enumerate(b_ref_uploaded, start=1):
                p = refs_dir / f"batch_upload_{i}.png"
                write_bytes(p, uf.getvalue())
                refs.append(p)

        single_ref_path = build_single_reference(
            refs, st.session_state.output_dir, (canvas_w, canvas_h),
            montage_layout=montage_layout,
            montage_cols=int(montage_cols),
            montage_rows=int(montage_rows),
            montage_padding=int(montage_padding),
        )
        refs = [single_ref_path] if single_ref_path else []

        if refs:
            st.image(str(refs[0]), caption="Shared reference used for batch", use_container_width=True)

        completed = 0
        for job in st.session_state.batch_jobs:
            if job["status"] not in ("queued", "error"):
                continue
            job["status"] = "running"
            st.write(f"Running: {job['prompt'][:80]}{'...' if len(job['prompt'])>80 else ''}")

            j_est = estimate_cost(1, int(seconds), model)
            if st.session_state.budget_enabled and (st.session_state.spent_usd + j_est) > st.session_state.budget_limit:
                job["status"] = "error"
                st.error("Budget cap reached. Halting queue.")
                break

            final_prompt = enhance_prompt(client, job["prompt"], enhance_style) if batch_enhance else job["prompt"]
            try:
                out_path = generate_single(client, final_prompt, size, int(seconds), refs, model)
                if out_path and out_path.exists():
                    job["status"] = "done"
                    job["path"] = out_path
                    st.success(f"Done: {out_path.name}")
                    completed += 1
                else:
                    job["status"] = "error"
                    st.error("Job failed (no output).")
            except Exception as e:
                job["status"] = "error"
                st.exception(e)

        st.success(f"Queue finished. Completed {completed} job(s).")

# ---------- ASSETS ----------
with tabs[4]:
    st.subheader("Session Assets")
    out_dir = st.session_state.output_dir
    videos = sorted(out_dir.glob("*.mp4"))
    images = sorted((out_dir / "refs").glob("*")) if (out_dir / "refs").exists() else []

    if videos:
        st.write("Videos")
        for v in videos:
            with st.expander(v.name):
                st.video(str(v))
                st.download_button("Download", data=v.read_bytes(), file_name=v.name, mime="video/mp4", key=f"dl_asset_{v.name}")
    else:
        st.info("No videos yet.")

    if images:
        st.write("Reference Images")
        st.image([str(x) for x in images], caption=[x.name for x in images])
        for img in images:
            st.download_button(f"Download {img.name}", data=img.read_bytes(), file_name=img.name, key=f"dl_img_{img.name}")
    else:
        st.info("No references yet.")
