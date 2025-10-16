#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sora Streamlit Studio — Budget, Rate-Limits, Batch Queue, Duration Slider
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

# Optional Pillow for saving generated refs
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

# OpenAI client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

APP_NAME = "Sora Streamlit Studio"
DEFAULT_OUTPUT_DIR = Path("outputs")

# Pricing used for the estimator (update if your plan changes)
PRICE_PER_SECOND = {
    "sora-2": 0.10,
    "sora-2-pro": 0.30,
}

# Informational hint for the "Strict supported durations" toggle
CURRENT_SUPPORTED_DURATIONS = [4, 8, 12]

# Example RPM tiers for a token-bucket limiter
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

def pil_save_image(b64_png: str, out: Path, size: Tuple[int, int] | None = (720, 1280)) -> Optional[Path]:
    try:
        raw = base64.b64decode(b64_png)
        if Image is None:
            return write_bytes(out, raw)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if size:
            img = img.resize(size)
        ensure_dir(out.parent)
        img.save(out)
        return out
    except Exception:
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
    """Token-bucket limiter keyed to RPM. Thread-safe; blocks until refill when empty."""
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
    if OpenAI is None:
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

def with_retries(func: Callable[[], Any], attempts: int = 2, base_delay: float = 1.5, max_delay: float = 6.0) -> Any:
    last_exc = None
    for i in range(attempts):
        try:
            return func()
        except Exception as e:
            info = _classify_error(e)
            # 429 or quota → retry once if not quota; otherwise surface
            if info["is_quota"] or info["status"] == 429:
                if i < attempts - 1 and not info["is_quota"]:
                    time.sleep(min(max_delay, base_delay * (2 ** i)))
                    last_exc = e
                    continue
                st.error("Rate-limit or quota error from API. Details: " + info["text"])
                raise
            # transient 5xx → retry
            if info["status"] and 500 <= info["status"] < 600:
                time.sleep(min(max_delay, base_delay * (2 ** i)))
                last_exc = e
                continue
            last_exc = e
            break
    if last_exc:
        raise last_exc

def call_api_with_rl(fn: Callable[[], Any]) -> Any:
    rl_acquire(1)
    return with_retries(fn, attempts=2)

# -------------------------
# Poll & Download, Prompt Enhance, Reference Image
# -------------------------

def poll_and_download_video(client: OpenAI, video, out_dir: Path, name_prefix: str, seconds: int, model: str) -> Tuple[Optional[Path], Optional[str]]:
    status = getattr(video, "status", None)
    progress = getattr(video, "progress", 0) or 0
    ph = st.empty()
    bar = st.progress(0)

    while status not in ("completed", "failed", "cancelled"):
        progress = getattr(video, "progress", progress) or progress
        status = getattr(video, "status", status)
        bar.progress(int(max(0, min(100, progress))))
        ph.info(f"Status: {status} • {progress}%")
        time.sleep(1.0)
        try:
            status = getattr(video, "status", status)
        except Exception:
            pass

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
        resp = call_api_with_rl(lambda: OpenAI().responses.create(model="gpt-5", input=prompt, instructions=ins))
        return getattr(resp, "output_text", None) or str(resp)
    except Exception as e:
        st.warning(f"Enhancement failed, using original prompt. ({e})")
        return prompt

def make_reference_image(client: OpenAI, text: str, out_dir: Path, name: str) -> Optional[Path]:
    try:
        resp = call_api_with_rl(lambda: OpenAI().responses.create(model="gpt-image-1", input=text))
        b64 = None
        for path in [
            ("output", 0, "content", 0, "image_base64"),
            ("content", 0, "image_base64"),
        ]:
            try:
                r = resp
                for key in path:
                    r = r[key] if isinstance(key, int) else getattr(r, key)
                if isinstance(r, str):
                    b64 = r
                    break
            except Exception:
                continue
        if not b64:
            st.error("Could not extract image from API response.")
            return None
        out_path = out_dir / f"{name}.png"
        return pil_save_image(b64, out_path, (720, 1280))
    except Exception as e:
        st.error(f"Reference image generation failed: {e}")
        return None

# -------------------------
# Data models & Flows
# -------------------------

@dataclass
class GenJob:
    idx: int
    prompt: str
    enhanced_prompt: Optional[str] = None
    references: List[Path] = None  # type: ignore
    result_path: Optional[Path] = None

def _video_create(client: OpenAI, kwargs: Dict[str, Any]):
    return call_api_with_rl(lambda: client.videos.create(**kwargs))

def generate_single(client: OpenAI, prompt: str, size: str, seconds: int, references: List[Path] | None, model: str) -> Optional[Path]:
    kwargs = dict(model=model, prompt=prompt, size=size, seconds=str(seconds))
    if references:
        kwargs["input_reference"] = references[0]  # type: ignore
    vid = _video_create(client, kwargs)
    path, _ = poll_and_download_video(client, vid, st.session_state.output_dir, "single", seconds, model)
    return path

def generate_and_remix(client: OpenAI, base_prompt: str, remix_prompts: List[str], size: str, seconds: int, references: List[Path] | None, model: str) -> List[Optional[Path]]:
    outputs: List[Optional[Path]] = []
    base_kwargs = dict(model=model, prompt=base_prompt, size=size, seconds=str(seconds))
    if references:
        base_kwargs["input_reference"] = references[0]  # type: ignore
    base = _video_create(client, base_kwargs)
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

def generate_multiple(client: OpenAI, jobs: List[GenJob], size: str, seconds: int, model: str, concurrent_workers: int = 2) -> List[GenJob]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def worker(job: GenJob):
        kwargs = dict(model=model, prompt=(job.enhanced_prompt or job.prompt), size=size, seconds=str(seconds))
        if job.references:
            kwargs["input_reference"] = job.references[0]  # type: ignore
        try:
            vid = _video_create(client, kwargs)
            out_path, _ = poll_and_download_video(client, vid, st.session_state.output_dir, f"multi_{job.idx+1}", seconds, model)
            return job.idx, out_path
        except Exception:
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

# -------------------------
# UI Helper — Multiple tab
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
    """Render N prompt textareas in rows of `cols_per_row`, labeled Prompt #1..#N.
    Uses top-aligned columns (if supported) to avoid uneven spacing."""
    prompts: List[str] = []

    # Detect Streamlit support for vertical_alignment (newer versions)
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
                # Keep an empty cell so row heights match
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

# Sidebar
with st.sidebar:
    st.header("API & Limits")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = st.selectbox("Sora Model", ["sora-2", "sora-2-pro"], index=0)

    # Duration slider + strict snapping
    seconds_slider = st.slider("Duration (seconds)", min_value=1, max_value=60, value=4, step=1)
    strict_durations = st.checkbox("Strict supported durations", value=True, help="When on, snaps to currently-supported values to avoid API errors.")
    st.caption(f"Currently supported durations: {', '.join(map(str, CURRENT_SUPPORTED_DURATIONS))} seconds.")
    seconds = min(CURRENT_SUPPORTED_DURATIONS, key=lambda x: abs(x - seconds_slider)) if strict_durations else seconds_slider

    size_choice = st.selectbox("Frame Size", ["720x1280 (portrait)", "1280x720 (landscape)", "Custom"], index=0)
    size = st.text_input("Custom size (e.g., 720x1280)", "720x1280") if size_choice == "Custom" else size_choice.split()[0]

    # Rate-limit tier → token bucket
    tier = st.selectbox("Rate limit tier", list(RPM_TIERS.keys()), index=1)
    rpm = RPM_TIERS[tier]
    if "rate_limiter" not in st.session_state or getattr(st.session_state.get("rate_limiter"), "rpm", None) != rpm:
        st.session_state["rate_limiter"] = RateLimiter(rpm)

    # Prompt enhancement
    st.subheader("Prompt Enhancement")
    use_enhance = st.checkbox("Enhance prompts before generating", value=False)
    enhance_style = st.selectbox("Enhancement style", ["director", "pixar", "clean"], index=0)

    # Budget
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

    # Outputs
    st.subheader("Outputs")
    out_root = st.text_input("Output folder", value=str(DEFAULT_OUTPUT_DIR))
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = ensure_dir(Path(out_root) / time.strftime("%Y%m%d_%H%M%S"))

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

# Init API client
client = init_openai_client(api_key_input)

# Tabs
tabs = st.tabs(["Single", "Multiple", "Remix", "Batch", "Assets"])

# ---------- SINGLE ----------
with tabs[0]:
    st.subheader("Single Generation")
    colA, colB = st.columns([1, 1], gap="large")
    with colA:
        user_prompt = st.text_area("Prompt", height=150, placeholder="Describe your video...")
        gen_refs_from_prompt = st.text_input("Optional: Generate reference image from a prompt (leave blank to skip)")
        ref_uploaded = st.file_uploader("Or upload reference image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        est_cost = estimate_cost(1, seconds, model)
        st.info(f"**Estimated cost:** ~${est_cost:.2f} • **API calls:** {1 + (1 if use_enhance else 0)}")

        if st.session_state.budget_enabled and (spent + est_cost) > st.session_state.budget_limit:
            st.warning(f"This job (~${est_cost:.2f}) may exceed your remaining budget.")

    with colB:
        st.write("Preview & Actions")
        if client and st.button("Generate (Single)"):
            if st.session_state.budget_enabled and (spent + est_cost) > st.session_state.budget_limit:
                st.stop()

            # Build references
            refs: List[Path] = []
            if gen_refs_from_prompt.strip():
                out_img = make_reference_image(client, gen_refs_from_prompt, st.session_state.output_dir / "refs", "ref_1")
                if out_img:
                    refs.append(out_img)
                    st.image(str(out_img), caption="Generated reference", use_container_width=True)
            if ref_uploaded:
                refs_dir = ensure_dir(st.session_state.output_dir / "refs")
                for i, uf in enumerate(ref_uploaded, start=1):
                    p = refs_dir / f"upload_{i}.png"
                    write_bytes(p, uf.getvalue())
                    refs.append(p)
                if refs:
                    st.success(f"Loaded {len(refs)} reference image(s).")
                    st.image([str(x) for x in refs], caption=[x.name for x in refs])

            # Enhance
            final_prompt = user_prompt
            enhanced_txt = None
            if use_enhance and user_prompt.strip():
                enhanced_txt = enhance_prompt(client, user_prompt, enhance_style)
                final_prompt = enhanced_txt

            if not user_prompt.strip() and not gen_refs_from_prompt.strip():
                st.warning("Please provide at least a video prompt, or generate a reference image prompt.")
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
    st.caption("We run up to 'Concurrent workers' jobs in parallel; rate limiter still gates calls globally.")

    m_est_cost = estimate_cost(int(n), seconds, model)
    m_calls_per_job = 1 + (1 if use_enhance else 0)
    m_total_calls = int(n) * m_calls_per_job
    st.info(f"**Estimated cost:** ~${m_est_cost:.2f} • **Total API calls:** ~{m_total_calls}")

    if st.session_state.budget_enabled and (spent + m_est_cost) > st.session_state.budget_limit:
        st.warning(f"This run (~${m_est_cost:.2f}) may exceed your remaining budget.")

    st.write("Prompts")
    # FIRST FIX: row-wise, top-aligned prompt inputs with stable keys
    multi_prompts: List[str] = render_multi_prompt_inputs(int(n))

    st.write("References (optional)")
    m_gen_ref_prompt = st.text_input("Generate a shared reference image from prompt (optional)", key="multi_ref_gen")
    m_ref_uploaded = st.file_uploader("Or upload reference image(s) shared by all jobs", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="multi_ref_upload")

    if client and st.button("Generate All"):
        if st.session_state.budget_enabled and (spent + m_est_cost) > st.session_state.budget_limit:
            st.stop()

        # Shared references
        refs: List[Path] = []
        if m_gen_ref_prompt.strip():
            out_img = make_reference_image(client, m_gen_ref_prompt, st.session_state.output_dir / "refs", "shared_ref")
            if out_img:
                refs.append(out_img)
                st.image(str(out_img), caption="Generated shared reference", use_container_width=True)
        if m_ref_uploaded:
            refs_dir = ensure_dir(st.session_state.output_dir / "refs")
            for i, uf in enumerate(m_ref_uploaded, start=1):
                p = refs_dir / f"shared_upload_{i}.png"
                write_bytes(p, uf.getvalue())
                refs.append(p)

        # Build jobs
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
                            st.download_button("Download video", data=job.result_path.read_bytes(), file_name=job.result_path.name, mime="video/mp4")
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
    r_cost = estimate_cost(r_videos, seconds, model)
    st.info(f"**Estimated cost:** ~${r_cost:.2f} • **Clips:** {r_videos}")

    if st.session_state.budget_enabled and (spent + r_cost) > st.session_state.budget_limit:
        st.warning(f"This remix run (~${r_cost:.2f}) may exceed your remaining budget.")

    rr_gen_refs_from_prompt = st.text_input("Optional: Generate reference image from a prompt (for base shot)")
    rr_ref_uploaded = st.file_uploader("Or upload reference image(s) for base shot", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="remix_refs")

    if client and st.button("Generate Remix Sequence"):
        if st.session_state.budget_enabled and (spent + r_cost) > st.session_state.budget_limit:
            st.stop()

        if not base_prompt.strip():
            st.warning("Please enter a base prompt.")
        else:
            refs: List[Path] = []
            if rr_gen_refs_from_prompt.strip():
                out_img = make_reference_image(client, rr_gen_refs_from_prompt, st.session_state.output_dir / "refs", "remix_base_ref")
                if out_img:
                    refs.append(out_img)
                    st.image(str(out_img), caption="Generated reference for base", use_container_width=True)
            if rr_ref_uploaded:
                refs_dir = ensure_dir(st.session_state.output_dir / "refs")
                for i, uf in enumerate(rr_ref_uploaded, start=1):
                    p = refs_dir / f"remix_upload_{i}.png"
                    write_bytes(p, uf.getvalue())
                    refs.append(p)

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
                        st.download_button("Download video", data=p.read_bytes(), file_name=p.name, mime="video/mp4")
                if len(valid_paths) >= 2 and check_ffmpeg():
                    st.divider()
                    stitch_name = st.text_input("Stitched file name", value="sequence.mp4")
                    if st.button("Stitch all clips"):
                        outp = stitch_videos(valid_paths, st.session_state.output_dir / stitch_name)
                        if outp and outp.exists():
                            st.success(f"Stitched: {outp.name}")
                            st.video(str(outp))
                            st.download_button("Download stitched video", data=outp.read_bytes(), file_name=outp.name, mime="video/mp4")
            except Exception as e:
                st.exception(e)

# ---------- BATCH (rate-limit aware) ----------
with tabs[3]:
    st.subheader("Batch Queue (Rate-limit aware)")
    st.caption("Queue jobs and let the token-bucket limiter handle pacing (auto-pauses when tokens deplete, resumes after refill).")

    if "batch_jobs" not in st.session_state:
        st.session_state.batch_jobs = []  # {'prompt': str, 'status': 'queued|running|done|error', 'path': Optional[Path]}

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

    # Shared references for batch
    st.write("Shared references for all queued jobs (optional)")
    b_gen_ref_prompt = st.text_input("Generate a shared reference image from prompt (optional)", key="batch_ref_gen")
    b_ref_uploaded = st.file_uploader("Or upload reference image(s) shared by all batch jobs", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch_ref_upload")

    if st.session_state.batch_jobs:
        st.write("Current queue:")
        for i, job in enumerate(st.session_state.batch_jobs):
            st.write(f"{i+1}. **{job['status'].upper()}** — {job['prompt'][:80]}{'...' if len(job['prompt'])>80 else ''}")

    colBq1, colBq2 = st.columns([1, 1])
    with colBq1:
        batch_enhance = st.checkbox("Enhance prompts in batch", value=False, key="batch_enhance")
    with colBq2:
        batch_workers = st.number_input("Workers (internally; limiter still gates calls)", min_value=1, max_value=8, value=1, step=1)

    if client and st.button("Start queue run"):
        # Build shared refs
        refs: List[Path] = []
        if b_gen_ref_prompt.strip():
            out_img = make_reference_image(client, b_gen_ref_prompt, st.session_state.output_dir / "refs", "batch_shared_ref")
            if out_img:
                refs.append(out_img)
                st.image(str(out_img), caption="Generated shared reference", use_container_width=True)
        if b_ref_uploaded:
            refs_dir = ensure_dir(st.session_state.output_dir / "refs")
            for i, uf in enumerate(b_ref_uploaded, start=1):
                p = refs_dir / f"batch_upload_{i}.png"
                write_bytes(p, uf.getvalue())
                refs.append(p)

        completed = 0
        for job in st.session_state.batch_jobs:
            if job["status"] not in ("queued", "error"):
                continue
            job["status"] = "running"
            st.write(f"Running: {job['prompt'][:80]}{'...' if len(job['prompt'])>80 else ''}")

            # Budget check per job
            j_est = estimate_cost(1, seconds, model)
            if st.session_state.budget_enabled and (st.session_state.spent_usd + j_est) > st.session_state.budget_limit:
                job["status"] = "error"
                st.error("Budget cap reached. Halting queue.")
                break

            # Enhance (if opted)
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
                st.download_button("Download", data=v.read_bytes(), file_name=v.name, mime="video/mp4")
    else:
        st.info("No videos yet.")

    if images:
        st.write("Reference Images")
        st.image([str(x) for x in images], caption=[x.name for x in images])
        for img in images:
            st.download_button(f"Download {img.name}", data=img.read_bytes(), file_name=img.name)
    else:
        st.info("No references yet.")
