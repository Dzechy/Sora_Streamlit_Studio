
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Sora App (patched)
============================
See docstring in previous attempt for details.
"""
import os
import io
import time
import base64
import tempfile
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Any, Dict

import streamlit as st

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

try:
    from openai import OpenAI
    from openai import APIStatusError  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
    APIStatusError = Exception

APP_NAME = "Sora Streamlit Studio"
DEFAULT_OUTPUT_DIR = Path("outputs")
ALLOWED_DURATIONS = [4, 8, 12]

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

def init_openai_client(api_key: Optional[str]) -> Optional[OpenAI]:
    if OpenAI is None:
        st.error("OpenAI SDK not installed. Try: `pip install openai`")
        return None
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        st.warning("Provide an OpenAI API key in the sidebar to enable generation.")
        return None
    os.environ["OPENAI_API_KEY"] = key
    try:
        client = OpenAI()
        return client
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
    quota = "insufficient_quota" in s or "quota" in s.lower()
    return {"status": status, "is_quota": quota, "text": s}

def with_retries(func: Callable[[], Any], attempts: int = 3, base_delay: float = 1.5, max_delay: float = 6.0) -> Any:
    last_exc = None
    for i in range(attempts):
        try:
            return func()
        except Exception as e:
            info = _classify_error(e)
            if info["is_quota"] or info["status"] == 429:
                if i < attempts - 1 and not info["is_quota"]:
                    time.sleep(min(max_delay, base_delay * (2 ** i)))
                    last_exc = e
                    continue
                st.error("Rate-limit or quota error from API.\n\n"
                         "- If it's **quota**, check your billing/plan.\n"
                         "- If it's a **burst rate limit**, lower concurrency or try again.\n\n"
                         f"Details: {info['text']}")
                raise
            if info["status"] and 500 <= info["status"] < 600:
                time.sleep(min(max_delay, base_delay * (2 ** i)))
                last_exc = e
                continue
            last_exc = e
            break
    if last_exc:
        raise last_exc

def poll_and_download_video(client: OpenAI, video, out_dir: Path, name_prefix: str = "video") -> Optional[Path]:
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
        return None
    ph.success("Completed. Downloading...")
    try:
        vid_id = getattr(video, "id", "sora_video")
        content = client.videos.download_content(vid_id, variant="video")
        out_path = out_dir / f"{name_prefix}_{vid_id}.mp4"
        content.write_to_file(out_path)
        ph.success(f"Saved: {out_path.name}")
        return out_path
    except Exception as e:
        ph.error(f"Download failed: {e}")
        return None

def enhance_prompt(client: OpenAI, prompt: str, style: str = "director", retries: int = 2) -> str:
    TEMPLATE_MAP = {
        "director": "You are a sharp film director. Rewrite the following video prompt for clarity, specificity, and cinematic detail (camera, lighting, lens, movement, environment). Return only the revised prompt.",
        "clean": "Rewrite clearly and concisely without losing meaning. Return only the revised text.",
        "pixar": "Rewrite as a warm, detailed Pixar-style prompt focusing on character, wardrobe, lensing, lighting, and blocking. Return only the revised prompt.",
    }
    ins = TEMPLATE_MAP.get(style, TEMPLATE_MAP["director"])
    def _call():
        resp = client.responses.create(model="gpt-5", input=prompt, instructions=ins)
        return getattr(resp, "output_text", None) or str(resp)
    try:
        return with_retries(_call, attempts=max(1, retries))
    except Exception as e:
        st.warning(f"Enhancement failed, using original prompt. ({e})")
        return prompt

def make_reference_image(client: OpenAI, text: str, out_dir: Path, name: str) -> Optional[Path]:
    def _call():
        return client.responses.create(model="gpt-image-1", input=text)
    try:
        resp = with_retries(_call, attempts=2)
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

@dataclass
class GenJob:
    idx: int
    prompt: str
    enhanced_prompt: Optional[str] = None
    references: List[Path] = None  # type: ignore
    result_path: Optional[Path] = None

def _video_create(client: OpenAI, kwargs: Dict[str, Any]):
    def _call():
        return client.videos.create(**kwargs)
    return with_retries(_call, attempts=2)

def generate_single(client: OpenAI, prompt: str, size: str, seconds: int, references: List[Path] | None, model: str) -> Optional[Path]:
    if seconds not in [4,8,12]:
        seconds = min([4,8,12], key=lambda x: abs(x - seconds))
        st.warning(f"Adjusted unsupported duration to {seconds}s.")
    kwargs = dict(model=model, prompt=prompt, size=size, seconds=str(seconds))
    if references:
        kwargs["input_reference"] = references[0]  # type: ignore
    vid = _video_create(client, kwargs)
    return poll_and_download_video(client, vid, st.session_state.output_dir, "single")

def generate_and_remix(client: OpenAI, base_prompt: str, remix_prompts: List[str], size: str, seconds: int, references: List[Path] | None, model: str) -> List[Optional[Path]]:
    if seconds not in [4,8,12]:
        seconds = min([4,8,12], key=lambda x: abs(x - seconds))
        st.warning(f"Adjusted unsupported duration to {seconds}s.")
    outputs: List[Optional[Path]] = []
    base_kwargs = dict(model=model, prompt=base_prompt, size=size, seconds=str(seconds))
    if references:
        base_kwargs["input_reference"] = references[0]  # type: ignore
    base = _video_create(client, base_kwargs)
    base_path = poll_and_download_video(client, base, st.session_state.output_dir, "remix_base")
    outputs.append(base_path)
    current = base
    for i, rprompt in enumerate(remix_prompts, start=1):
        st.write(f"Remix step {i}…")
        def _call():
            return client.videos.remix(video_id=current.id, prompt=rprompt)
        remix_job = with_retries(_call, attempts=2)
        out = poll_and_download_video(client, remix_job, st.session_state.output_dir, f"remix_{i}")
        outputs.append(out)
        current = remix_job
    return outputs

def generate_multiple(client: OpenAI, jobs: List[GenJob], size: str, seconds: int, model: str, concurrent_workers: int = 2) -> List[GenJob]:
    if seconds not in [4,8,12]:
        seconds = min([4,8,12], key=lambda x: abs(x - seconds))
        st.warning(f"Adjusted unsupported duration to {seconds}s.")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def worker(job: GenJob):
        kwargs = dict(model=model, prompt=(job.enhanced_prompt or job.prompt), size=size, seconds=str(seconds))
        if job.references:
            kwargs["input_reference"] = job.references[0]  # type: ignore
        try:
            vid = _video_create(client, kwargs)
            out = poll_and_download_video(client, vid, st.session_state.output_dir, f"multi_{job.idx+1}")
            return job.idx, out
        except Exception as e:
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
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", out_path.as_posix(), "-y"],
            check=True
        )
        return out_path
    except Exception as e:
        st.error(f"Stitch failed: {e}")
        return None

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Web UI for Sora video generation with prompt enhancement, references, multi-gen concurrency, and remix.")

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = st.selectbox("Sora Model", ["sora-2", "sora-2-pro"], index=0)
    seconds = st.selectbox("Duration (seconds)", options=[4,8,12], index=0, help="Sora currently accepts 4, 8, or 12.")
    size_choice = st.selectbox("Frame Size", ["720x1280 (9:16)", "1024x1792 (9:16)", "1280x720 (16:9)", "1920x1080 (16:9)", "Custom"], index=0)
    if size_choice == "Custom":
        size = st.text_input("Custom size (e.g., 720x1280)", "720x1280")
    else:
        size = size_choice.split()[0]
    st.subheader("Prompt Enhancement")
    use_enhance = st.checkbox("Enhance prompts before generating", value=False)
    enhance_style = st.selectbox("Enhancement style", ["director", "pixar", "clean"], index=0)
    st.subheader("Outputs")
    out_root = st.text_input("Output folder", value=str(DEFAULT_OUTPUT_DIR))
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = ensure_dir(Path(out_root) / time.strftime("%Y%m%d_%H%M%S"))
    colX, colY, colZ = st.columns([1,1,1])
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
tabs = st.tabs(["Single", "Multiple", "Remix", "Assets"])

with tabs[0]:
    st.subheader("Single Generation")
    colA, colB = st.columns([1,1], gap="large")
    with colA:
        user_prompt = st.text_area("Prompt", height=150, placeholder="Describe your video...")
        gen_refs_from_prompt = st.text_input("Optional: Generate reference image from a prompt (leave blank to skip)")
        ref_uploaded = st.file_uploader("Or upload reference image(s)", type=["png","jpg","jpeg"], accept_multiple_files=True)
    with colB:
        st.write("Preview & Actions")
        if client and st.button("Generate (Single)"):
            refs: List[Path] = []
            if gen_refs_from_prompt.strip():
                out_img = make_reference_image(client, gen_refs_from_prompt, st.session_state.output_dir / "refs", "ref_1")
                if out_img: refs.append(out_img)
                if out_img: st.image(str(out_img), caption="Generated reference", use_container_width=True)
            if ref_uploaded:
                refs_dir = ensure_dir(st.session_state.output_dir / "refs")
                for i, uf in enumerate(ref_uploaded, start=1):
                    p = refs_dir / f"upload_{i}.png"
                    write_bytes(p, uf.getvalue())
                    refs.append(p)
                if refs:
                    st.success(f"Loaded {len(refs)} reference image(s).")
                    st.image([str(x) for x in refs], caption=[x.name for x in refs])
            final_prompt = user_prompt
            enhanced_txt = None
            if use_enhance and user_prompt.strip():
                enhanced_txt = enhance_prompt(client, user_prompt, enhance_style, retries=2)
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

with tabs[1]:
    st.subheader("Multiple / Concurrent Generation")
    n = st.number_input("Number of generations", min_value=2, max_value=12, value=3, step=1)
    workers = st.number_input("Concurrent workers", min_value=1, max_value=12, value=2, step=1)
    st.caption("We run up to 'Concurrent workers' jobs in parallel; if you hit rate limits, lower this value.")
    st.write("Prompts")
    multi_prompts: List[str] = []
    cols = st.columns(2)
    for i in range(int(n)):
        with cols[i % 2]:
            p = st.text_area(f"Prompt #{i+1}", height=120, key=f"multi_prompt_{i}")
            multi_prompts.append(p)
    st.write("References (optional)")
    m_gen_ref_prompt = st.text_input("Generate a shared reference image from prompt (optional)", key="multi_ref_gen")
    m_ref_uploaded = st.file_uploader("Or upload reference image(s) shared by all jobs", type=["png","jpg","jpeg"], accept_multiple_files=True, key="multi_ref_upload")
    if client and st.button("Generate All"):
        refs: List[Path] = []
        if m_gen_ref_prompt.strip():
            out_img = make_reference_image(client, m_gen_ref_prompt, st.session_state.output_dir / "refs", "shared_ref")
            if out_img: refs.append(out_img)
            if out_img: st.image(str(out_img), caption="Generated shared reference", use_container_width=True)
        if m_ref_uploaded:
            refs_dir = ensure_dir(st.session_state.output_dir / "refs")
            for i, uf in enumerate(m_ref_uploaded, start=1):
                p = refs_dir / f"shared_upload_{i}.png"
                write_bytes(p, uf.getvalue())
                refs.append(p)
        jobs: List[GenJob] = []
        for i, p in enumerate(multi_prompts):
            if not p.strip():
                continue
            enhanced_txt = enhance_prompt(client, p, enhance_style, retries=2) if use_enhance else None
            jobs.append(GenJob(idx=i, prompt=p, enhanced_prompt=enhanced_txt, references=refs))
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

with tabs[2]:
    st.subheader("Remix Sequence")
    base_prompt = st.text_area("Base Shot Prompt", height=140, placeholder="Describe the initial shot...")
    remix_count = st.number_input("Number of remix steps", min_value=1, max_value=10, value=2, step=1)
    remix_prompts: List[str] = []
    for i in range(int(remix_count)):
        remix_prompts.append(st.text_input(f"Remix Prompt #{i+1}", key=f"remix_prompt_{i}", placeholder=f"Describe remix step {i+1}..."))
    rr_gen_refs_from_prompt = st.text_input("Optional: Generate reference image from a prompt (for base shot)")
    rr_ref_uploaded = st.file_uploader("Or upload reference image(s) for base shot", type=["png","jpg","jpeg"], accept_multiple_files=True, key="remix_refs")
    if client and st.button("Generate Remix Sequence"):
        if not base_prompt.strip():
            st.warning("Please enter a base prompt.")
        else:
            refs: List[Path] = []
            if rr_gen_refs_from_prompt.strip():
                out_img = make_reference_image(client, rr_gen_refs_from_prompt, st.session_state.output_dir / "refs", "remix_base_ref")
                if out_img: refs.append(out_img)
                if out_img: st.image(str(out_img), caption="Generated reference for base", use_container_width=True)
            if rr_ref_uploaded:
                refs_dir = ensure_dir(st.session_state.output_dir / "refs")
                for i, uf in enumerate(rr_ref_uploaded, start=1):
                    p = refs_dir / f"remix_upload_{i}.png"
                    write_bytes(p, uf.getvalue())
                    refs.append(p)
            base_final = enhance_prompt(client, base_prompt, enhance_style, retries=2) if use_enhance else base_prompt
            remix_final = [enhance_prompt(client, rp, enhance_style, retries=2) if use_enhance else rp for rp in remix_prompts]
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

with tabs[3]:
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
