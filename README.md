# Sora Streamlit Studio

*An end-to-end Streamlit app for creating watermark-free videos with OpenAI’s Sora API — featuring reference images, prompt enhancement, single/multiple/concurrent generation, remix sequences, a rate-limit aware batch queue, and budget guardrails with a live “remaining budget” meter. It also includes download buttons for generated assets, optional FFmpeg stitching for multi-clip sequences, and one-click zip export of your session outputs.*

**Heads-up:** The app is designed to be future-compatible. It uses a duration slider (1–60s) and a “Strict supported durations” toggle. When strict mode is on, your choice snaps to currently supported values (e.g., 4/8/12) to avoid API errors. You can turn strict mode off to try new durations as they become available.

-----

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots-optional)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Running on Streamlit Community Cloud](#running-on-streamlit-community-cloud)
- [Docker](#docker-optional)
- [App Tour](#app-tour)
  - [Sidebar](#sidebar)
  - [Single Tab](#single-tab)
  - [Multiple Tab](#multiple-tab)
  - [Remix Tab](#remix-tab)
  - [Batch Tab](#batch-tab-rate-limit-aware)
  - [Assets Tab](#assets-tab)
- [Cost, Budget & Limits](#cost-budget--limits)
- [Reference Images](#reference-images)
- [Prompt Enhancement](#prompt-enhancement)
- [FFmpeg Stitching](#ffmpeg-stitching)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Security Notes](#security-notes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Buy Me a Coffee (Crypto)](#buy-me-a-coffee-crypto)

-----

## Features

- **Web UI (Streamlit)** — zero-boilerplate local GUI for Sora video generation.
- **Reference images:**
  - Upload one or more images.
  - Or generate a reference image from a text prompt (uses an image model).
  - **Automatic montage combining** — upload multiple images and combine them into a single reference with configurable layouts (Auto, 1×N, N×1, 2×2, 3×2, 3×3, 3×N, Custom).
- **Prompt enhancement (optional)** with before/after display:
  - Styles: `director`, `pixar`, `clean`.
- **Single generation flow.**
- **Multiple generation flow with concurrency:**
  - Choose N jobs and workers.
  - Jobs run in parallel (threaded) and are still gated by a global rate limiter.
  - Row-wise, top-aligned prompt inputs for better UX.
- **Remix flow:**
  - Create a base clip, then apply N remix prompts sequentially.
  - Optional FFmpeg stitch into one MP4.
- **Batch queue (local)** with rate-limit aware scheduler:
  - Token-bucket limiter tied to your selected RPM tier.
  - Auto-pauses when tokens deplete and resumes after the 60-second refill window.
  - Optional enhancement for all prompts in the queue.
  - Shared reference images for the entire queue.
- **Budget guardrails:**
  - Set a session budget; see live spend meter and remaining budget.
  - Blocks new jobs that would exceed your cap.
- **Cost estimator (upfront):**
  - Based on model rates (defaults set in the code; update as needed).
- **Assets tab:**
  - Preview generated videos and references.
  - One-click download.
  - Zip session and clear outputs tools.

-----

## Requirements

- **Python 3.10+** (3.11 recommended)
- **OpenAI API key** with Sora access
- **FFmpeg (optional)** for stitching clips:
  - macOS: `brew install ffmpeg`
  - Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`
  - Windows: Install from <https://ffmpeg.org> and add to PATH

-----

## Quickstart

1. **Clone and install:**

```bash
git clone https://github.com/Dzechy/sora-streamlit-studio.git
cd sora-streamlit-studio

# Minimal deps
pip install --upgrade streamlit openai pillow
```

1. **Set your API key:**

```bash
export OPENAI_API_KEY=sk-...    # macOS/Linux
# PowerShell (Windows):
# setx OPENAI_API_KEY "sk-..."
```

1. **Run the app:**

```bash
streamlit run Sora_Streamlit_Studio.py
```

**Tip:** You can also paste your API key directly into the app’s sidebar.

-----

## Configuration

Most settings live in the sidebar at runtime. For static defaults, edit the constants at the top of `Sora_Streamlit_Studio.py`:

```python
PRICE_PER_SECOND = {"sora-2": 0.10, "sora-2-pro": 0.30}
CURRENT_SUPPORTED_DURATIONS = [4, 8, 12]   # informational hint for "Strict durations"
RPM_TIERS = {"Tier 1": 5, "Tier 2": 10, "Tier 3": 25, "Tier 4": 40, "Tier 5": 75}
```

If pricing, supported durations, or rate-limit tiers change, update them here.

-----

## Running on Streamlit Community Cloud

1. Push the repo to GitHub.
1. Create a new Streamlit app and target `Sora_Streamlit_Studio.py`.
1. **Set Secrets:**

- In your app dashboard → Settings → Secrets:

```toml
OPENAI_API_KEY = "sk-..."
```

1. Deploy. (You can also paste the key into the sidebar at runtime if you prefer.)

-----

## Docker (optional)

### Dockerfile

```dockerfile
FROM python:3.11-slim

# System deps for Pillow (optional) + ffmpeg (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Sora_Streamlit_Studio.py /app/

RUN pip install --no-cache-dir streamlit openai pillow

ENV OPENAI_API_KEY=""
EXPOSE 8501

CMD ["streamlit", "run", "Sora_Streamlit_Studio.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

**Build & run:**

```bash
docker build -t sora-studio .
docker run -it --rm -p 8501:8501 -e OPENAI_API_KEY=sk-... sora-studio
```

-----

## App Tour

### Sidebar

**API & Limits**

- **OpenAI API Key (masked)** — or use `OPENAI_API_KEY` env var.
- **Model:** `sora-2` or `sora-2-pro`.
- **Duration (seconds):** slider (1–60s).
- **Strict supported durations:** When enabled, your chosen value snaps to the nearest currently supported duration (listed beneath, e.g., 4/8/12) to avoid 400 errors. Turn it off to try new durations when the API updates.
- **Frame Size:** Presets or Custom (e.g., 720×1280).
- **Rate limit tier (RPM):** informs the token-bucket limiter that gates all API calls.

**Prompt Enhancement**

- Toggle on/off and pick a style: `director`, `pixar`, `clean`.

**Reference Montage**

- **Montage layout:** Choose from Auto, 1×N (vertical stack), N×1 (horizontal strip), 2×2 grid, 3×2 grid, 3×3 grid, 3×N (3 cols, rows as needed), or Custom.
- **Padding:** Set spacing between images in pixels.
- **Custom columns/rows:** For Custom layout, specify exact grid dimensions.

**Budget**

- Optional budget cap for the session, with live meter (spent / cap / remaining).
- Reset spend counter to zero the session’s totals.

**Outputs**

- Choose your output folder (session subfolder created automatically).
- New session folder, Zip session, Clear session.

-----

### Single tab

- Enter a video prompt.
- Optionally generate a reference image from text or **upload multiple reference images** to automatically combine them into a single montage.
- Optional prompt enhancement, with the enhanced version shown after generation.
- Estimated cost and API calls are shown before you run.
- On success, you’ll see a preview, download button, and (if used) the enhanced prompt.

-----

### Multiple tab

- Specify N generations and Concurrent workers.
- Provide N prompts using **row-wise, top-aligned prompt inputs** for easier editing.
- Optional shared reference image(s) (prompt-generated or uploaded, with automatic montage support).
- Optional enhancement per prompt; enhanced text appears with each result.
- Shows total estimated cost and total API calls up front.
- Global rate limiter still gates the calls to reduce 429s.

-----

### Remix tab

- Enter a **Base Shot Prompt** and a number of remix steps.
- Optional base reference images (prompt-generated or uploaded, with automatic montage support).
- Optional prompt enhancement (applies to base + each remix).
- Returns each clip with a download button.
- If FFmpeg is installed, you can stitch all the clips into one MP4.

-----

### Batch tab (rate-limit aware)

- Add any number of prompts to a local queue.
- When you **Start queue run:**
  - A token-bucket rate limiter paces API calls based on your RPM tier.
  - The scheduler auto-pauses when tokens run out and resumes when the minute window resets — no manual retry needed.
- You can enable prompt enhancement for the entire queue.
- Supply shared references for all batch jobs (generated or uploaded, with automatic montage support).
- The queue stops early if your budget cap would be exceeded by the next job.

-----

### Assets tab

- Shows all videos created in the current session folder and any reference images (uploaded or generated).
- One-click downloads.
- Works hand-in-hand with **Zip session** in the sidebar.

-----

## Cost, Budget & Limits

- **Cost estimator** uses per-second rates:
  - `sora-2`: $0.10/s
  - `sora-2-pro`: $0.30/s
- The app tracks spend after a clip successfully downloads (deduped by video id).
- **Budget cap:**
  - If enabled, the app warns when a planned run may exceed the remaining budget and blocks if it would.
- **Rate limits:**
  - The sidebar RPM setting configures a token-bucket limiter used by all API calls (enhancement, image generation, video create, remix).
  - This is advisory — pick the tier that fits your account to minimize 429s.

Update the constants at the top of `Sora_Streamlit_Studio.py` if pricing or rate-limit tiers change for your account.

-----

## Reference Images

You can either:

- **Upload** one or more images (PNG/JPG), or
- **Generate** a reference image from a short text prompt (uses an image model under the hood). The app saves it to your session’s `refs` folder and shows a preview.

**Multiple images are automatically combined** into a single reference montage using configurable layouts:

- **Auto:** Intelligently chooses grid layout based on image count
- **1×N / N×1:** Vertical stack or horizontal strip
- **2×2 / 3×2 / 3×3:** Fixed grid layouts
- **3×N:** 3 columns with rows as needed
- **Custom:** Specify exact columns and rows

When generating a video, the app passes the combined reference image to the Sora API.

-----

## Prompt Enhancement

An optional one-click step that rewrites your prompt for clarity and specificity. **Styles:**

- **director** — cinematic language, camera, lensing, lighting, movement, environment
- **pixar** — warm, character-focused, wardrobe + blocking details
- **clean** — concise without changing intent

If enhancement fails (e.g., quota or rate-limit), the app falls back to your original prompt and continues.

-----

## FFmpeg Stitching

The **Remix** tab can stitch multiple clips into one MP4 if FFmpeg is found on your PATH.  
Install FFmpeg (see [Requirements](#requirements)) and you’ll see a “Stitch all clips” button once multiple clips exist.

-----

## Troubleshooting

- **`insufficient_quota` / 429**
  - If it’s quota, add billing or use a key with access.
  - If it’s burst rate-limit, reduce **Concurrent workers** or pick a lower RPM tier in the sidebar.
- **Invalid seconds (400)**
  - Enable **Strict supported durations** or select a supported duration (e.g., 4/8/12).
- **Download failed**
  - Check that your key has Sora permissions and that the job finished.
  - Intermittent issues are retried gently; try again if needed.
- **FFmpeg not found**
  - Install FFmpeg and restart the app; the stitch controls will appear.
- **OpenAI SDK not installed**
  - `pip install openai` — also ensure `OPENAI_API_KEY` is set or paste it into the sidebar.
- **Pillow not installed**
  - `pip install pillow` — required for reference image montage combining and resizing.

-----

## Advanced Configuration

Open `Sora_Streamlit_Studio.py` and adjust:

```python
# Pricing (used by the cost estimator)
PRICE_PER_SECOND = {"sora-2": 0.10, "sora-2-pro": 0.30}

# Informational hint used by the "Strict supported durations" toggle
CURRENT_SUPPORTED_DURATIONS = [4, 8, 12]

# Rate-limit tiers for the token-bucket limiter
RPM_TIERS = {"Tier 1": 5, "Tier 2": 10, "Tier 3": 25, "Tier 4": 40, "Tier 5": 75}
```

You can also tweak:

- Default output root folder
- Default model list
- Retry counts / backoff behavior
- Thread pool workers for multi-generation
- Montage background color and padding defaults

-----

## Security Notes

- **Never commit your API keys to source control.**
- For Streamlit Cloud, store secrets in **Settings → Secrets**.
- Locally, prefer environment variables over hardcoding.

-----

## Roadmap

- Status dashboard (tokens remaining, next refill ETA)
- Per-job model/duration in Multi & Batch tabs
- Per-clip cost breakdown & CSV export
- Persistent job history across sessions
- Advanced montage options (custom backgrounds, borders, captions)

-----

## Contributing

PRs are welcome! Please:

1. Fork the repo.
1. Create a feature branch: `feat/<your-feature>`
1. Add tests or a test plan if applicable.
1. Submit a PR with a clear description and screenshots/GIFs.

-----

## License

MIT.

-----

## Buy Me a Coffee (Crypto)

If this saved you time (or a shoot 👀), you can fuel more dev with a small tip:

- **Bitcoin (BTC):** `bc1qpav33sclt900r2q4q9y6vuewftx22jx0xun0pe`
- **Ethereum (ETH):** `0x27d1FC34b431Aa9c80d5F9EE7464c03Eb1D15be0`
- **USDT (ETH Network):**
`0x27d1FC34b431Aa9c80d5F9EE7464c03Eb1D15be0`

Thanks a ton! 🙏

-----

**Happy creating ✨**
