# antigravity-mobile-architecture

Python simulation and `pytest` verification suite modeling the **Google Antigravity Mobile Architecture (Android Chrome PWA)** from the combined reference manual (*คู่มือฉบับรวมสมบูรณ์: การเปิดและใช้งาน Google Antigravity บนมือถือ Android*).

## Architecture Pillars Covered

1. **Part 1 — Chrome PWA & Play Store Hardware Lock** ([`pwa_client.py`](src/antigravity_mobile/pwa_client.py))
   - Google Play Store native APK is hardware-locked to **Googlebook OS** (`PlayStoreFilter`).
   - Standard Android smartphones run the official web app via **Google Chrome PWA** at `https://antigravity.google.com` with full-screen UI and **Service Worker** push notifications.
2. **Part 2 — 4-Step Mobile Setup & Pure Cloud Remote Project** ([`pwa_client.py`](src/antigravity_mobile/pwa_client.py), [`cloud_project.py`](src/antigravity_mobile/cloud_project.py))
   - Open Chrome (`antigravity.google.com`) -> Sign in with Personal Google Account (`Google AI Pro` / `Google AI Ultra`) -> Install to Home Screen (PWA) -> Create Remote Project.
   - Attaches **GitHub Repositories** and **Google Drive (e.g., 5TB)** via OAuth Connectors in **Resources** (rejects local filesystem mounts like `C:\` or `/home`).
3. **Part 3 — Non-Blocking Asynchronous Cloud Task Dispatch** ([`cloud_project.py`](src/antigravity_mobile/cloud_project.py))
   - Model tier specialization:
     - **Argon** — Planning & subtask breakdown (*Project Director / Lead Structural Architect*)
     - **Hydrogen** — Branch coding (*Production Engineer*)
     - **Claude Opus 5.5** — Code review & verification (*Independent Peer Reviewer / Checker*)
   - Runs in Google's **Isolated Cloud Sandbox** without requiring an open mobile browser or persistent WebSocket; fires **Push Notifications** when finished or `BLOCKED_FOR_USER_INPUT`.
4. **Part 4 — Windows 11 Remote Control Mode, Instance Switcher & `ChindaMT-4B` Runner** ([`remote_control.py`](src/antigravity_mobile/remote_control.py), [`chinda_mt.py`](src/antigravity_mobile/chinda_mt.py))
   - Enable on home PC via Antigravity Desktop App or `agy --remote-agent`.
   - Switch between **Remote Conversations (Pure Cloud)** and **Windows PC Workstation** via the Top Bar **Instance Switcher**.
   - Dispatches lightweight commands through **Web Relay** (no heavy RDP pixel streaming) to access local disk files and local workstation hardware (e.g., `128 GB RAM`, `NVIDIA 16 GB VRAM` for `ChindaMT-4B` or drone crack inspection).
   - Includes the full **`iapp/ChindaMT-4B` (Thai-English)** inference configuration (`TranslatorConfig`) and Hugging Face Transformers / PyTorch runner (`ChindaTranslator`) supporting `bfloat16`/`float16`, optional 4-bit quantization (`load_in_4bit=True`), auxiliary rules (`Rules:`), paragraph-aware document chunking, and left-padded micro-batched generation.
5. **Skill Pack & Closed-Loop Eval Harness (`modern-web-guidance` Principles)** ([`skill_guidance.py`](src/antigravity_mobile/skill_guidance.py))
   - Inspired by [`GoogleChrome/modern-web-guidance`](https://github.com/GoogleChrome/modern-web-guidance) (`npx modern-web-guidance@latest install`).
   - Models **Progressive Disclosure** (`SKILL.md` ~205-token index -> `search` -> `retrieve` -> `verify` -> disclosure), **Baseline Fallback Policy**, `legacy-scan` with `dropWhen` fallback filtering, and **Closed-Loop Eval Calibration** (Golden 100% / Base 0% + Unguided vs. Guided Uplift).

6. **Installed Skill Pack & Interactive Web Companion** ([`.agents/skills/modern-web-guidance`](.agents/skills/modern-web-guidance), [`web/index.html`](web/index.html))
   - Installed directly via `npx modern-web-guidance@latest install` (209 guides bundled for Antigravity and other coding agents).
   - Interactive web companion built using retrieved `modern-web-guidance` patterns (`dark-mode`, `responsive-table`, `light-dismiss-a-dialog`, `declarative-dialog-popover-control`).

## Project Structure

```text
├── .agents/skills/modern-web-guidance/  # Installed GoogleChrome/modern-web-guidance skill pack
├── config.py                            # ChindaMT-4B TranslatorConfig & Direction re-export
├── translator.py                        # ChindaMT-4B ChindaTranslator re-export
├── main.py                              # Standalone ChindaMT-4B Thai-English CLI demo
├── requirements.txt                     # PyTorch / Hugging Face Transformers GPU dependencies
├── docs/
│   └── architecture_summary.md
├── src/
│   └── antigravity_mobile/
│       ├── __init__.py
│       ├── __main__.py
│       ├── chinda_mt.py                 # ChindaMT-4B config, prompt builder, chunker & runner
│       ├── models.py
│       ├── pwa_client.py
│       ├── cloud_project.py
│       ├── remote_control.py
│       ├── skill_guidance.py
│       └── simulator.py
├── web/
│   └── index.html
└── tests/
    ├── test_part1_pwa_and_playstore.py
    ├── test_part2_setup_and_resources.py
    ├── test_part3_non_blocking_dispatch.py
    ├── test_part4_remote_control_switcher.py
    ├── test_chinda_mt.py
    ├── test_modern_web_guidance.py
    ├── test_web_companion.py
    └── test_end_to_end_workflows.py
```

## Quick Start

Create a virtual environment, install dev tools (`pytest`, `ruff`, `mypy`), run checks and tests, and execute the simulation:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src tests
pytest
python -m antigravity_mobile
```

### Running `ChindaMT-4B` (Thai-English) on a GPU Workstation (16 GB VRAM)

Install the GPU inference dependencies (`torch>=2.1`, `transformers>=5.2`, `accelerate>=0.27`) and run `main.py` or `chinda-mt`:

```bash
uv pip install -e ".[chinda]"
# or: pip install -r requirements.txt
python main.py
```

## License

Licensed under the [MIT License](LICENSE).
