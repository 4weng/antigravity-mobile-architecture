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
4. **Part 4 — Windows 11 Remote Control Mode & Instance Switcher** ([`remote_control.py`](src/antigravity_mobile/remote_control.py))
   - Enable on home PC via Antigravity Desktop App or `agy --remote-agent`.
   - Switch between **Remote Conversations (Pure Cloud)** and **Windows PC Workstation** via the Top Bar **Instance Switcher**.
   - Dispatches lightweight commands through **Web Relay** (no heavy RDP pixel streaming) to access local disk files and local workstation hardware (e.g., `128 GB RAM`, `NVIDIA 16 GB VRAM` for `ChindaMT-4B` or drone crack inspection).

## Project Structure

```text
├── docs/
│   └── architecture_summary.md
├── src/
│   └── antigravity_mobile/
│       ├── __init__.py
│       ├── models.py
│       ├── pwa_client.py
│       ├── cloud_project.py
│       ├── remote_control.py
│       └── simulator.py
└── tests/
    ├── test_part1_pwa_and_playstore.py
    ├── test_part2_setup_and_resources.py
    ├── test_part3_non_blocking_dispatch.py
    ├── test_part4_remote_control_switcher.py
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

## License

Licensed under the [MIT License](LICENSE).
