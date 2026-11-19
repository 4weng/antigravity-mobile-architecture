"""Part 4 walkthrough: Windows 11 Remote Control + ChindaMT-4B (มือจับทำ).

รันทีเดียว:
  .\\.venv\\Scripts\\Activate.ps1
  python examples\\part4_walkthrough.py

โหลดโมเดลจริงบน GPU ด้วย:
  python examples\\part4_walkthrough.py --gpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ให้รันจากโฟลเดอร์ไหนก็เจอแพ็กเกจใน src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from antigravity_mobile import (  # noqa: E402
    ANTIGRAVITY_WEB_URL,
    AndroidChromePWA,
    ExecutionMode,
    GoogleAccount,
    SubscriptionPlan,
    WebRelay,
    Windows11Workstation,
    WorkstationOfflineError,
    WorkstationSpec,
)
from antigravity_mobile.chinda_mt import ChindaTranslator, TranslatorConfig  # noqa: E402


def banner(step: int, title_th: str, title_en: str) -> None:
    print()
    print("=" * 72)
    print(f"  ขั้นที่ {step}: {title_th}")
    print(f"  Step {step}: {title_en}")
    print("=" * 72)


def explain(th: str, en: str = "") -> None:
    print(f"  → {th}")
    if en:
        print(f"     ({en})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="จับมือทำ Part 4: Remote Control + Web Relay (+ optional GPU)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="โหลด ChindaMT-4B จริงบน GPU แล้วแปลประโยคตัวอย่าง",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Antigravity Part 4 Walkthrough")
    print("โปรเจกต์นี้เป็นแบบจำลองโครงสร้าง — ไม่ใช่สายต่อจริงไปยังแอปมือถือ")
    print(f"โฟลเดอร์โปรเจกต์: {ROOT}")
    if args.gpu:
        print("โหมด: จำลอง Part 4 + แปลจริงบน GPU (--gpu)")
    else:
        print("โหมด: จำลอง Part 4 อย่างเดียว (เพิ่ม --gpu ถ้าอยากแปลจริง)")

    # ------------------------------------------------------------------
    banner(1, "สร้างเครื่อง Windows 11 ที่บ้าน", "Create home Windows 11 workstation")
    pc = Windows11Workstation(
        hostname="home-win11",
        spec=WorkstationSpec(
            os_name="Windows 11",
            ram_gb=128,
            gpu_name="NVIDIA RTX",
            vram_gb=16,
        ),
        local_files={
            str(ROOT / "main.py"): "ChindaMT-4B demo entrypoint",
            r"D:\drone_inspections\tower_b_cracks.raw": "High-res drone crack inspection images",
        },
    )
    explain(
        f"สร้าง PC ชื่อ '{pc.hostname}' | Remote เปิดอยู่ไหม? {pc.remote_control_enabled}",
        f"hostname={pc.hostname}, remote_control_enabled={pc.remote_control_enabled}",
    )
    explain("ตอนนี้ Remote ยังปิด — มือถือยังสั่งงานเข้าเครื่องนี้ไม่ได้")

    # ------------------------------------------------------------------
    banner(
        2,
        "ลองสลับโหมดก่อนเปิด Remote (ควรพัง)",
        "Switch to Remote Control BEFORE enabling (should fail)",
    )
    phone = AndroidChromePWA()
    phone.open_chrome(ANTIGRAVITY_WEB_URL)
    phone.sign_in(
        GoogleAccount(email="student@example.com", plan=SubscriptionPlan.GOOGLE_AI_ULTRA)
    )
    phone.install_to_home_screen()
    relay = WebRelay()
    explain(
        f"มือถือพร้อมแล้ว | โหมดปัจจุบัน: {phone.active_mode}",
        f"active_mode={phone.active_mode}",
    )

    try:
        phone.switch_instance(
            target_mode=ExecutionMode.REMOTE_CONTROL,
            relay=relay,
            workstation=pc,
        )
        print("  !! ไม่ควรมาถึงตรงนี้ — ถ้ามาถึงแปลว่า logic พัง")
        return 1
    except WorkstationOfflineError as exc:
        explain("ถูกต้อง! สลับไม่ได้เพราะ PC ยังไม่เปิด Remote Control")
        explain(f"ข้อความ error: {exc}")

    # ------------------------------------------------------------------
    banner(
        3,
        "เปิด Remote Control บน PC (จำลอง agy --remote-agent)",
        "Enable Remote Control (simulate `agy --remote-agent`)",
    )
    msg = pc.start_remote_control(via_cli=True)
    explain(msg)
    explain(f"คำสั่งที่ใช้: {pc.cli_command_used}")
    explain(f"Remote เปิดแล้ว? {pc.remote_control_enabled}")
    explain(
        "ของจริงบนเครื่องคุณ: เปิด Antigravity Desktop แล้วรัน `agy --remote-agent`",
        "Real product: Antigravity Desktop Top Bar → Remote Control, or CLI",
    )

    # ------------------------------------------------------------------
    banner(
        4,
        "มือถือสลับ Instance มาเป็น Windows PC",
        "Phone Instance Switcher → Windows PC",
    )
    mode = phone.switch_instance(
        target_mode=ExecutionMode.REMOTE_CONTROL,
        relay=relay,
        workstation=pc,
    )
    explain(f"โหมดปัจจุบันของมือถือ: {mode}")
    explain(
        "แปลว่าคำสั่งจากแชตมือถือจะถูกส่งผ่าน Web Relay ไปที่ PC บ้าน",
        "Chat commands go over lightweight Web Relay (not full RDP pixel streaming)",
    )
    explain(
        f"Relay endpoint: {relay.relay_endpoint} | "
        f"pixel streaming? {relay.pixel_streaming_enabled}"
    )

    # ------------------------------------------------------------------
    banner(
        5,
        "สั่งงานหนักบน PC (เช่น วิเคราะห์รูปโดรน)",
        "Dispatch a heavy local workload via Web Relay",
    )
    drone = relay.dispatch_command(
        workstation=pc,
        workload_name="drone-crack-inspection",
        input_path=r"D:\drone_inspections\tower_b_cracks.raw",
        service_worker=phone.service_worker,
    )
    explain(drone.output_summary)
    explain(f"ใช้ RAM โดยประมาณ: {drone.used_ram_gb} GB | VRAM: {drone.used_vram_gb} GB")
    explain(f"โหมดส่งงาน: {drone.bandwidth_mode}")

    # ------------------------------------------------------------------
    banner(
        6,
        "สั่งรัน / เรียกงานในโปรเจกต์ (main.py)",
        "Dispatch project workload pointing at main.py",
    )
    main_py = str(ROOT / "main.py")
    job = relay.dispatch_command(
        workstation=pc,
        workload_name="run-main-py",
        input_path=main_py,
        service_worker=phone.service_worker,
    )
    explain(job.output_summary)
    explain(
        "หมายเหตุ: ในแบบจำลอง ขั้นนี้แค่ 'ส่งคำสั่งสำเร็จ' — ยังไม่ได้รัน main.py จริง",
        "Simulation only acknowledges the command path; use --gpu for real translation",
    )

    # ------------------------------------------------------------------
    banner(
        7,
        "ส่งงานแปล ChindaMT-4B ผ่าน Web Relay",
        "Dispatch ChindaMT-4B translation over Web Relay",
    )
    sample = (
        "The **shear wall** and <code>prestressed concrete</code> passed "
        "[seismic load](https://example.com/spec) inspection."
    )
    cfg = TranslatorConfig.formal_engineering_preset(
        direction="en2th",
        tone="official_report",
        preserve_html_tags=True,
        preserve_markdown=True,
    )

    if args.gpu:
        explain("กำลังโหลดโมเดลจริงบน GPU... (อาจใช้เวลาสักครู่)")
        translator = ChindaTranslator(cfg).load()
        chinda = relay.dispatch_chinda_translation(
            workstation=pc,
            text=sample,
            translator=translator,
            service_worker=phone.service_worker,
        )
    else:
        explain("โหมดจำลอง: ไม่โหลดน้ำหนักโมเดล (ใส่ --gpu เพื่อแปลจริง)")
        chinda = relay.dispatch_chinda_translation(
            workstation=pc,
            text=sample,
            config=cfg,
            service_worker=phone.service_worker,
        )

    explain(f"ต้นฉบับ: {sample}")
    explain(chinda.output_summary)
    explain(f"ใช้ RAM โดยประมาณ: {chinda.used_ram_gb} GB | VRAM: {chinda.used_vram_gb} GB")

    # ------------------------------------------------------------------
    banner(8, "สรุปแจ้งเตือนบนมือถือ", "Mobile push notifications summary")
    notes = phone.service_worker.notifications
    explain(f"จำนวน Push Notifications: {len(notes)}")
    for i, note in enumerate(notes, start=1):
        print(f"  [{i}] {note.title}")
        print(f"      {note.body[:120]}{'...' if len(note.body) > 120 else ''}")

    print()
    print("-" * 72)
    print("เสร็จแล้ว!")
    print("  จำลองอย่างเดียว:  python examples\\part4_walkthrough.py")
    print("  แปลจริงบน GPU:   python examples\\part4_walkthrough.py --gpu")
    print("  ของจริงบนมือถือ: ติดตั้ง Antigravity Desktop แล้วรัน `agy --remote-agent`")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
