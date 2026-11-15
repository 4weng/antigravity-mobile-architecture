"""Part 4 tests: Windows 11 Remote Control Mode, `agy --remote-agent`, Web Relay, Instance Switcher."""

import pytest

from antigravity_mobile import (
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


@pytest.fixture
def signed_in_pwa() -> AndroidChromePWA:
    pwa = AndroidChromePWA()
    pwa.open_chrome(ANTIGRAVITY_WEB_URL)
    pwa.sign_in(GoogleAccount(email="senior@example.com", plan=SubscriptionPlan.GOOGLE_AI_ULTRA))
    pwa.install_to_home_screen()
    return pwa


def test_cannot_switch_to_remote_control_when_windows_pc_is_offline(
    signed_in_pwa: AndroidChromePWA,
) -> None:
    workstation = Windows11Workstation()
    relay = WebRelay()

    with pytest.raises(WorkstationOfflineError):
        signed_in_pwa.switch_instance(
            target_mode=ExecutionMode.REMOTE_CONTROL,
            relay=relay,
            workstation=workstation,
        )
    assert signed_in_pwa.active_mode == ExecutionMode.REMOTE_CONVERSATIONS


def test_start_remote_control_via_cli_and_run_chindamt_4b_via_lightweight_relay(
    signed_in_pwa: AndroidChromePWA,
) -> None:
    workstation = Windows11Workstation(
        hostname="win11-128gb-rtx",
        spec=WorkstationSpec(os_name="Windows 11", ram_gb=128, gpu_name="NVIDIA RTX", vram_gb=16),
        local_files={
            r"D:\models\ChindaMT-4B\input.txt": "Thai-English structural engineering report",
        },
    )
    msg = workstation.start_remote_control(via_cli=True)
    assert "agy --remote-agent" in msg
    assert workstation.cli_command_used == "agy --remote-agent"

    relay = WebRelay()
    assert relay.pixel_streaming_enabled is False

    mode = signed_in_pwa.switch_instance(
        target_mode=ExecutionMode.REMOTE_CONTROL,
        relay=relay,
        workstation=workstation,
    )
    assert mode == ExecutionMode.REMOTE_CONTROL

    result = relay.dispatch_command(
        workstation=workstation,
        workload_name="ChindaMT-4B-translation",
        input_path=r"D:\models\ChindaMT-4B\input.txt",
        service_worker=signed_in_pwa.service_worker,
    )
    assert result.bandwidth_mode == "lightweight_command_relay"
    assert result.used_ram_gb == 64
    assert result.used_vram_gb == 14
    assert "128 GB RAM" in result.output_summary
    assert "16 GB VRAM" in result.output_summary
    assert signed_in_pwa.service_worker.notifications[-1].mode == ExecutionMode.REMOTE_CONTROL


def test_switch_freely_between_pure_cloud_and_windows_pc(
    signed_in_pwa: AndroidChromePWA,
) -> None:
    workstation = Windows11Workstation()
    workstation.start_remote_control(via_cli=False)
    relay = WebRelay()

    assert (
        signed_in_pwa.switch_instance(ExecutionMode.REMOTE_CONTROL, relay=relay, workstation=workstation)
        == ExecutionMode.REMOTE_CONTROL
    )
    assert signed_in_pwa.switch_instance(ExecutionMode.REMOTE_CONVERSATIONS) == ExecutionMode.REMOTE_CONVERSATIONS
