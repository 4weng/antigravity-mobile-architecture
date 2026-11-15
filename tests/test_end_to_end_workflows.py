"""End-to-end integration tests covering all 4 parts of the Antigravity Mobile manual."""

from antigravity_mobile import AntigravityMobileSimulator, ExecutionMode, ModelTier, TaskStatus


def test_full_end_to_end_simulation_demo() -> None:
    sim = AntigravityMobileSimulator()
    report = sim.run_full_demo()

    # Part 1: Smartphone blocked on Play Store, PWA active
    assert report.play_store_blocked_on_smartphone is True
    assert report.pwa_installed is True
    assert report.fullscreen_ui is True

    # Part 2: GitHub + Google Drive 5TB attached to Remote Project
    assert "4weng/antigravity-mobile-architecture" in report.remote_project_resources
    assert "Civil-Specs-5TB" in report.remote_project_resources

    # Part 3: Argon -> Hydrogen (with approval) -> Claude Opus 5.5
    assert [t.model_tier for t in report.cloud_tasks] == [
        ModelTier.ARGON,
        ModelTier.HYDROGEN,
        ModelTier.CLAUDE_OPUS_5_5,
    ]
    assert all(t.status == TaskStatus.COMPLETED for t in report.cloud_tasks)

    # Part 4: Remote Control on Windows 11 PC via lightweight Web Relay
    assert sim.pwa.active_mode == ExecutionMode.REMOTE_CONTROL
    assert report.remote_control_result.workload_name == "drone-crack-inspection"
    assert report.remote_control_result.bandwidth_mode == "lightweight_command_relay"

    # Push notifications: Argon complete + Hydrogen approval + Hydrogen complete + Opus complete + Remote Control complete
    assert report.push_notifications_received == 5
