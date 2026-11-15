"""Part 2 tests: 4-step PWA install & Pure Cloud Remote Project OAuth Resources."""

import pytest

from antigravity_mobile import (
    ANTIGRAVITY_WEB_URL,
    AndroidChromePWA,
    GoogleAccount,
    LocalMountNotAllowedError,
    PWARequirementError,
    RemoteProject,
    ResourceType,
    SubscriptionPlan,
    SubscriptionRequiredError,
)


def test_sign_in_requires_personal_google_account_with_ai_pro_or_ultra() -> None:
    pwa = AndroidChromePWA()
    pwa.open_chrome(ANTIGRAVITY_WEB_URL)

    with pytest.raises(SubscriptionRequiredError, match="Google AI Pro or Google AI Ultra"):
        pwa.sign_in(GoogleAccount(email="free@example.com", plan=SubscriptionPlan.FREE))

    with pytest.raises(SubscriptionRequiredError, match="Personal Google Account"):
        pwa.sign_in(
            GoogleAccount(
                email="corp@company.com",
                is_personal=False,
                plan=SubscriptionPlan.GOOGLE_AI_ULTRA,
            )
        )

    ultra_account = GoogleAccount(email="senior@example.com", plan=SubscriptionPlan.GOOGLE_AI_ULTRA)
    pwa.sign_in(ultra_account)
    assert pwa.signed_in_account == ultra_account


def test_cannot_install_pwa_before_opening_url_and_signing_in() -> None:
    pwa = AndroidChromePWA()
    with pytest.raises(PWARequirementError):
        pwa.install_to_home_screen()


def test_cannot_open_unofficial_url_for_pwa() -> None:
    pwa = AndroidChromePWA()
    with pytest.raises(PWARequirementError):
        pwa.open_chrome("https://untrusted.example.com")


def test_remote_project_attaches_github_and_google_drive_and_rejects_local_mount() -> None:
    project = RemoteProject(project_id="proj-01", name="CloudBIM")
    gh = project.attach_github_repo("4weng/antigravity-mobile-architecture")
    gd = project.attach_google_drive("Structural-Calculations-5TB", capacity_tb=5)

    assert project.has_connected_resources() is True
    assert gh.resource_type == ResourceType.GITHUB_REPO
    assert gh.oauth_connected is True
    assert gd.resource_type == ResourceType.GOOGLE_DRIVE
    assert gd.capacity_tb == 5

    with pytest.raises(LocalMountNotAllowedError):
        project.mount_local_path(r"C:\Users\Engineer\Projects")

    with pytest.raises(LocalMountNotAllowedError):
        project.mount_local_path("/home/engineer/projects")
