"""Part 1 tests: Play Store Googlebook hardware lock vs. Android Chrome PWA."""

import pytest

from antigravity_mobile import (
    ANTIGRAVITY_WEB_URL,
    AndroidChromePWA,
    DeviceHardware,
    GoogleAccount,
    PlayStoreFilter,
    PlayStoreHardwareLockError,
    SubscriptionPlan,
)


def test_play_store_blocks_standard_android_smartphone() -> None:
    assert PlayStoreFilter.check_apk_eligibility(DeviceHardware.ANDROID_SMARTPHONE) is False
    with pytest.raises(PlayStoreHardwareLockError, match="Googlebooks"):
        PlayStoreFilter.install_native_apk(DeviceHardware.ANDROID_SMARTPHONE)


def test_play_store_allows_googlebook_hardware() -> None:
    assert PlayStoreFilter.check_apk_eligibility(DeviceHardware.GOOGLEBOOK) is True
    result = PlayStoreFilter.install_native_apk(DeviceHardware.GOOGLEBOOK)
    assert "Googlebook OS" in result


def test_chrome_pwa_enables_fullscreen_and_service_worker_on_android_phone() -> None:
    pwa = AndroidChromePWA(device=DeviceHardware.ANDROID_SMARTPHONE)
    pwa.open_chrome(ANTIGRAVITY_WEB_URL)
    pwa.sign_in(GoogleAccount(email="dev@example.com", plan=SubscriptionPlan.GOOGLE_AI_PRO))
    pwa.install_to_home_screen()

    assert pwa.installed_to_home_screen is True
    assert pwa.fullscreen_ui is True
    assert pwa.service_worker.active is True
    assert pwa.service_worker.push_subscription_active is True
    assert pwa.service_worker.scope == ANTIGRAVITY_WEB_URL
