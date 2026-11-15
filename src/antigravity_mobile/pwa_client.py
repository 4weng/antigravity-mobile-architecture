"""Part 1 & Part 2 simulation: Android Chrome PWA, Play Store lock, and Service Worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .models import (
    DeviceHardware,
    ExecutionMode,
    PlayStoreHardwareLockError,
    PushNotification,
    PWARequirementError,
    SubscriptionPlan,
    SubscriptionRequiredError,
)

if TYPE_CHECKING:
    from .remote_control import WebRelay, Windows11Workstation

ANTIGRAVITY_WEB_URL = "https://antigravity.google.com"


class PlayStoreFilter:
    """Simulates Google Play Store APK hardware-lock filtering (Part 1)."""

    SUPPORTED_HARDWARE = frozenset({DeviceHardware.GOOGLEBOOK})

    @classmethod
    def check_apk_eligibility(cls, device: DeviceHardware) -> bool:
        """Return True only when device hardware is a Googlebook."""
        return device in cls.SUPPORTED_HARDWARE

    @classmethod
    def install_native_apk(cls, device: DeviceHardware) -> str:
        """Attempt direct Play Store APK installation.

        Standard Android smartphones are blocked by AndroidManifest/Play Store
        hardware filters that restrict native installation to Googlebooks.
        """
        if not cls.check_apk_eligibility(device):
            raise PlayStoreHardwareLockError(
                f"Play Store APK installation blocked on '{device.value}': "
                "AndroidManifest and Play Store filter lock native APK to Googlebooks only. "
                f"Use Chrome PWA at {ANTIGRAVITY_WEB_URL} instead."
            )
        return "Installed Google Antigravity native APK on Googlebook OS."


@dataclass
class GoogleAccount:
    """Google Account profile used to sign into antigravity.google.com (Part 2, Step 2)."""

    email: str
    is_personal: bool = True
    plan: SubscriptionPlan = SubscriptionPlan.GOOGLE_AI_PRO

    def has_remote_agent_access(self) -> bool:
        """Verify personal Google account with active Google AI Pro or AI Ultra."""
        return self.is_personal and self.plan in {
            SubscriptionPlan.GOOGLE_AI_PRO,
            SubscriptionPlan.GOOGLE_AI_ULTRA,
        }


@dataclass
class ServiceWorker:
    """PWA Service Worker enabling background push notifications (Part 1 & 2)."""

    scope: str = ANTIGRAVITY_WEB_URL
    active: bool = False
    push_subscription_active: bool = False
    notifications: list[PushNotification] = field(default_factory=list)

    def register(self) -> None:
        """Activate Service Worker and subscribe to mobile push notifications."""
        self.active = True
        self.push_subscription_active = True

    def receive_push(self, notification: PushNotification) -> None:
        """Receive a push notification even if Chrome is closed or screen is locked."""
        if not (self.active and self.push_subscription_active):
            raise PWARequirementError(
                "Service Worker push subscription is inactive; install PWA to Home Screen first."
            )
        self.notifications.append(notification)


@dataclass
class AndroidChromePWA:
    """Android smartphone Chrome Progressive Web App client (Parts 1–4)."""

    device: DeviceHardware = DeviceHardware.ANDROID_SMARTPHONE
    current_url: str | None = None
    signed_in_account: GoogleAccount | None = None
    installed_to_home_screen: bool = False
    fullscreen_ui: bool = False
    browser_open: bool = False
    screen_locked: bool = False
    active_mode: ExecutionMode = ExecutionMode.REMOTE_CONVERSATIONS
    service_worker: ServiceWorker = field(default_factory=ServiceWorker)

    def open_chrome(self, url: str = ANTIGRAVITY_WEB_URL) -> None:
        """Step 1: Open Google Chrome on Android and navigate to antigravity.google.com."""
        if url != ANTIGRAVITY_WEB_URL:
            raise PWARequirementError(
                f"Expected official Antigravity Web URL '{ANTIGRAVITY_WEB_URL}', got '{url}'."
            )
        self.browser_open = True
        self.screen_locked = False
        self.current_url = url

    def sign_in(self, account: GoogleAccount) -> None:
        """Step 2: Sign in using a Personal Google Account with AI Pro / AI Ultra."""
        if self.current_url != ANTIGRAVITY_WEB_URL:
            raise PWARequirementError("Must open antigravity.google.com in Chrome before sign-in.")
        if not account.is_personal:
            raise SubscriptionRequiredError(
                "Sign-in requires a Personal Google Account with Google AI Pro or AI Ultra."
            )
        if not account.has_remote_agent_access():
            raise SubscriptionRequiredError(
                f"Account '{account.email}' has plan '{account.plan.value}'. "
                "Google AI Pro or Google AI Ultra is required."
            )
        self.signed_in_account = account

    def install_to_home_screen(self) -> None:
        """Step 3: Menu (⋮) -> Add to Home screen -> Install (PWA).

        Creates a standalone Home Screen icon, enables Full-screen UI,
        and registers the Service Worker for Mobile Push Notifications.
        """
        if self.current_url != ANTIGRAVITY_WEB_URL or self.signed_in_account is None:
            raise PWARequirementError(
                "Open antigravity.google.com and sign in before installing PWA to Home Screen."
            )
        self.installed_to_home_screen = True
        self.fullscreen_ui = True
        self.service_worker.register()

    def close_chrome_and_lock_screen(self) -> None:
        """Simulate closing Chrome and locking phone screen after dispatching a task."""
        self.browser_open = False
        self.screen_locked = True

    def reopen_from_home_screen(self) -> None:
        """Reopen the standalone PWA icon from the Android Home Screen."""
        if not self.installed_to_home_screen:
            raise PWARequirementError("PWA is not installed on the Home Screen.")
        self.browser_open = True
        self.screen_locked = False
        self.fullscreen_ui = True

    def switch_instance(
        self,
        target_mode: ExecutionMode,
        relay: WebRelay | None = None,
        workstation: Windows11Workstation | None = None,
    ) -> ExecutionMode:
        """Part 4: Top Bar Instance Switcher between Pure Cloud and Windows 11 PC."""
        if self.signed_in_account is None:
            raise PWARequirementError("Must be signed in to switch execution instances.")

        if target_mode == ExecutionMode.REMOTE_CONTROL:
            if relay is None or workstation is None:
                raise PWARequirementError(
                    "Switching to Remote Control requires a WebRelay and Windows11Workstation."
                )
            relay.verify_connection(workstation)

        self.active_mode = target_mode
        return self.active_mode
