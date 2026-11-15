"""Simulation package modeling Google Antigravity Mobile architecture.

Covers all four architectural pillars from the reference manual:
1. Android Chrome PWA access vs. Play Store Googlebook hardware lock.
2. Pure Cloud Remote Project setup with OAuth GitHub & Google Drive resources.
3. Non-blocking asynchronous cloud task dispatch across model tiers
   (Argon, Hydrogen, Claude Opus 5.5) with Service Worker push notifications.
4. Windows 11 Remote Control mode via `agy --remote-agent`, lightweight Web Relay,
   and Top Bar Instance Switcher.
"""

from .cloud_project import CloudTask, IsolatedCloudSandbox, NonBlockingDispatcher, RemoteProject
from .models import (
    DeviceHardware,
    ExecutionMode,
    LocalMountNotAllowedError,
    ModelTier,
    PlayStoreHardwareLockError,
    ProjectResource,
    PushNotification,
    PWARequirementError,
    ResourceType,
    SubscriptionPlan,
    SubscriptionRequiredError,
    TaskStatus,
    WorkstationOfflineError,
    WorkstationSpec,
)
from .pwa_client import (
    ANTIGRAVITY_WEB_URL,
    AndroidChromePWA,
    GoogleAccount,
    PlayStoreFilter,
    ServiceWorker,
)
from .remote_control import WebRelay, Windows11Workstation
from .simulator import AntigravityMobileSimulator

__all__ = [
    "ANTIGRAVITY_WEB_URL",
    "AndroidChromePWA",
    "AntigravityMobileSimulator",
    "CloudTask",
    "DeviceHardware",
    "ExecutionMode",
    "GoogleAccount",
    "IsolatedCloudSandbox",
    "LocalMountNotAllowedError",
    "ModelTier",
    "NonBlockingDispatcher",
    "PWARequirementError",
    "PlayStoreFilter",
    "PlayStoreHardwareLockError",
    "ProjectResource",
    "PushNotification",
    "RemoteProject",
    "ResourceType",
    "ServiceWorker",
    "SubscriptionPlan",
    "SubscriptionRequiredError",
    "TaskStatus",
    "WebRelay",
    "Windows11Workstation",
    "WorkstationOfflineError",
    "WorkstationSpec",
]
