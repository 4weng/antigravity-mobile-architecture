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
from .skill_guidance import (
    BaselineSupport,
    BrowserSupportPolicy,
    ComplianceResult,
    EvalBenchmarkResult,
    EvalCalibrationResult,
    EvalTask,
    GuidanceEvalHarness,
    LegacyFinding,
    LegacyRule,
    SkillPack,
    WebGuide,
)

__all__ = [
    "ANTIGRAVITY_WEB_URL",
    "AndroidChromePWA",
    "AntigravityMobileSimulator",
    "BaselineSupport",
    "BrowserSupportPolicy",
    "CloudTask",
    "ComplianceResult",
    "DeviceHardware",
    "EvalBenchmarkResult",
    "EvalCalibrationResult",
    "EvalTask",
    "ExecutionMode",
    "GoogleAccount",
    "GuidanceEvalHarness",
    "IsolatedCloudSandbox",
    "LegacyFinding",
    "LegacyRule",
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
    "SkillPack",
    "SubscriptionPlan",
    "SubscriptionRequiredError",
    "TaskStatus",
    "WebGuide",
    "WebRelay",
    "Windows11Workstation",
    "WorkstationOfflineError",
    "WorkstationSpec",
]
