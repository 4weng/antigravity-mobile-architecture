"""Core domain models, enums, and exceptions for Antigravity Mobile architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DeviceHardware(str, Enum):
    """Hardware target evaluated by Google Play Store filters."""

    ANDROID_SMARTPHONE = "android_smartphone"
    GOOGLEBOOK = "googlebook"


class SubscriptionPlan(str, Enum):
    """Google Account subscription tier for Pure Cloud Remote Agents."""

    FREE = "free"
    GOOGLE_AI_PRO = "google_ai_pro"
    GOOGLE_AI_ULTRA = "google_ai_ultra"


class ModelTier(str, Enum):
    """Model tier roles described in Part 3 of the manual."""

    ARGON = "Argon"
    HYDROGEN = "Hydrogen"
    CLAUDE_OPUS_5_5 = "Claude Opus 5.5"

    @property
    def software_role(self) -> str:
        """Return the software engineering role for this tier."""
        roles = {
            ModelTier.ARGON: "Planning & Subtask Breakdown (วางแผนโครงสร้างและแตกงานย่อย)",
            ModelTier.HYDROGEN: "Branch Coding (เขียนโค้ดแยกตาม Branch)",
            ModelTier.CLAUDE_OPUS_5_5: "Code Review & Verification (ตรวจสอบความถูกต้องของโค้ด)",
        }
        return roles[self]

    @property
    def civil_engineering_analogy(self) -> str:
        """Return the senior civil/structural engineering analogy from the manual."""
        analogies = {
            ModelTier.ARGON: "Project Director / Lead Structural Architect",
            ModelTier.HYDROGEN: "Production Engineer",
            ModelTier.CLAUDE_OPUS_5_5: "Independent Peer Reviewer / Checker",
        }
        return analogies[self]


class ExecutionMode(str, Enum):
    """Execution targets switched via the Top Bar Instance Switcher (Part 4)."""

    REMOTE_CONVERSATIONS = "remote_conversations"  # Pure Cloud (Remote Agents)
    REMOTE_CONTROL = "remote_control"  # Windows 11 Home PC Workstation


class ResourceType(str, Enum):
    """Cloud connector resource types bound to a Remote Project (Part 2)."""

    GITHUB_REPO = "github_repo"
    GOOGLE_DRIVE = "google_drive"


class TaskStatus(str, Enum):
    """Lifecycle states of an asynchronous cloud task (Part 3)."""

    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED_FOR_USER_INPUT = "blocked_for_user_input"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PushNotification:
    """Notification delivered to Android PWA via Service Worker."""

    task_id: str
    title: str
    body: str
    action_required: bool
    mode: ExecutionMode


@dataclass(frozen=True)
class ProjectResource:
    """OAuth-connected cloud resource (GitHub repo or Google Drive)."""

    resource_type: ResourceType
    name: str
    uri: str
    oauth_connected: bool = True
    capacity_tb: int | None = None


@dataclass(frozen=True)
class WorkstationSpec:
    """Hardware profile of a home Windows 11 workstation (Part 4)."""

    os_name: str = "Windows 11"
    ram_gb: int = 128
    gpu_name: str = "NVIDIA RTX"
    vram_gb: int = 16


@dataclass
class WorkloadResult:
    """Result returned from executing a workload on the home Windows 11 PC."""

    workload_name: str
    input_path: str
    used_ram_gb: int
    used_vram_gb: int
    output_summary: str
    bandwidth_mode: str = field(default="lightweight_command_relay")


class AntigravityArchitectureError(Exception):
    """Base exception for architecture rule violations."""


class PlayStoreHardwareLockError(AntigravityArchitectureError):
    """Raised when attempting to install Play Store APK on non-Googlebook hardware."""


class SubscriptionRequiredError(AntigravityArchitectureError):
    """Raised when account is not a Personal Google Account with AI Pro or AI Ultra."""


class PWARequirementError(AntigravityArchitectureError):
    """Raised when an operation requires Chrome PWA setup or active Service Worker."""


class LocalMountNotAllowedError(AntigravityArchitectureError):
    """Raised when trying to mount local PC disk paths in a Pure Cloud Remote Project."""


class WorkstationOfflineError(AntigravityArchitectureError):
    """Raised when Remote Control is selected while the Windows 11 host is offline."""
