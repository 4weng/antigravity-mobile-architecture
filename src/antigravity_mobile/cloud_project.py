"""Part 2 & Part 3 simulation: Pure Cloud Remote Project and Non-blocking Task Dispatch."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .models import (
    ExecutionMode,
    LocalMountNotAllowedError,
    ModelTier,
    ProjectResource,
    PushNotification,
    PWARequirementError,
    ResourceType,
    TaskStatus,
)
from .pwa_client import AndroidChromePWA, ServiceWorker


@dataclass
class RemoteProject:
    """Cloud-hosted project using OAuth/Connectors instead of local disk mounts (Part 2, Step 4)."""

    project_id: str
    name: str
    resources: list[ProjectResource] = field(default_factory=list)

    def attach_github_repo(self, repo_slug: str, oauth_connected: bool = True) -> ProjectResource:
        """Attach a GitHub repository (e.g., '4weng/antigravity-mobile-architecture') to Resources."""
        resource = ProjectResource(
            resource_type=ResourceType.GITHUB_REPO,
            name=repo_slug,
            uri=f"https://github.com/{repo_slug}",
            oauth_connected=oauth_connected,
        )
        self.resources.append(resource)
        return resource

    def attach_google_drive(
        self,
        folder_or_doc: str,
        capacity_tb: int = 5,
        oauth_connected: bool = True,
    ) -> ProjectResource:
        """Attach Google Drive specifications/calculation sheets (e.g., 5TB Cloud BIM Drive)."""
        resource = ProjectResource(
            resource_type=ResourceType.GOOGLE_DRIVE,
            name=folder_or_doc,
            uri=f"https://drive.google.com/drive/folders/{folder_or_doc}",
            oauth_connected=oauth_connected,
            capacity_tb=capacity_tb,
        )
        self.resources.append(resource)
        return resource

    def mount_local_path(self, local_path: str) -> None:
        """Reject local PC filesystem mounts in Pure Cloud Remote Projects."""
        raise LocalMountNotAllowedError(
            f"Cannot mount local path '{local_path}' in Pure Cloud Remote Project '{self.name}'. "
            "Pure Cloud projects use OAuth Connectors for GitHub and Google Drive Resources."
        )

    def has_connected_resources(self) -> bool:
        """Return True if at least one OAuth-connected resource is attached."""
        return any(res.oauth_connected for res in self.resources)


@dataclass
class CloudTask:
    """Represents a task dispatched to an Isolated Cloud Sandbox (Part 3)."""

    task_id: str
    project_id: str
    prompt: str
    model_tier: ModelTier
    status: TaskStatus = TaskStatus.QUEUED
    requires_approval: bool = False
    approved: bool | None = None
    branch_name: str | None = None
    output_summary: str = ""
    websocket_bound: bool = False
    requires_home_pc: bool = False


@dataclass
class IsolatedCloudSandbox:
    """Google Isolated Cloud Server executing Remote Agents 100% in the cloud (Part 3)."""

    sandbox_id: str = field(default_factory=lambda: f"sandbox-{uuid.uuid4().hex[:8]}")
    requires_home_pc: bool = False
    persistent_mobile_websocket: bool = False
    tasks: dict[str, CloudTask] = field(default_factory=dict)

    def execute(
        self,
        task: CloudTask,
        project: RemoteProject,
        service_worker: ServiceWorker | None = None,
    ) -> CloudTask:
        """Execute task asynchronously without keeping mobile Chrome open.

        - Argon: produces architectural plan & subtask breakdown.
        - Hydrogen: generates implementation on an isolated feature branch.
        - Claude Opus 5.5: performs code review & verification.
        """
        self.tasks[task.task_id] = task
        task.status = TaskStatus.RUNNING
        resource_names = ", ".join(r.name for r in project.resources) or "no external resources"

        if task.requires_approval and task.approved is not True:
            task.status = TaskStatus.BLOCKED_FOR_USER_INPUT
            task.output_summary = (
                f"[{task.model_tier.value}] Paused for user approval on project "
                f"'{project.name}' ({resource_names})."
            )
            if service_worker is not None and service_worker.active:
                service_worker.receive_push(
                    PushNotification(
                        task_id=task.task_id,
                        title=f"Approval Needed ({task.model_tier.value})",
                        body=task.output_summary,
                        action_required=True,
                        mode=ExecutionMode.REMOTE_CONVERSATIONS,
                    )
                )
            return task

        if task.model_tier == ModelTier.ARGON:
            task.output_summary = (
                f"[Argon | {task.model_tier.civil_engineering_analogy}] "
                f"Planned structure & subtasks for '{task.prompt}' using [{resource_names}]."
            )
        elif task.model_tier == ModelTier.HYDROGEN:
            task.branch_name = task.branch_name or f"feat/{task.task_id}"
            task.output_summary = (
                f"[Hydrogen | {task.model_tier.civil_engineering_analogy}] "
                f"Implemented branch '{task.branch_name}' for '{task.prompt}' using [{resource_names}]."
            )
        else:
            task.output_summary = (
                f"[Claude Opus 5.5 | {task.model_tier.civil_engineering_analogy}] "
                f"Reviewed and verified code correctness for '{task.prompt}'."
            )

        task.status = TaskStatus.COMPLETED
        if service_worker is not None and service_worker.active:
            service_worker.receive_push(
                PushNotification(
                    task_id=task.task_id,
                    title=f"Task Completed ({task.model_tier.value})",
                    body=task.output_summary,
                    action_required=False,
                    mode=ExecutionMode.REMOTE_CONVERSATIONS,
                )
            )
        return task

    def approve_task(
        self,
        task_id: str,
        project: RemoteProject,
        service_worker: ServiceWorker | None = None,
    ) -> CloudTask:
        """Approve a task currently in BLOCKED_FOR_USER_INPUT and finish execution."""
        task = self.tasks[task_id]
        task.approved = True
        return self.execute(task=task, project=project, service_worker=service_worker)


@dataclass
class NonBlockingDispatcher:
    """Dispatches prompts from Android PWA to Google Isolated Cloud Sandbox (Part 3, Step 5)."""

    sandbox: IsolatedCloudSandbox = field(default_factory=IsolatedCloudSandbox)

    def dispatch(
        self,
        pwa: AndroidChromePWA,
        project: RemoteProject,
        prompt: str,
        model_tier: ModelTier,
        requires_approval: bool = False,
        branch_name: str | None = None,
        auto_run: bool = True,
    ) -> CloudTask:
        """Dispatch a non-blocking task.

        The mobile client may immediately close Chrome or lock its screen after
        dispatch because execution occurs in Google's Isolated Cloud Sandbox.
        """
        if pwa.signed_in_account is None:
            raise PWARequirementError("Sign in to antigravity.google.com before dispatching tasks.")
        if pwa.active_mode != ExecutionMode.REMOTE_CONVERSATIONS:
            raise PWARequirementError(
                "NonBlockingDispatcher targets Pure Cloud (REMOTE_CONVERSATIONS). "
                "Switch instance to REMOTE_CONVERSATIONS first."
            )
        if not project.has_connected_resources():
            raise PWARequirementError(
                f"Remote project '{project.name}' has no OAuth-connected Resources "
                "(attach GitHub repo or Google Drive first)."
            )

        task = CloudTask(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            project_id=project.project_id,
            prompt=prompt,
            model_tier=model_tier,
            status=TaskStatus.QUEUED,
            requires_approval=requires_approval,
            branch_name=branch_name,
            websocket_bound=False,
            requires_home_pc=False,
        )
        if auto_run:
            return self.sandbox.execute(
                task=task,
                project=project,
                service_worker=pwa.service_worker,
            )
        self.sandbox.tasks[task.task_id] = task
        return task
