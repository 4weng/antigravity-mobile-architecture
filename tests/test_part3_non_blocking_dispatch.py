"""Part 3 tests: Non-blocking asynchronous cloud dispatch, model tiers, and push notifications."""

import pytest

from antigravity_mobile import (
    ANTIGRAVITY_WEB_URL,
    AndroidChromePWA,
    GoogleAccount,
    ModelTier,
    NonBlockingDispatcher,
    PWARequirementError,
    RemoteProject,
    SubscriptionPlan,
    TaskStatus,
)


@pytest.fixture
def ready_pwa_and_project() -> tuple[AndroidChromePWA, RemoteProject]:
    pwa = AndroidChromePWA()
    pwa.open_chrome(ANTIGRAVITY_WEB_URL)
    pwa.sign_in(GoogleAccount(email="dev@example.com", plan=SubscriptionPlan.GOOGLE_AI_PRO))
    pwa.install_to_home_screen()

    project = RemoteProject(project_id="proj-cloud", name="PureCloudProject")
    project.attach_github_repo("4weng/antigravity-mobile-architecture")
    project.attach_google_drive("Specs-5TB", capacity_tb=5)
    return pwa, project


def test_model_tier_roles_and_civil_engineering_analogies() -> None:
    assert "Planning" in ModelTier.ARGON.software_role
    assert "Project Director" in ModelTier.ARGON.civil_engineering_analogy

    assert "Branch Coding" in ModelTier.HYDROGEN.software_role
    assert "Production Engineer" in ModelTier.HYDROGEN.civil_engineering_analogy

    assert "Code Review" in ModelTier.CLAUDE_OPUS_5_5.software_role
    assert "Independent Peer Reviewer" in ModelTier.CLAUDE_OPUS_5_5.civil_engineering_analogy


def test_non_blocking_execution_succeeds_with_chrome_closed_and_screen_locked(
    ready_pwa_and_project: tuple[AndroidChromePWA, RemoteProject],
) -> None:
    pwa, project = ready_pwa_and_project
    dispatcher = NonBlockingDispatcher()

    # Queue task, then close Chrome & lock screen immediately (no persistent WebSocket)
    task = dispatcher.dispatch(
        pwa=pwa,
        project=project,
        prompt="Plan FEM microservice layout",
        model_tier=ModelTier.ARGON,
        auto_run=False,
    )
    pwa.close_chrome_and_lock_screen()
    assert pwa.browser_open is False
    assert pwa.screen_locked is True
    assert task.websocket_bound is False
    assert task.requires_home_pc is False

    # Cloud sandbox finishes in background and delivers Push Notification to Service Worker
    completed = dispatcher.sandbox.execute(
        task=task,
        project=project,
        service_worker=pwa.service_worker,
    )
    assert completed.status == TaskStatus.COMPLETED
    assert len(pwa.service_worker.notifications) == 1
    notif = pwa.service_worker.notifications[0]
    assert notif.task_id == task.task_id
    assert notif.action_required is False


def test_hydrogen_branch_coding_blocked_for_approval_then_completed(
    ready_pwa_and_project: tuple[AndroidChromePWA, RemoteProject],
) -> None:
    pwa, project = ready_pwa_and_project
    dispatcher = NonBlockingDispatcher()

    task = dispatcher.dispatch(
        pwa=pwa,
        project=project,
        prompt="Write wind-load calculation module",
        model_tier=ModelTier.HYDROGEN,
        branch_name="feat/wind-load",
        requires_approval=True,
    )
    assert task.status == TaskStatus.BLOCKED_FOR_USER_INPUT
    assert pwa.service_worker.notifications[-1].action_required is True

    # User approves from mobile notification -> task resumes and completes on branch
    approved_task = dispatcher.sandbox.approve_task(
        task_id=task.task_id,
        project=project,
        service_worker=pwa.service_worker,
    )
    assert approved_task.status == TaskStatus.COMPLETED
    assert approved_task.branch_name == "feat/wind-load"
    assert pwa.service_worker.notifications[-1].action_required is False


def test_dispatch_requires_connected_project_resources(
    ready_pwa_and_project: tuple[AndroidChromePWA, RemoteProject],
) -> None:
    pwa, _ = ready_pwa_and_project
    empty_project = RemoteProject(project_id="empty", name="NoResources")
    dispatcher = NonBlockingDispatcher()

    with pytest.raises(PWARequirementError, match="no OAuth-connected Resources"):
        dispatcher.dispatch(
            pwa=pwa,
            project=empty_project,
            prompt="Run review",
            model_tier=ModelTier.CLAUDE_OPUS_5_5,
        )
