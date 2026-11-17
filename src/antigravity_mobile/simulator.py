"""End-to-end simulator reproducing all 4 parts of the Antigravity Mobile manual."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chinda_mt import TranslatorConfig
from .cloud_project import CloudTask, NonBlockingDispatcher, RemoteProject
from .models import (
    DeviceHardware,
    ExecutionMode,
    ModelTier,
    SubscriptionPlan,
    WorkloadResult,
    WorkstationSpec,
)
from .pwa_client import ANTIGRAVITY_WEB_URL, AndroidChromePWA, GoogleAccount, PlayStoreFilter
from .remote_control import WebRelay, Windows11Workstation
from .skill_guidance import EvalBenchmarkResult, EvalTask, GuidanceEvalHarness, SkillPack


@dataclass
class SimulationReport:
    """Structured summary produced by an end-to-end simulation run."""

    play_store_blocked_on_smartphone: bool
    pwa_installed: bool
    fullscreen_ui: bool
    remote_project_resources: list[str]
    cloud_tasks: list[CloudTask]
    remote_control_result: WorkloadResult
    push_notifications_received: int
    guidance_eval: EvalBenchmarkResult | None = None
    chinda_translation_result: WorkloadResult | None = None


@dataclass
class AntigravityMobileSimulator:
    """Orchestrates the complete mobile architecture lifecycle from the manual."""

    pwa: AndroidChromePWA = field(
        default_factory=lambda: AndroidChromePWA(device=DeviceHardware.ANDROID_SMARTPHONE)
    )
    dispatcher: NonBlockingDispatcher = field(default_factory=NonBlockingDispatcher)
    relay: WebRelay = field(default_factory=WebRelay)
    skill_pack: SkillPack = field(default_factory=SkillPack)
    workstation: Windows11Workstation = field(
        default_factory=lambda: Windows11Workstation(
            hostname="civil-eng-win11-ws",
            spec=WorkstationSpec(
                os_name="Windows 11", ram_gb=128, gpu_name="NVIDIA RTX", vram_gb=16
            ),
            local_files={
                r"D:\drone_inspections\tower_b_cracks.raw": (
                    "High-res drone crack inspection images"
                ),
                r"D:\models\ChindaMT-4B\weights.safetensors": (
                    "Local ChindaMT-4B translation weights"
                ),
            },
        )
    )

    def run_full_demo(self) -> SimulationReport:
        """Execute Parts 1–4 plus modern-web-guidance Skill Pack evaluation."""
        # Part 1: Verify Play Store APK lock on standard Android smartphone
        play_store_blocked = not PlayStoreFilter.check_apk_eligibility(self.pwa.device)

        # Part 2 (Steps 1–3): Open Chrome PWA, sign in with AI Pro, install to Home Screen
        self.pwa.open_chrome(ANTIGRAVITY_WEB_URL)
        account = GoogleAccount(
            email="engineer@example.com",
            is_personal=True,
            plan=SubscriptionPlan.GOOGLE_AI_PRO,
        )
        self.pwa.sign_in(account)
        self.pwa.install_to_home_screen()

        # Part 2 (Step 4): Create Pure Cloud Remote Project with GitHub + 5TB Google Drive
        project = RemoteProject(project_id="proj-mobile-01", name="StructuralBIM-Agent")
        project.attach_github_repo("4weng/antigravity-mobile-architecture")
        project.attach_google_drive("Civil-Specs-5TB", capacity_tb=5)

        # Part 3 (Step 5): Non-blocking multi-tier dispatch (Home PC is OFF; phone screen locked)
        plan_task = self.dispatcher.dispatch(
            pwa=self.pwa,
            project=project,
            prompt="Design seismic frame module architecture with PWA push",
            model_tier=ModelTier.ARGON,
            skill_pack=self.skill_pack,
            guide_query="pwa service worker push",
        )
        # Engineer closes Chrome & locks phone while cloud continues running!
        self.pwa.close_chrome_and_lock_screen()

        code_task = self.dispatcher.dispatch(
            pwa=self.pwa,
            project=project,
            prompt="Implement nonlinear solver branch",
            model_tier=ModelTier.HYDROGEN,
            branch_name="feat/nonlinear-solver",
            requires_approval=True,
        )
        # Engineer receives push notification and approves from mobile
        self.pwa.reopen_from_home_screen()
        self.dispatcher.sandbox.approve_task(
            task_id=code_task.task_id,
            project=project,
            service_worker=self.pwa.service_worker,
        )

        review_task = self.dispatcher.dispatch(
            pwa=self.pwa,
            project=project,
            prompt="Peer review nonlinear solver implementation",
            model_tier=ModelTier.CLAUDE_OPUS_5_5,
        )

        # Part 4: Start Remote Control on home Windows 11 PC (`agy --remote-agent`)
        # and switch instance on Android Top Bar to run local GPU workload via Web Relay
        self.workstation.start_remote_control(via_cli=True)
        self.pwa.switch_instance(
            target_mode=ExecutionMode.REMOTE_CONTROL,
            relay=self.relay,
            workstation=self.workstation,
        )
        rc_result = self.relay.dispatch_command(
            workstation=self.workstation,
            workload_name="drone-crack-inspection",
            input_path=r"D:\drone_inspections\tower_b_cracks.raw",
            service_worker=self.pwa.service_worker,
        )
        chinda_result = self.relay.dispatch_chinda_translation(
            workstation=self.workstation,
            text=(
                "The **shear wall** and <code>prestressed concrete</code> passed "
                "[seismic load](https://example.com/spec) inspection."
            ),
            config=TranslatorConfig.formal_engineering_preset(
                direction="en2th",
                tone="official_report",
                preserve_html_tags=True,
                preserve_markdown=True,
            ),
        )

        # Skill Pack Eval Harness (modern-web-guidance closed-loop calibration & uplift)
        eval_tasks = [
            EvalTask(
                task_id="eval-dialog",
                prompt="confirm before deleting structural model",
                guide_id="light-dismiss-dialog",
                golden_code="<dialog>showModal() closedby-click-backdrop</dialog>",
                unmodified_base_code="<div role='dialog'>Confirm</div>",
            ),
            EvalTask(
                task_id="eval-css-has",
                prompt="highlight invalid form group without JS",
                guide_id="css-has-parent-selector",
                golden_code=".field:has(:invalid) { border-color: red; }",
                unmodified_base_code=".field.invalid { border-color: red; }",
            ),
        ]
        harness = GuidanceEvalHarness(skill_pack=self.skill_pack)
        harness.calibrate(eval_tasks)
        guidance_eval = harness.run_benchmark(
            agent_name="antigravity_cli",
            model_tier=ModelTier.ARGON,
            tasks=eval_tasks,
            unguided_outputs={
                "eval-dialog": eval_tasks[0].unmodified_base_code,
                "eval-css-has": eval_tasks[1].golden_code,
            },
            guided_outputs={
                "eval-dialog": eval_tasks[0].golden_code,
                "eval-css-has": eval_tasks[1].golden_code,
            },
        )

        return SimulationReport(
            play_store_blocked_on_smartphone=play_store_blocked,
            pwa_installed=self.pwa.installed_to_home_screen,
            fullscreen_ui=self.pwa.fullscreen_ui,
            remote_project_resources=[r.name for r in project.resources],
            cloud_tasks=[plan_task, code_task, review_task],
            remote_control_result=rc_result,
            push_notifications_received=len(self.pwa.service_worker.notifications),
            guidance_eval=guidance_eval,
            chinda_translation_result=chinda_result,
        )


def main() -> None:
    """CLI entry point to run and print the full architecture simulation."""
    sim = AntigravityMobileSimulator()
    report = sim.run_full_demo()
    print("=== Antigravity Mobile Architecture Simulation ===")
    print(
        "Part 1 - Play Store APK blocked on Android phone: "
        f"{report.play_store_blocked_on_smartphone}"
    )
    print(
        f"Part 2 - Chrome PWA installed: {report.pwa_installed} (fullscreen={report.fullscreen_ui})"
    )
    print(f"Part 2 - Remote Resources: {report.remote_project_resources}")
    for task in report.cloud_tasks:
        print(
            f"Part 3 - [{task.model_tier.value}] status={task.status.value} "
            f"-> {task.output_summary}"
        )
    print(
        f"Part 4 - Remote Control ({report.remote_control_result.bandwidth_mode}): "
        f"{report.remote_control_result.output_summary}"
    )
    if report.chinda_translation_result is not None:
        print(
            f"Part 4 - ChindaMT-4B ({report.chinda_translation_result.bandwidth_mode}): "
            f"{report.chinda_translation_result.output_summary}"
        )
    if report.guidance_eval is not None:
        ev = report.guidance_eval
        print(
            f"Skill Pack ({sim.skill_pack.name}): {ev.unguided_pass_rate_pct:.0f}% -> "
            f"{ev.guided_pass_rate_pct:.0f}% (+{ev.uplift_pp:.0f}pp uplift)"
        )
    print(f"Total Mobile Push Notifications received: {report.push_notifications_received}")


if __name__ == "__main__":
    main()
