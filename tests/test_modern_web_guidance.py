"""Tests for modern-web-guidance Skill Pack, Legacy Scanner, and Closed-Loop Eval Harness."""

from antigravity_mobile import (
    ANTIGRAVITY_WEB_URL,
    AndroidChromePWA,
    BrowserSupportPolicy,
    EvalTask,
    GoogleAccount,
    GuidanceEvalHarness,
    ModelTier,
    NonBlockingDispatcher,
    RemoteProject,
    SkillPack,
    SubscriptionPlan,
)


def test_progressive_disclosure_search_retrieve_and_disclosure() -> None:
    skill = SkillPack()
    assert skill.catalog_size == 212
    assert skill.index_token_budget == 205
    assert "antigravity_cli" in skill.supported_agents
    assert "claude_code" in skill.supported_agents

    # Step 1: Search returns lightweight metadata only
    matches = skill.search("light dismiss dialog")
    assert any(m["id"] == "light-dismiss-dialog" for m in matches)

    # Step 2: Retrieve full guide on demand
    guides = skill.retrieve(["light-dismiss-dialog"])
    assert len(guides) == 1

    # Step 4: Disclosure formatting
    disclosure = skill.format_disclosure(guides)
    assert "### Modern Web Guidance" in disclosure
    assert "`light-dismiss-dialog`" in disclosure


def test_baseline_policy_enforces_fallbacks_for_newly_available_features() -> None:
    skill = SkillPack()
    guides = skill.retrieve(["light-dismiss-dialog"])

    # Missing fallback fails default Baseline policy
    missing_fallback_code = "<dialog>elem.showModal()</dialog>"
    default_check = skill.verify_compliance(missing_fallback_code, guides)
    assert default_check.passed is False
    assert "light-dismiss-dialog" in default_check.missing_fallbacks

    # Adding fallback passes default policy
    with_fallback = "<dialog>elem.showModal() /* closedby-click-backdrop */</dialog>"
    assert skill.verify_compliance(with_fallback, guides).passed is True

    # Custom policy targeting Baseline 2025 allows Newly Available without fallback
    policy_2025 = BrowserSupportPolicy(baseline_year_target=2025)
    assert skill.verify_compliance(missing_fallback_code, guides, policy=policy_2025).passed is True


def test_legacy_scan_filters_safe_fallbacks_via_drop_when_rubric() -> None:
    skill = SkillPack()

    legacy_only = "<div role='dialog'>Legacy Modal</div>"
    findings = skill.legacy_scan(legacy_only)
    assert len(findings) == 1
    assert findings[0].guide_id == "light-dismiss-dialog"

    # Safe fallback present -> dropped from active findings
    with_safe_fallback = "<dialog>showModal()</dialog><div role='dialog'>Fallback</div>"
    assert skill.legacy_scan(with_safe_fallback) == []

    dropped = skill.legacy_scan(with_safe_fallback, include_dropped=True)
    assert len(dropped) == 1
    assert dropped[0].dropped_as_fallback is True


def test_closed_loop_eval_harness_calibration_and_uplift() -> None:
    skill = SkillPack()
    harness = GuidanceEvalHarness(skill_pack=skill)
    tasks = [
        EvalTask(
            task_id="t1",
            prompt="confirm dialog",
            guide_id="light-dismiss-dialog",
            golden_code="<dialog>showModal() closedby-click-backdrop</dialog>",
            unmodified_base_code="<div role='dialog'>Confirm</div>",
        ),
        EvalTask(
            task_id="t2",
            prompt="popover menu",
            guide_id="popover-hint-api",
            golden_code="<div popover='auto'>Menu</div>",
            unmodified_base_code="<div class='portal-menu'>Menu</div>",
        ),
        EvalTask(
            task_id="t3",
            prompt="parent selector styling",
            guide_id="css-has-parent-selector",
            golden_code=".group:has(input:invalid) { color: red; }",
            unmodified_base_code=".group.invalid { color: red; }",
        ),
        EvalTask(
            task_id="t4",
            prompt="scroll reveal animation",
            guide_id="scroll-driven-animations",
            golden_code=(
                "@supports (animation-timeline: scroll()) { animation-timeline: scroll(); }"
            ),
            unmodified_base_code="window.addEventListener('scroll', onScroll);",
        ),
    ]

    calibration = harness.calibrate(tasks)
    assert calibration.calibrated is True
    assert calibration.golden_pass_rate_pct == 100.0
    assert calibration.base_pass_rate_pct == 0.0

    # Simulate unguided (50% pass) -> guided (100% pass)
    result = harness.run_benchmark(
        agent_name="antigravity_cli",
        model_tier=ModelTier.ARGON,
        tasks=tasks,
        unguided_outputs={
            "t1": tasks[0].unmodified_base_code,
            "t2": tasks[1].golden_code,
            "t3": tasks[2].golden_code,
            "t4": tasks[3].unmodified_base_code,
        },
        guided_outputs={t.task_id: t.golden_code for t in tasks},
    )
    assert result.unguided_pass_rate_pct == 50.0
    assert result.guided_pass_rate_pct == 100.0
    assert result.opportunity_pct == 50.0
    assert result.uplift_pp == 50.0


def test_dispatcher_attaches_skill_pack_guidance_to_cloud_task() -> None:
    pwa = AndroidChromePWA()
    pwa.open_chrome(ANTIGRAVITY_WEB_URL)
    pwa.sign_in(GoogleAccount(email="dev@example.com", plan=SubscriptionPlan.GOOGLE_AI_PRO))
    pwa.install_to_home_screen()

    project = RemoteProject(project_id="proj-web", name="ModernWebApp")
    project.attach_github_repo("4weng/antigravity-mobile-architecture")

    dispatcher = NonBlockingDispatcher()
    skill = SkillPack()
    task = dispatcher.dispatch(
        pwa=pwa,
        project=project,
        prompt="Add accessible popover menu",
        model_tier=ModelTier.HYDROGEN,
        skill_pack=skill,
        guide_query="popover menu",
    )
    assert task.guided_by_skill == "modern-web-guidance"
    assert "popover-hint-api" in task.applied_guide_ids
    assert "### Modern Web Guidance" in task.disclosure_markdown
    assert "Guided by modern-web-guidance" in task.output_summary
