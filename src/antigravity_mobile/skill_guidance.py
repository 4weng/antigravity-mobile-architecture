"""Skill Pack & Closed-Loop Eval Harness inspired by GoogleChrome/modern-web-guidance.

Implements the core principles verified from https://github.com/GoogleChrome/modern-web-guidance:
1. Progressive Disclosure (`SKILL.md` ~205 tokens upfront -> `search` -> `retrieve` -> `verify`).
2. Baseline Web Platform & Fallback Policy (Widely Available vs. Newly Available + fallbacks).
3. Legacy Pattern Scanner (`legacy-scan`) with `keepWhen` / `dropWhen` false-positive filtering.
4. Closed-Loop Eval Calibration (Golden = 100%, Base = 0%) and Unguided vs. Guided Uplift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .models import ModelTier


class BaselineSupport(StrEnum):
    """Web Platform Baseline availability tier."""

    WIDELY_AVAILABLE = "widely_available"
    NEWLY_AVAILABLE = "newly_available"


@dataclass(frozen=True)
class WebGuide:
    """Single best-practice guide stored in the guidance catalog."""

    guide_id: str
    title: str
    description: str
    keywords: tuple[str, ...]
    baseline_support: BaselineSupport
    baseline_since_year: int
    modern_pattern: str
    fallback_strategy: str | None
    rule_quote: str
    tokens_cost: int = 420


@dataclass(frozen=True)
class BrowserSupportPolicy:
    """Project browser-support policy controlling fallback requirements."""

    allow_newly_available_without_fallback: bool = False
    baseline_year_target: int | None = None

    def requires_fallback(self, guide: WebGuide) -> bool:
        """Return True if a Newly Available feature needs a fallback under this policy."""
        if guide.baseline_support == BaselineSupport.WIDELY_AVAILABLE:
            return False
        if self.allow_newly_available_without_fallback:
            return False
        if self.baseline_year_target is not None:
            return guide.baseline_since_year > self.baseline_year_target
        return True


@dataclass(frozen=True)
class ComplianceResult:
    """Verification outcome checking modern pattern and required fallback usage."""

    passed: bool
    missing_patterns: list[str]
    missing_fallbacks: list[str]


@dataclass(frozen=True)
class LegacyRule:
    """Deterministic scan rule with `keepWhen` / `dropWhen` rubric."""

    rule_id: str
    guide_id: str
    legacy_substring: str
    safe_fallback_markers: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True)
class LegacyFinding:
    """Verified legacy finding after dropping fallback/false-positive matches."""

    rule_id: str
    guide_id: str
    recommendation: str
    dropped_as_fallback: bool


def default_web_guides() -> tuple[WebGuide, ...]:
    """Curated subset representing modern CSS, DOM, and PWA best practices."""
    return (
        WebGuide(
            guide_id="light-dismiss-dialog",
            title="Light-dismiss modal dialog",
            description="Native HTML <dialog> with closedby='any' and @starting-style animation",
            keywords=("dialog", "modal", "light", "dismiss", "confirmation"),
            baseline_support=BaselineSupport.NEWLY_AVAILABLE,
            baseline_since_year=2025,
            modern_pattern="showModal()",
            fallback_strategy="closedby-click-backdrop",
            rule_quote="Use native <dialog> showModal() with backdrop click fallback.",
        ),
        WebGuide(
            guide_id="popover-hint-api",
            title="Declarative Popover API",
            description="Native popover attribute and CSS anchor positioning for overlays",
            keywords=("popover", "tooltip", "overlay", "anchor", "menu"),
            baseline_support=BaselineSupport.WIDELY_AVAILABLE,
            baseline_since_year=2024,
            modern_pattern="popover=",
            fallback_strategy=None,
            rule_quote="Prefer declarative HTML popover attributes over manual z-index portals.",
        ),
        WebGuide(
            guide_id="css-has-parent-selector",
            title="Relational :has() styling",
            description="Style parent containers declaratively with CSS :has() without JS classes",
            keywords=("css", "layout", "parent", "selector", "has", "form", "validation"),
            baseline_support=BaselineSupport.WIDELY_AVAILABLE,
            baseline_since_year=2023,
            modern_pattern=":has(",
            fallback_strategy=None,
            rule_quote="Use CSS :has() instead of toggling state classes from JavaScript.",
        ),
        WebGuide(
            guide_id="scroll-driven-animations",
            title="CSS Scroll-Driven Animations",
            description="Compositor-thread scroll() / view() timelines replacing scroll listeners",
            keywords=("scroll", "animation", "parallax", "reveal", "performance"),
            baseline_support=BaselineSupport.NEWLY_AVAILABLE,
            baseline_since_year=2025,
            modern_pattern="animation-timeline:",
            fallback_strategy="@supports (animation-timeline: scroll())",
            rule_quote="Gate animation-timeline with @supports and keep static fallback visible.",
        ),
        WebGuide(
            guide_id="pwa-service-worker-push",
            title="PWA Service Worker & Web Push",
            description="Background task notifications and standalone display for mobile PWAs",
            keywords=("pwa", "service", "worker", "push", "notification", "mobile"),
            baseline_support=BaselineSupport.WIDELY_AVAILABLE,
            baseline_since_year=2023,
            modern_pattern="navigator.serviceWorker.register",
            fallback_strategy=None,
            rule_quote="Register Service Worker at root scope for reliable background push events.",
        ),
    )


def default_legacy_rules() -> tuple[LegacyRule, ...]:
    """Deterministic legacy-scan rules matching ~20% subset heuristic."""
    return (
        LegacyRule(
            rule_id="legacy-modal-div",
            guide_id="light-dismiss-dialog",
            legacy_substring="role='dialog'",
            safe_fallback_markers=("showModal()", "closedby-click-backdrop"),
            recommendation="Replace custom div[role='dialog'] with native <dialog>.showModal().",
        ),
        LegacyRule(
            rule_id="legacy-scroll-listener",
            guide_id="scroll-driven-animations",
            legacy_substring="addEventListener('scroll'",
            safe_fallback_markers=("@supports (animation-timeline: scroll())", "/* fallback */"),
            recommendation="Prefer CSS animation-timeline over main-thread scroll event listeners.",
        ),
    )


@dataclass
class SkillPack:
    """Progressive-disclosure AI skill pack modeled on GoogleChrome/modern-web-guidance."""

    name: str = "modern-web-guidance"
    source_repo: str = "https://github.com/GoogleChrome/modern-web-guidance"
    install_command: str = "npx modern-web-guidance@latest install"
    catalog_size: int = 212
    index_token_budget: int = 205
    supported_agents: tuple[str, ...] = (
        "antigravity_cli",
        "claude_code",
        "codex_cli",
        "github_copilot",
        "cursor",
    )
    guides: tuple[WebGuide, ...] = field(default_factory=default_web_guides)
    legacy_rules: tuple[LegacyRule, ...] = field(default_factory=default_legacy_rules)

    def search(self, query: str) -> list[dict[str, str]]:
        """Step 1: Search concise guide index without loading full guide bodies."""
        terms = {t.strip().lower() for t in query.split() if t.strip()}
        matches: list[dict[str, str]] = []
        for guide in self.guides:
            haystack = set(guide.keywords) | set(guide.description.lower().split())
            if terms & haystack:
                matches.append({"id": guide.guide_id, "description": guide.description})
        return matches

    def retrieve(self, guide_ids: list[str]) -> list[WebGuide]:
        """Step 2: Retrieve full guides on demand for matched IDs only."""
        by_id = {g.guide_id: g for g in self.guides}
        return [by_id[gid] for gid in guide_ids if gid in by_id]

    def verify_compliance(
        self,
        code: str,
        guides: list[WebGuide],
        policy: BrowserSupportPolicy | None = None,
    ) -> ComplianceResult:
        """Step 3: Cross-check generated code for modern patterns & Baseline fallbacks."""
        active_policy = policy or BrowserSupportPolicy()
        missing_patterns: list[str] = []
        missing_fallbacks: list[str] = []

        for guide in guides:
            if guide.modern_pattern not in code:
                missing_patterns.append(guide.guide_id)
            if (
                active_policy.requires_fallback(guide)
                and guide.fallback_strategy is not None
                and guide.fallback_strategy not in code
            ):
                missing_fallbacks.append(guide.guide_id)

        return ComplianceResult(
            passed=not missing_patterns and not missing_fallbacks,
            missing_patterns=missing_patterns,
            missing_fallbacks=missing_fallbacks,
        )

    def format_disclosure(self, guides: list[WebGuide]) -> str:
        """Step 4: Format the required `### Modern Web Guidance` disclosure block."""
        if not guides:
            return (
                "### Modern Web Guidance\n"
                "I searched `modern-web-guidance` for this task, "
                "but no web-platform guide applied."
            )
        bullets = [
            f'- **`{g.guide_id}`**: Applied `{g.modern_pattern}` ("{g.rule_quote}").'
            for g in guides
        ]
        return "### Modern Web Guidance\n" + "\n".join(bullets)

    def legacy_scan(
        self,
        source_code: str,
        include_dropped: bool = False,
    ) -> list[LegacyFinding]:
        """Alternative flow: Scan legacy patterns and apply `dropWhen` fallback filter."""
        findings: list[LegacyFinding] = []
        for rule in self.legacy_rules:
            if rule.legacy_substring not in source_code:
                continue
            is_fallback = any(marker in source_code for marker in rule.safe_fallback_markers)
            if is_fallback and not include_dropped:
                continue
            findings.append(
                LegacyFinding(
                    rule_id=rule.rule_id,
                    guide_id=rule.guide_id,
                    recommendation=rule.recommendation,
                    dropped_as_fallback=is_fallback,
                )
            )
        return findings


@dataclass(frozen=True)
class EvalTask:
    """Single evaluation task with golden solution, baseline app, and required guide."""

    task_id: str
    prompt: str
    guide_id: str
    golden_code: str
    unmodified_base_code: str


@dataclass(frozen=True)
class EvalCalibrationResult:
    """Calibration ensures Golden = 100% pass and Unmodified Base = 0% pass."""

    calibrated: bool
    golden_pass_rate_pct: float
    base_pass_rate_pct: float


@dataclass(frozen=True)
class EvalBenchmarkResult:
    """Comparison of unguided vs. guided agent pass rates."""

    agent_name: str
    model_tier: ModelTier
    tasks_evaluated: int
    unguided_pass_rate_pct: float
    guided_pass_rate_pct: float
    opportunity_pct: float
    uplift_pp: float


@dataclass
class GuidanceEvalHarness:
    """Closed-loop eval harness modeled after GoogleChrome/modern-web-guidance."""

    skill_pack: SkillPack = field(default_factory=SkillPack)
    policy: BrowserSupportPolicy = field(default_factory=BrowserSupportPolicy)

    def calibrate(self, tasks: list[EvalTask]) -> EvalCalibrationResult:
        """Verify graders pass 100% of golden solutions and 0% of unmodified baselines."""
        if not tasks:
            raise ValueError("Calibration requires at least one EvalTask.")

        golden_passes = 0
        base_passes = 0
        for task in tasks:
            guides = self.skill_pack.retrieve([task.guide_id])
            if self.skill_pack.verify_compliance(task.golden_code, guides, self.policy).passed:
                golden_passes += 1
            if self.skill_pack.verify_compliance(
                task.unmodified_base_code, guides, self.policy
            ).passed:
                base_passes += 1

        total = len(tasks)
        golden_pct = (golden_passes / total) * 100.0
        base_pct = (base_passes / total) * 100.0
        return EvalCalibrationResult(
            calibrated=(golden_pct == 100.0 and base_pct == 0.0),
            golden_pass_rate_pct=golden_pct,
            base_pass_rate_pct=base_pct,
        )

    def run_benchmark(
        self,
        agent_name: str,
        model_tier: ModelTier,
        tasks: list[EvalTask],
        unguided_outputs: dict[str, str],
        guided_outputs: dict[str, str],
    ) -> EvalBenchmarkResult:
        """Measure unguided vs. guided pass rates, opportunity, and percentage-point uplift."""
        if not tasks:
            raise ValueError("Benchmark requires at least one EvalTask.")

        unguided_passes = 0
        guided_passes = 0
        for task in tasks:
            guides = self.skill_pack.retrieve([task.guide_id])
            unguided_code = unguided_outputs.get(task.task_id, task.unmodified_base_code)
            guided_code = guided_outputs.get(task.task_id, task.unmodified_base_code)

            if self.skill_pack.verify_compliance(unguided_code, guides, self.policy).passed:
                unguided_passes += 1
            if self.skill_pack.verify_compliance(guided_code, guides, self.policy).passed:
                guided_passes += 1

        total = len(tasks)
        unguided_pct = round((unguided_passes / total) * 100.0, 1)
        guided_pct = round((guided_passes / total) * 100.0, 1)
        return EvalBenchmarkResult(
            agent_name=agent_name,
            model_tier=model_tier,
            tasks_evaluated=total,
            unguided_pass_rate_pct=unguided_pct,
            guided_pass_rate_pct=guided_pct,
            opportunity_pct=round(100.0 - unguided_pct, 1),
            uplift_pp=round(guided_pct - unguided_pct, 1),
        )
