"""Part 4 simulation: Windows 11 Remote Control Mode, `agy --remote-agent`, and Web Relay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .chinda_mt import ChindaTranslator, Direction, TranslatorConfig
from .models import (
    ExecutionMode,
    PushNotification,
    WorkloadResult,
    WorkstationOfflineError,
    WorkstationSpec,
)

if TYPE_CHECKING:
    from .pwa_client import ServiceWorker


@dataclass
class Windows11Workstation:
    """Home Windows 11 Workstation supporting Remote Control Mode (Part 4)."""

    hostname: str = "home-win11-workstation"
    spec: WorkstationSpec = field(default_factory=WorkstationSpec)
    local_files: dict[str, str] = field(default_factory=dict)
    desktop_app_running: bool = False
    remote_control_enabled: bool = False
    cli_command_used: str | None = None

    def start_remote_control(self, via_cli: bool = False) -> str:
        """Enable Remote Control via Antigravity Desktop App or `agy --remote-agent` CLI."""
        self.desktop_app_running = True
        self.remote_control_enabled = True
        if via_cli:
            self.cli_command_used = "agy --remote-agent"
            return f"Remote Control started via CLI (`{self.cli_command_used}`) on {self.hostname}."
        self.cli_command_used = None
        return f"Remote Control started via Antigravity Desktop App on {self.hostname}."

    def stop_remote_control(self) -> None:
        """Shut down or disable Remote Control on the home Windows 11 PC."""
        self.remote_control_enabled = False
        self.desktop_app_running = False
        self.cli_command_used = None

    def run_local_workload(self, workload_name: str, input_path: str) -> WorkloadResult:
        """Execute a heavy local RAM/GPU workload using files on the Windows 11 hard drive."""
        if not self.remote_control_enabled:
            raise WorkstationOfflineError(
                f"Workstation '{self.hostname}' is offline or Remote Control is not enabled."
            )
        if input_path not in self.local_files:
            raise FileNotFoundError(
                f"Local file '{input_path}' not found on workstation '{self.hostname}'."
            )

        file_desc = self.local_files[input_path]
        return WorkloadResult(
            workload_name=workload_name,
            input_path=input_path,
            used_ram_gb=min(64, self.spec.ram_gb),
            used_vram_gb=min(14, self.spec.vram_gb),
            output_summary=(
                f"Executed '{workload_name}' on {self.spec.os_name} "
                f"({self.spec.ram_gb} GB RAM, {self.spec.gpu_name} {self.spec.vram_gb} GB VRAM) "
                f"using local input '{input_path}' ({file_desc})."
            ),
            bandwidth_mode="lightweight_command_relay",
        )

    def run_chinda_translation(
        self,
        text: str,
        *,
        direction: Direction | None = None,
        rules: Sequence[str] | None = None,
        config: TranslatorConfig | None = None,
        translator: ChindaTranslator | None = None,
        weights_path: str = r"D:\models\ChindaMT-4B\weights.safetensors",
    ) -> WorkloadResult:
        """Configure and run ChindaMT-4B (Thai-English) on the Windows 11 GPU workstation."""
        if not self.remote_control_enabled:
            raise WorkstationOfflineError(
                f"Workstation '{self.hostname}' is offline or Remote Control is not enabled."
            )

        active_translator = translator or ChindaTranslator(config)
        cfg = active_translator.config
        active_direction: Direction = direction or cfg.direction
        chunks = [c for c in active_translator.chunk_text(text) if c.strip()]
        prompt_preview = active_translator.build_prompt(
            chunks[0] if chunks else text,
            direction=active_direction,
            rules=rules,
        )

        if active_translator.model is not None and active_translator.tokenizer is not None:
            translated_output = active_translator.translate(
                text, direction=active_direction, rules=rules
            )
        else:
            translated_output = (
                f"[{cfg.model_id} {active_direction} ({cfg.torch_dtype}) "
                f"translated {len(chunks)} chunk(s)]"
            )

        vram_needed = 4 if cfg.load_in_4bit else 9
        return WorkloadResult(
            workload_name="ChindaMT-4B-translation",
            input_path=weights_path,
            used_ram_gb=min(16, self.spec.ram_gb),
            used_vram_gb=min(vram_needed, self.spec.vram_gb),
            output_summary=(
                f"Executed '{cfg.model_id}' ({active_direction}, {cfg.torch_dtype}, "
                f"chunks={len(chunks)}) on {self.spec.os_name} "
                f"({self.spec.ram_gb} GB RAM, {self.spec.gpu_name} {self.spec.vram_gb} GB VRAM) "
                f"-> {translated_output} | Prompt: {prompt_preview.splitlines()[0]}"
            ),
            bandwidth_mode="lightweight_command_relay",
        )


@dataclass
class WebRelay:
    """Lightweight Web Relay connecting Android PWA to home Windows 11 PC (Part 4).

    Unlike traditional Remote Desktop (RDP/VNC) that streams heavy video pixels,
    Web Relay sends lightweight command payloads and returns concise status/notifications.
    """

    relay_endpoint: str = "wss://relay.antigravity.google.com"
    bandwidth_mode: str = "lightweight_command_relay"
    pixel_streaming_enabled: bool = False

    def verify_connection(self, workstation: Windows11Workstation) -> bool:
        """Verify that the target Windows 11 PC has Remote Control active."""
        if not workstation.remote_control_enabled:
            raise WorkstationOfflineError(
                f"Cannot connect via {self.relay_endpoint}: '{workstation.hostname}' "
                "has not started Antigravity Desktop App or `agy --remote-agent`."
            )
        return True

    def dispatch_command(
        self,
        workstation: Windows11Workstation,
        workload_name: str,
        input_path: str,
        service_worker: ServiceWorker | None = None,
    ) -> WorkloadResult:
        """Forward a lightweight command from Android PWA to the home Windows 11 PC."""
        self.verify_connection(workstation)
        result = workstation.run_local_workload(workload_name=workload_name, input_path=input_path)

        if service_worker is not None and service_worker.active:
            service_worker.receive_push(
                PushNotification(
                    task_id=f"rc-{workload_name}",
                    title=f"Remote Control Completed: {workload_name}",
                    body=result.output_summary,
                    action_required=False,
                    mode=ExecutionMode.REMOTE_CONTROL,
                )
            )
        return result

    def dispatch_chinda_translation(
        self,
        workstation: Windows11Workstation,
        text: str,
        *,
        direction: Direction | None = None,
        rules: Sequence[str] | None = None,
        config: TranslatorConfig | None = None,
        translator: ChindaTranslator | None = None,
        service_worker: ServiceWorker | None = None,
    ) -> WorkloadResult:
        """Dispatch a ChindaMT-4B Thai-English translation job over Web Relay."""
        self.verify_connection(workstation)
        result = workstation.run_chinda_translation(
            text=text,
            direction=direction,
            rules=rules,
            config=config,
            translator=translator,
        )

        if service_worker is not None and service_worker.active:
            service_worker.receive_push(
                PushNotification(
                    task_id="rc-ChindaMT-4B-translation",
                    title="Remote Control Completed: ChindaMT-4B-translation",
                    body=result.output_summary,
                    action_required=False,
                    mode=ExecutionMode.REMOTE_CONTROL,
                )
            )
        return result
