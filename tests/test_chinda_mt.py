"""Tests for ChindaMT-4B Thai-English configuration, translator, and Part 4 integration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import pytest

from antigravity_mobile import (
    ANTIGRAVITY_WEB_URL,
    DEFAULT_ARCHITECTURE_GLOSSARY_EN2TH,
    DEFAULT_ARCHITECTURE_GLOSSARY_TH2EN,
    DEFAULT_CIVIL_ENGINEERING_GLOSSARY_EN2TH,
    DEFAULT_CIVIL_ENGINEERING_GLOSSARY_TH2EN,
    DEFAULT_CIVIL_LAW_GLOSSARY_EN2TH,
    DEFAULT_CIVIL_LAW_GLOSSARY_TH2EN,
    DEFAULT_CIVIL_PROCEDURE_GLOSSARY_EN2TH,
    DEFAULT_CIVIL_PROCEDURE_GLOSSARY_TH2EN,
    DEFAULT_COMPUTER_GLOSSARY_EN2TH,
    DEFAULT_COMPUTER_GLOSSARY_TH2EN,
    DEFAULT_CRIMINAL_LAW_GLOSSARY_EN2TH,
    DEFAULT_CRIMINAL_LAW_GLOSSARY_TH2EN,
    DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_EN2TH,
    DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_TH2EN,
    DEFAULT_ENGINEERING_GLOSSARY_EN2TH,
    DEFAULT_ENGINEERING_GLOSSARY_TH2EN,
    DEFAULT_FIDIC_CONTRACT_GLOSSARY_EN2TH,
    DEFAULT_FIDIC_CONTRACT_GLOSSARY_TH2EN,
    DEFAULT_LEGAL_GLOSSARY_EN2TH,
    DEFAULT_PROCEDURAL_LAW_GLOSSARY_EN2TH,
    AndroidChromePWA,
    ChindaTranslator,
    ExecutionMode,
    GoogleAccount,
    SubscriptionPlan,
    TranslatorConfig,
    WebRelay,
    Windows11Workstation,
    WorkstationOfflineError,
    get_domain_glossary,
    run_chinda_demo,
)


@dataclass
class _FakeTensor:
    """Minimal tensor stub supporting `.shape` and `.to(device)`."""

    rows: list[list[int]]
    device: str = "cpu"

    @property
    def shape(self) -> tuple[int, int]:
        width = len(self.rows[0]) if self.rows else 0
        return (len(self.rows), width)

    def to(self, device: str) -> _FakeTensor:
        return _FakeTensor(rows=self.rows, device=device)


class _FakeTokenizer:
    """Minimal Hugging Face AutoTokenizer stub for deterministic unit testing."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.padding_side = "right"
        self.pad_token: str | None = None
        self.eos_token: str = "<|endoftext|>"
        self.pad_token_id: int = 0
        self.chat_template_calls: list[dict[str, Any]] = []
        self._PromptRegistry: list[str] = []

    @classmethod
    def from_pretrained(cls, model_id: str) -> _FakeTokenizer:
        return cls(model_id)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        enable_thinking: bool,
    ) -> str:
        content = messages[0]["content"]
        self.chat_template_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "enable_thinking": enable_thinking,
            }
        )
        return f"<|user|>{content}<|assistant|>"

    def __call__(
        self,
        chat_texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool,
    ) -> dict[str, _FakeTensor]:
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is False
        rows: list[list[int]] = []
        for text in chat_texts:
            idx = len(self._PromptRegistry) + 100
            self._PromptRegistry.append(text)
            # Fixed prompt width of 3 tokens (left-padded)
            rows.append([0, 1, idx])
        return {
            "input_ids": _FakeTensor(rows=rows),
            "attention_mask": _FakeTensor(rows=[[1, 1, 1] for _ in rows]),
        }

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        prompt_idx = token_ids[0] - 100
        chat_text = self._PromptRegistry[prompt_idx]
        if "EN: " in chat_text:
            src = chat_text.split("EN: ", 1)[1].split("<|assistant|>", 1)[0]
            return f"  [TH:{src}]  "
        src = chat_text.split("TH: ", 1)[1].split("<|assistant|>", 1)[0]
        return f"  [EN:{src}]  "


@dataclass
class _FakeBitsAndBytesConfig:
    """Minimal BitsAndBytesConfig stub."""

    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class _FakeModel:
    """Minimal Hugging Face AutoModelForCausalLM stub."""

    model_id: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    device: str = "cuda:0"
    in_eval_mode: bool = False
    generate_calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeModel:
        return cls(model_id=model_id, kwargs=kwargs)

    def eval(self) -> _FakeModel:
        self.in_eval_mode = True
        return self

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        input_ids: _FakeTensor = kwargs["input_ids"]
        outputs: list[list[int]] = []
        for row in input_ids.rows:
            prompt_token = row[-1]
            # Append new generated token after prompt_len (3)
            outputs.append([*row, prompt_token])
        return outputs


class _FakeTorch:
    """Minimal torch module stub."""

    bfloat16 = "torch.bfloat16"
    float16 = "torch.float16"

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


class _FakeTransformers:
    """Minimal transformers module stub."""

    AutoTokenizer = _FakeTokenizer
    AutoModelForCausalLM = _FakeModel
    BitsAndBytesConfig = _FakeBitsAndBytesConfig


def test_translator_config_defaults_match_model_card() -> None:
    cfg = TranslatorConfig()
    assert cfg.model_id == "iapp/ChindaMT-4B"
    assert cfg.torch_dtype == "bfloat16"
    assert cfg.device_map == "auto"
    assert cfg.load_in_4bit is False
    assert cfg.temperature == 0.01
    assert cfg.top_p == 0.7
    assert cfg.top_k == 20
    assert cfg.repetition_penalty == 1.05
    assert cfg.max_new_tokens == 1024
    assert cfg.do_sample is True
    assert cfg.enable_thinking is False
    assert cfg.direction == "en2th"
    assert cfg.rules == ["Return only the translated text"]
    assert cfg.chunk_by_paragraph is True
    assert cfg.max_chars_per_chunk == 1500
    assert cfg.batch_size == 4


def test_build_prompt_formats_en2th_and_th2en_with_and_without_rules() -> None:
    translator = ChindaTranslator()

    en_prompt = translator.build_prompt("The weather is nice today.")
    assert en_prompt == (
        "Translate English to Thai.\n"
        "Rules:\n"
        "- Return only the translated text\n"
        "\n"
        "EN: The weather is nice today."
    )

    th_prompt = translator.build_prompt(
        "วันนี้อากาศดีมาก",
        direction="th2en",
        rules=["Return only the translated text", "Keep technical terms intact"],
    )
    assert th_prompt == (
        "Translate Thai to English.\n"
        "Rules:\n"
        "- Return only the translated text\n"
        "- Keep technical terms intact\n"
        "\n"
        "TH: วันนี้อากาศดีมาก"
    )

    no_rules_prompt = translator.build_prompt("Bridge load test", rules=[])
    assert no_rules_prompt == "Translate English to Thai.\n\nEN: Bridge load test"


def test_chunk_text_packs_paragraphs_and_preserves_blank_separators() -> None:
    cfg = TranslatorConfig(chunk_by_paragraph=True, max_chars_per_chunk=40)
    translator = ChindaTranslator(cfg)

    # Two short paragraphs pack together; third exceeds 40 chars and starts a new chunk
    text = "First short para.\n\nSecond short.\n\nThird paragraph that is longer."
    chunks = translator.chunk_text(text)
    assert chunks == [
        "First short para.\n\nSecond short.",
        "Third paragraph that is longer.",
    ]

    # Blank paragraph in between is preserved as its own chunk
    with_blank = "Para A.\n\n\n\nPara B."
    blank_chunks = translator.chunk_text(with_blank)
    assert blank_chunks == ["Para A.", "", "Para B."]

    # Disabled chunking returns the whole document as a single chunk
    no_chunk_translator = ChindaTranslator(TranslatorConfig(chunk_by_paragraph=False))
    assert no_chunk_translator.chunk_text(text) == [text]


def test_translator_load_and_translate_pipeline_with_fake_torch_and_transformers() -> None:
    cfg = TranslatorConfig(batch_size=2, max_chars_per_chunk=60)
    translator = ChindaTranslator(cfg)

    with pytest.raises(RuntimeError, match="Call load\\(\\) before translating"):
        translator.translate("Hello")

    loaded = translator.load(
        torch_module=_FakeTorch,
        transformers_module=_FakeTransformers,
    )
    assert loaded is translator
    assert translator.tokenizer.padding_side == "left"
    assert translator.tokenizer.pad_token == "<|endoftext|>"
    assert translator.model.in_eval_mode is True
    assert translator.model.kwargs["torch_dtype"] == "torch.bfloat16"

    # Single sentence EN -> TH
    out = translator.translate("The weather is nice today.")
    assert out == "[TH:The weather is nice today.]"
    assert translator.tokenizer.chat_template_calls[-1]["enable_thinking"] is False

    # Reverse direction TH -> EN
    back = translator.translate("อากาศดี", direction="th2en")
    assert back == "[EN:อากาศดี]"

    # Empty batch returns []
    assert translator.translate_batch([]) == []

    # Micro-batched translation (3 items with batch_size=2 -> 2 generate() calls)
    before_calls = len(translator.model.generate_calls)
    batch_out = translator.translate_batch(["One", "Two", "Three"])
    assert batch_out == ["[TH:One]", "[TH:Two]", "[TH:Three]"]
    assert len(translator.model.generate_calls) - before_calls == 2

    # Long text with blank separator preserves spacing
    doc = "Paragraph one is quite long here.\n\n\n\nParagraph two is also quite long here."
    doc_out = translator.translate(doc)
    assert doc_out == (
        "[TH:Paragraph one is quite long here.]\n\n\n\n[TH:Paragraph two is also quite long here.]"
    )

    # Full demo smoke test
    run_chinda_demo(translator=translator)


def test_translator_supports_4bit_quantization_and_fp16() -> None:
    cfg = TranslatorConfig(torch_dtype="float16", load_in_4bit=True)
    translator = ChindaTranslator(cfg).load(
        torch_module=_FakeTorch,
        transformers_module=_FakeTransformers,
    )
    assert translator.model.kwargs["torch_dtype"] == "torch.float16"
    quant_cfg = translator.model.kwargs["quantization_config"]
    assert quant_cfg.load_in_4bit is True
    assert quant_cfg.bnb_4bit_compute_dtype == "torch.float16"


def test_part4_remote_control_dispatches_chindamt_4b_translation() -> None:
    pwa = AndroidChromePWA()
    pwa.open_chrome(ANTIGRAVITY_WEB_URL)
    pwa.sign_in(GoogleAccount(email="eng@example.com", plan=SubscriptionPlan.GOOGLE_AI_PRO))
    pwa.install_to_home_screen()

    workstation = Windows11Workstation(hostname="win11-rtx-16gb")
    relay = WebRelay()

    with pytest.raises(WorkstationOfflineError):
        workstation.run_chinda_translation("Structural inspection report")

    workstation.start_remote_control(via_cli=True)
    pwa.switch_instance(ExecutionMode.REMOTE_CONTROL, relay=relay, workstation=workstation)

    result = relay.dispatch_chinda_translation(
        workstation=workstation,
        text="The bridge inspection is scheduled for next week.",
        direction="en2th",
        service_worker=pwa.service_worker,
    )
    assert result.workload_name == "ChindaMT-4B-translation"
    assert result.used_vram_gb == 9
    assert "iapp/ChindaMT-4B" in result.output_summary
    assert pwa.service_worker.notifications[-1].task_id == "rc-ChindaMT-4B-translation"

    # Also test with a loaded ChindaTranslator instance
    loaded_translator = ChindaTranslator().load(
        torch_module=_FakeTorch,
        transformers_module=_FakeTransformers,
    )
    live_result = relay.dispatch_chinda_translation(
        workstation=workstation,
        text="The bridge inspection is scheduled for next week.",
        translator=loaded_translator,
    )
    assert "[TH:The bridge inspection is scheduled for next week.]" in live_result.output_summary


def test_root_level_config_translator_and_main_modules() -> None:
    import importlib
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    config_mod = importlib.import_module("config")
    translator_mod = importlib.import_module("translator")
    main_mod = importlib.import_module("main")

    assert config_mod.TranslatorConfig is TranslatorConfig
    assert translator_mod.ChindaTranslator is ChindaTranslator
    assert callable(main_mod.main)


def test_glossary_and_anti_transliteration_rules() -> None:
    cfg = TranslatorConfig.formal_engineering_preset(
        direction="en2th",
        extra_glossary={"anchor bolt": "สลักเกลียวสมอ"},
    )
    translator = ChindaTranslator(cfg)

    src = "Inspect the shear wall, prestressed concrete, and anchor bolt."
    prompt = translator.build_prompt(src)

    assert "ห้ามแปลทับศัพท์คำศัพท์เฉพาะทาง" in prompt
    assert '"shear wall" = "ผนังรับแรงเฉือน"' in prompt
    assert '"prestressed concrete" = "คอนกรีตอัดแรง"' in prompt
    assert '"anchor bolt" = "สลักเกลียวสมอ"' in prompt
    # Terms not present in `src` are filtered out by default to keep prompt concise
    assert '"finite element analysis"' not in prompt

    # Reverse direction TH -> EN glossary
    th_cfg = TranslatorConfig.formal_engineering_preset(direction="th2en")
    th_translator = ChindaTranslator(th_cfg)
    th_prompt = th_translator.build_prompt("ตรวจสอบผนังรับแรงเฉือนและคอนกรีตอัดแรง")
    assert "Do not romanize or transliterate Thai technical terms" in th_prompt
    assert '"ผนังรับแรงเฉือน" = "shear wall"' in th_prompt
    assert '"คอนกรีตอัดแรง" = "prestressed concrete"' in th_prompt
    assert DEFAULT_ENGINEERING_GLOSSARY_TH2EN["ผนังรับแรงเฉือน"] == "shear wall"
    assert DEFAULT_ENGINEERING_GLOSSARY_EN2TH["shear wall"] == "ผนังรับแรงเฉือน"


def test_formal_tone_of_voice_presets() -> None:
    translator = ChindaTranslator()

    formal_en2th = translator.build_prompt("Submit the findings.", tone="formal")
    assert "ใช้ภาษาเขียนระดับทางการ" in formal_en2th

    report_en2th = translator.build_prompt("Load test results.", tone="official_report")
    assert "ใช้สำนวนรายงานทางวิชาการและวิศวกรรมที่เป็นทางการ" in report_en2th
    assert "Keep all numerals in Arabic digits" in report_en2th

    academic_th2en = translator.build_prompt(
        "ผลการทดสอบการแอ่นตัว", direction="th2en", tone="academic"
    )
    assert "rigorous academic English register" in academic_th2en


def test_html_and_markdown_preservation_rules_chunking_and_validation() -> None:
    cfg = TranslatorConfig(
        preserve_html_tags=True,
        preserve_markdown=True,
        max_chars_per_chunk=120,
    )
    translator = ChindaTranslator(cfg)

    prompt = translator.build_prompt("Click <a href='/spec'>here</a> for **details**.")
    assert "Preserve all HTML tags, attributes, and structure" in prompt
    assert "Preserve all Markdown syntax" in prompt

    # Fenced code block containing blank lines (\n\n) must NOT be split across chunks
    doc_with_code = (
        "Intro paragraph before code.\n\n"
        "```python\ndef check_wall():\n    x = 1\n\n    return x\n```\n\n"
        "Closing paragraph after code."
    )
    chunks = translator.chunk_text(doc_with_code)
    assert any("def check_wall():\n    x = 1\n\n    return x" in c for c in chunks)

    # Validate markup preservation (passing case)
    src_markup = (
        "Check <strong>shear wall</strong> and `beam_id` in "
        "[Spec](https://example.com/spec):\n```sql\nSELECT * FROM beams;\n```"
    )
    good_translated = (
        "ตรวจสอบ <strong>ผนังรับแรงเฉือน</strong> และ `beam_id` ใน "
        "[ข้อกำหนด](https://example.com/spec):\n```sql\nSELECT * FROM beams;\n```"
    )
    valid = translator.validate_markup_preservation(src_markup, good_translated)
    assert valid.passed is True
    assert valid.missing_html_tags == []
    assert valid.missing_code_blocks == []
    assert valid.missing_inline_codes == []
    assert valid.missing_link_urls == []

    # Validate markup preservation (failing case when tags/code/URL are stripped)
    bad_translated = "ตรวจสอบ ผนังรับแรงเฉือน และ beam_id ใน ข้อกำหนด"
    invalid = translator.validate_markup_preservation(src_markup, bad_translated)
    assert invalid.passed is False
    assert "<strong>" in invalid.missing_html_tags
    assert "</strong>" in invalid.missing_html_tags
    assert "`beam_id`" in invalid.missing_inline_codes
    assert "https://example.com/spec" in invalid.missing_link_urls
    assert len(invalid.missing_code_blocks) == 1


def test_thai_civil_law_and_criminal_law_glossaries_and_legal_preset() -> None:
    # 1. Civil and Commercial Code (กฎหมายแพ่งและพาณิชย์)
    civil_cfg = TranslatorConfig.formal_legal_preset(direction="en2th", domain="civil_law")
    civil_translator = ChindaTranslator(civil_cfg)
    civil_src = (
        "A juristic act that is voidable may be ratified, whereas a wrongful act "
        "or unjust enrichment creates an obligation subject to prescription."
    )
    civil_prompt = civil_translator.build_prompt(civil_src)
    assert "ใช้สำนวนภาษากฎหมายไทยตามประมวลกฎหมายแพ่งและพาณิชย์" in civil_prompt
    assert '"juristic act" = "นิติกรรม"' in civil_prompt
    assert '"voidable" = "โมฆียะ"' in civil_prompt
    assert '"wrongful act" = "ละเมิด"' in civil_prompt
    assert '"unjust enrichment" = "ลาภมิควรได้"' in civil_prompt
    assert '"obligation" = "หนี้"' in civil_prompt
    assert '"prescription" = "อายุความ"' in civil_prompt
    assert DEFAULT_CIVIL_LAW_GLOSSARY_EN2TH["superficies"] == "สิทธิเหนือพื้นดิน"
    assert DEFAULT_CIVIL_LAW_GLOSSARY_TH2EN["ภาระจำยอม"] == "servitude"

    # 2. Criminal / Penal Code (กฎหมายอาญา)
    crim_cfg = TranslatorConfig.formal_legal_preset(direction="en2th", domain="criminal_law")
    crim_translator = ChindaTranslator(crim_cfg)
    crim_src = (
        "The principal and instigator committed embezzlement and fraud with premeditation, "
        "which is a non-compoundable offense unless lawful defense applies."
    )
    crim_prompt = crim_translator.build_prompt(crim_src)
    assert '"principal" = "ตัวการ"' in crim_prompt
    assert '"instigator" = "ผู้ใช้ให้กระทำความผิด"' in crim_prompt
    assert '"embezzlement" = "ยักยอก"' in crim_prompt
    assert '"fraud" = "ฉ้อโกง"' in crim_prompt
    assert '"premeditation" = "ไตร่ตรองไว้ก่อน"' in crim_prompt
    assert '"non-compoundable offense" = "ความผิดอาญาแผ่นดิน"' in crim_prompt
    assert '"lawful defense" = "การป้องกันโดยชอบด้วยกฎหมาย"' in crim_prompt
    assert DEFAULT_CRIMINAL_LAW_GLOSSARY_EN2TH["gang-robbery"] == "ปล้นทรัพย์"
    assert DEFAULT_CRIMINAL_LAW_GLOSSARY_TH2EN["ความผิดลหุโทษ"] == "petty offense"

    # 3. Combined Legal & All-domain Glossaries (TH -> EN and EN -> TH)
    assert "juristic act" in DEFAULT_LEGAL_GLOSSARY_EN2TH
    assert "criminal offense" in DEFAULT_LEGAL_GLOSSARY_EN2TH
    all_th2en = get_domain_glossary("all", direction="th2en")
    assert all_th2en["ผนังรับแรงเฉือน"] == "shear wall"
    assert all_th2en["นิติกรรม"] == "juristic act"
    assert all_th2en["ความผิดอาญาแผ่นดิน"] == "non-compoundable offense"


def test_thai_civil_procedure_criminal_procedure_and_fidic_glossaries() -> None:
    # 1. Civil Procedure Code (กฎหมายวิธีพิจารณาความแพ่ง)
    civ_proc_cfg = TranslatorConfig.formal_legal_preset(direction="en2th", domain="civil_procedure")
    civ_proc_translator = ChindaTranslator(civ_proc_cfg)
    civ_proc_src = (
        "The plaintiff filed a plaint and an application for an injunction, while the "
        "defendant submitted a counterclaim and an interpleader before the settlement of issues."
    )
    civ_proc_prompt = civ_proc_translator.build_prompt(civ_proc_src)
    assert '"plaintiff" = "โจทก์"' in civ_proc_prompt
    assert '"defendant" = "จำเลย"' in civ_proc_prompt
    assert '"plaint" = "คำฟ้อง"' in civ_proc_prompt
    assert '"injunction" = "คำสั่งคุ้มครองชั่วคราว"' in civ_proc_prompt
    assert '"counterclaim" = "ฟ้องแย้ง"' in civ_proc_prompt
    assert '"interpleader" = "ร้องสอด"' in civ_proc_prompt
    assert '"settlement of issues" = "การชี้สองสถาน"' in civ_proc_prompt
    assert DEFAULT_CIVIL_PROCEDURE_GLOSSARY_EN2TH["writ of execution"] == "หมายบังคับคดี"
    assert DEFAULT_CIVIL_PROCEDURE_GLOSSARY_TH2EN["ภาระการพิสูจน์"] == "burden of proof"

    # 2. Criminal Procedure Code (กฎหมายวิธีพิจารณาความอาญา)
    crim_proc_cfg = TranslatorConfig.formal_legal_preset(
        direction="en2th", domain="criminal_procedure"
    )
    crim_proc_translator = ChindaTranslator(crim_proc_cfg)
    crim_proc_src = (
        "The injured person lodged a regular complaint with the inquiry official, "
        "and the public prosecutor issued a prosecution order after thealleged offender "
        "applied for provisional release."
    ).replace("thealleged", "the alleged")
    crim_proc_prompt = crim_proc_translator.build_prompt(crim_proc_src)
    assert '"injured person" = "ผู้เสียหาย"' in crim_proc_prompt
    assert '"regular complaint" = "คำร้องทุกข์"' in crim_proc_prompt
    assert '"inquiry official" = "พนักงานสอบสวน"' in crim_proc_prompt
    assert '"public prosecutor" = "พนักงานอัยการ"' in crim_proc_prompt
    assert '"prosecution order" = "คำสั่งฟ้อง"' in crim_proc_prompt
    assert '"alleged offender" = "ผู้ต้องหา"' in crim_proc_prompt
    assert '"provisional release" = "การปล่อยชั่วคราว"' in crim_proc_prompt
    assert DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_EN2TH["warrant of arrest"] == "หมายจับ"
    assert DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_TH2EN["การไต่สวนมูลฟ้อง"] == "preliminary examination"
    assert "warrant of arrest" in DEFAULT_PROCEDURAL_LAW_GLOSSARY_EN2TH
    assert "counterclaim" in DEFAULT_PROCEDURAL_LAW_GLOSSARY_EN2TH

    # 3. FIDIC Construction Contract (สัญญาจ้างก่อสร้างสากล FIDIC)
    fidic_cfg = TranslatorConfig.formal_fidic_preset(direction="en2th")
    fidic_translator = ChindaTranslator(fidic_cfg)
    fidic_src = (
        "The contractor submitted a notice of claim for an extension of time and "
        "an interim payment certificate to the engineer before the taking-over certificate "
        "and defects notification period."
    )
    fidic_prompt = fidic_translator.build_prompt(fidic_src)
    assert "ใช้สำนวนภาษาสัญญาจ้างก่อสร้างและวิศวกรรมที่ปรึกษาตามมาตรฐานสัญญา FIDIC" in fidic_prompt
    assert '"contractor" = "ผู้รับจ้าง"' in fidic_prompt
    assert '"notice of claim" = "หนังสือแจ้งการใช้สิทธิเรียกร้อง"' in fidic_prompt
    assert '"extension of time" = "การขยายระยะเวลาก่อสร้าง"' in fidic_prompt
    assert '"interim payment certificate" = "หนังสือรับรองการจ่ายเงินงวดระหว่างกาล"' in fidic_prompt
    assert '"the engineer" = "วิศวกรผู้ควบคุมงาน"' in fidic_prompt
    assert '"taking-over certificate" = "หนังสือรับรองการรับมอบงาน"' in fidic_prompt
    assert '"defects notification period" = "ระยะเวลารับประกันความชำรุดบกพร่อง"' in fidic_prompt
    assert DEFAULT_FIDIC_CONTRACT_GLOSSARY_EN2TH["bill of quantities"] == "บัญชีแสดงปริมาณงานและราคา"
    assert DEFAULT_FIDIC_CONTRACT_GLOSSARY_TH2EN["เงินประกันผลงานหักสะสม"] == "retention money"


def test_civil_engineering_and_computer_science_glossaries() -> None:
    # 1. Expanded Civil & Structural Engineering (หมวดวิศวกรรมโยธาและโครงสร้าง)
    civ_eng_cfg = TranslatorConfig.formal_engineering_preset(direction="en2th")
    civ_eng_translator = ChindaTranslator(civ_eng_cfg)
    civ_eng_src = (
        "Check punching shear and differential settlement of thebored pile and "
        "diaphragm wall using standard penetration test and bending moment diagrams."
    ).replace("thebored", "the bored")
    civ_eng_prompt = civ_eng_translator.build_prompt(civ_eng_src)
    assert '"punching shear" = "แรงเฉือนทะลุ"' in civ_eng_prompt
    assert '"differential settlement" = "การทรุดตัวต่างระดับ"' in civ_eng_prompt
    assert '"bored pile" = "เสาเข็มเจาะ"' in civ_eng_prompt
    assert '"diaphragm wall" = "กำแพงกันดินชนิดไดอะแฟรม"' in civ_eng_prompt
    assert '"standard penetration test" = "การทดสอบการตอกมาตรฐาน"' in civ_eng_prompt
    assert '"bending moment" = "โมเมนต์ดัด"' in civ_eng_prompt
    assert DEFAULT_CIVIL_ENGINEERING_GLOSSARY_EN2TH["deformed bar"] == "เหล็กข้ออ้อย"
    assert DEFAULT_CIVIL_ENGINEERING_GLOSSARY_TH2EN["ฐานรากแผ่"] == "spread footing"

    # 2. Computer Science, Software Engineering & AI (หมวดคอมพิวเตอร์และวิศวกรรมซอฟต์แวร์)
    comp_cfg = TranslatorConfig.formal_computer_preset(direction="en2th")
    comp_translator = ChindaTranslator(comp_cfg)
    comp_src = (
        "The algorithm and data structure prevent deadlock and race condition during "
        "concurrency, while retrieval-augmented generation and quantization optimize "
        "the large language model."
    )
    comp_prompt = comp_translator.build_prompt(comp_src)
    assert "ใช้ศัพท์บัญญัติสาขาคอมพิวเตอร์และเทคโนโลยีสารสนเทศของราชบัณฑิตยสภา" in comp_prompt
    assert '"algorithm" = "ขั้นตอนวิธี"' in comp_prompt
    assert '"data structure" = "โครงสร้างข้อมูล"' in comp_prompt
    assert '"deadlock" = "ภาวะติดตาย"' in comp_prompt
    assert '"race condition" = "ภาวะแข่งขัน"' in comp_prompt
    assert '"concurrency" = "การทำงานพร้อมกัน"' in comp_prompt
    assert '"retrieval-augmented generation" = "การสร้างข้อความเสริมด้วยการค้นคืน"' in comp_prompt
    assert '"quantization" = "การลดทอนความละเอียดของค่าถ่วงน้ำหนัก"' in comp_prompt
    assert '"large language model" = "โมเดลภาษาขนาดใหญ่"' in comp_prompt
    assert DEFAULT_COMPUTER_GLOSSARY_EN2TH["binary search tree"] == "ต้นไม้ค้นหาแบบทวิภาค"
    assert DEFAULT_COMPUTER_GLOSSARY_TH2EN["การฉีดพึ่งพา"] == "dependency injection"

    # 3. Architecture, Building Design & Building Code (หมวดงานสถาปัตยกรรมและกฎหมายควบคุมอาคาร)
    arch_cfg = TranslatorConfig.formal_architecture_preset(
        direction="en2th",
        include_civil_engineering=True,
    )
    arch_translator = ChindaTranslator(arch_cfg)
    arch_src = (
        "The reflected ceiling plan and elevation specify a curtain wall, sunshading device, "
        "and fire-enclosed staircase compliant with the floor area ratio and setback distance."
    )
    arch_prompt = arch_translator.build_prompt(arch_src)
    assert "ใช้สำนวนรายการประกอบแบบสถาปัตยกรรมและกฎหมายควบคุมอาคาร" in arch_prompt
    assert '"reflected ceiling plan" = "แปลนฝ้าเพดาน"' in arch_prompt
    assert '"elevation" = "รูปด้าน"' in arch_prompt
    assert '"curtain wall" = "ผนังกระจกเคอร์เทนวอลล์"' in arch_prompt
    assert '"sunshading device" = "แผงบังแดด"' in arch_prompt
    assert '"fire-enclosed staircase" = "บันไดหนีไฟระบบปิด"' in arch_prompt
    assert '"floor area ratio" = "อัตราส่วนพื้นที่อาคารรวมต่อพื้นที่ดิน"' in arch_prompt
    assert '"setback distance" = "ระยะร่นแนวอาคาร"' in arch_prompt
    assert DEFAULT_ARCHITECTURE_GLOSSARY_EN2TH["building envelope"] == "เปลือกอาคาร"
    assert DEFAULT_ARCHITECTURE_GLOSSARY_TH2EN["แปลนฝ้าเพดาน"] == "reflected ceiling plan"

    # 4. Combined all-domain check
    all_en2th = get_domain_glossary("all", direction="en2th")
    assert all_en2th["bored pile"] == "เสาเข็มเจาะ"
    assert all_en2th["curtain wall"] == "ผนังกระจกเคอร์เทนวอลล์"
    assert all_en2th["deadlock"] == "ภาวะติดตาย"
    assert all_en2th["juristic act"] == "นิติกรรม"
    assert all_en2th["taking-over certificate"] == "หนังสือรับรองการรับมอบงาน"
