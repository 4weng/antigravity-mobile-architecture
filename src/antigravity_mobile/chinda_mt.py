"""ChindaMT-4B (Thai-English) configuration and translator module.

Implements the paper and Hugging Face model-card inference pipeline for
`iapp/ChindaMT-4B`, including:
- Prompt formatting with auxiliary rules (`Rules:`)
- Domain Glossary enforcement & anti-transliteration (ห้ามแปลทับศัพท์คำเฉพาะทาง)
- Formal Tone of Voice presets (`formal`, `official_report`, `academic`)
- HTML tag & Markdown syntax preservation (fenced-code-aware chunking + validation)
- Paragraph-aware document chunking and micro-batched causal generation
- Optional 4-bit quantization (`load_in_4bit=True`) or `bfloat16`/`float16` execution
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Direction = Literal["en2th", "th2en"]
ToneOfVoice = Literal[
    "default",
    "formal",
    "official_report",
    "academic",
    "legal",
    "fidic_contract",
    "technical_software",
    "architectural_spec",
]
GlossaryDomain = Literal[
    "engineering",
    "civil_engineering",
    "architecture",
    "computer",
    "civil_law",
    "criminal_law",
    "civil_procedure",
    "criminal_procedure",
    "procedural_law",
    "legal",
    "fidic",
    "all",
]

# Tags used in the official prompt template.
_LANG: dict[Direction, tuple[str, str, str]] = {
    "en2th": ("English", "Thai", "EN"),
    "th2en": ("Thai", "English", "TH"),
}

# 1. Civil & Structural Engineering glossary (หมวดวิศวกรรมโยธาและโครงสร้าง ตามมาตรฐาน วสท./มยผ.)
DEFAULT_CIVIL_ENGINEERING_GLOSSARY_EN2TH: dict[str, str] = {
    # Structural Analysis & Design (การวิเคราะห์และออกแบบโครงสร้าง)
    "reinforcement": "เหล็กเสริม",
    "deformed bar": "เหล็กข้ออ้อย",
    "round bar": "เหล็กเส้นกลม",
    "stirrup": "เหล็กปลอก",
    "dowel bar": "เหล็กเดือย",
    "prestressed concrete": "คอนกรีตอัดแรง",
    "post-tensioned slab": "พื้นคอนกรีตอัดแรงชนิดดึงทีหลัง",
    "reinforced concrete": "คอนกรีตเสริมเหล็ก",
    "shear wall": "ผนังรับแรงเฉือน",
    "retaining wall": "กำแพงกันดิน",
    "cantilever beam": "คานยื่น",
    "continuous beam": "คานต่อเนื่อง",
    "simply supported beam": "คานช่วงเดี่ยว",
    "transfer beam": "คานถ่ายแรง",
    "waffle slab": "พื้นไร้คานชนิดตงสองทาง",
    "flat slab": "พื้นไร้คาน",
    "bending moment": "โมเมนต์ดัด",
    "shear force": "แรงเฉือน",
    "axial load": "แรงตามแนวแกน",
    "torsional moment": "โมเมนต์บิด",
    "punching shear": "แรงเฉือนทะลุ",
    "buckling": "การโก่งเดาะ",
    "lateral-torsional buckling": "การโก่งเดาะด้านข้างปนบิด",
    "deflection": "การแอ่นตัว",
    "inter-story drift": "การเคลื่อนตัวสัมพัทธ์ระหว่างชั้น",
    "modulus of elasticity": "โมดูลัสยืดหยุ่น",
    "moment of inertia": "โมเมนต์ความเฉื่อย",
    "section modulus": "โมดูลัสหน้าตัด",
    "yield strength": "กำลังคราก",
    "ultimate tensile strength": "กำลังดึงประลัย",
    "compressive strength": "กำลังรับแรงอัด",
    "flexural strength": "กำลังรับแรงดัด",
    "allowable stress design": "การออกแบบโดยวิธีหน่วยแรงที่ยอมให้",
    "strength design method": "การออกแบบโดยวิธีกำลัง",
    "load and resistance factor design": "การออกแบบด้วยตัวคูณน้ำหนักบรรทุกและความต้านทาน",
    "dead load": "น้ำหนักบรรทุกคงที่",
    "live load": "น้ำหนักบรรทุกจร",
    "wind load": "แรงลม",
    "seismic load": "แรงกระทำแผ่นดินไหว",
    "response spectrum analysis": "การวิเคราะห์สเปกตรัมการตอบสนอง",
    "base shear": "แรงเฉือนที่ฐาน",
    "ductility": "ความเหนียวของโครงสร้าง",
    "structural integrity": "ความมั่นคงแข็งแรงของโครงสร้าง",
    "finite element analysis": "การวิเคราะห์ไฟไนต์เอลิเมนต์",
    "crack propagation": "การลุกลามของรอยร้าว",
    "safety factor": "อัตราส่วนความปลอดภัย",
    "non-destructive testing": "การทดสอบแบบไม่ทำลาย",
    # Geotechnical & Foundation Engineering (วิศวกรรมปฐพีและฐานราก)
    "bearing capacity": "กำลังรับน้ำหนักบรรทุก",
    "allowable bearing capacity": "กำลังรับน้ำหนักบรรทุกปลอดภัย",
    "ultimate bearing capacity": "กำลังรับน้ำหนักบรรทุกประลัย",
    "shallow foundation": "ฐานรากตื้น",
    "deep foundation": "ฐานรากลึก",
    "spread footing": "ฐานรากแผ่",
    "mat foundation": "ฐานรากแพ",
    "raft foundation": "ฐานรากแพ",
    "pile cap": "ตอม่อหัวเสาเข็ม",
    "driven pile": "เสาเข็มตอก",
    "bored pile": "เสาเข็มเจาะ",
    "skin friction": "แรงเสียดทานรอบผิวเสาเข็ม",
    "end bearing": "แรงแบกทานที่ปลายเสาเข็ม",
    "differential settlement": "การทรุดตัวต่างระดับ",
    "consolidation settlement": "การทรุดตัวเนื่องจากการอัดตัวคายน้ำ",
    "soil liquefaction": "การเกิดสภาพดินเหลว",
    "standard penetration test": "การทดสอบการตอกมาตรฐาน",
    "slope stability": "เสถียรภาพของลาดดิน",
    "diaphragm wall": "กำแพงกันดินชนิดไดอะแฟรม",
    "sheet pile": "เข็มพืด",
    # Concrete, Construction & Surveying (คอนกรีต งานก่อสร้าง และการสำรวจ)
    "water-cement ratio": "อัตราส่วนน้ำต่อซีเมนต์",
    "slump test": "การทดสอบค่ายุบตัวของคอนกรีต",
    "curing of concrete": "การบ่มคอนกรีต",
    "shrinkage": "การหดตัว",
    "creep": "การคืบ",
    "honeycomb": "โพรงรังผึ้งในคอนกรีต",
    "formwork": "ไม้แบบหล่อคอนกรีต",
    "scaffolding": "นั่งร้าน",
    "shoring": "ค้ำยันชั่วคราว",
    "construction joint": "รอยต่อเพื่อการก่อสร้าง",
    "expansion joint": "รอยต่อเผื่อขยาย",
    "shop drawing": "แบบขยายรายละเอียดเพื่อการก่อสร้าง",
    "as-built drawing": "แบบก่อสร้างจริง",
    "contour line": "เส้นชั้นความสูง",
    "benchmark": "หมุดหลักฐานระดับ",
    "right of way": "เขตทาง",
}

DEFAULT_CIVIL_ENGINEERING_GLOSSARY_TH2EN: dict[str, str] = {}
for _en_term, _th_term in DEFAULT_CIVIL_ENGINEERING_GLOSSARY_EN2TH.items():
    DEFAULT_CIVIL_ENGINEERING_GLOSSARY_TH2EN.setdefault(_th_term, _en_term)

# Backward-compatible aliases
DEFAULT_ENGINEERING_GLOSSARY_EN2TH = DEFAULT_CIVIL_ENGINEERING_GLOSSARY_EN2TH
DEFAULT_ENGINEERING_GLOSSARY_TH2EN = DEFAULT_CIVIL_ENGINEERING_GLOSSARY_TH2EN

# 2. Thai Civil and Commercial Code glossary (หมวดกฎหมายแพ่งและพาณิชย์ของไทย)
DEFAULT_CIVIL_LAW_GLOSSARY_EN2TH: dict[str, str] = {
    "juristic act": "นิติกรรม",
    "declaration of intention": "การแสดงเจตนา",
    "juristic person": "นิติบุคคล",
    "natural person": "บุคคลธรรมดา",
    "void": "โมฆะ",
    "voidable": "โมฆียะ",
    "ratification": "สัตยาบัน",
    "prescription": "อายุความ",
    "statute of limitations": "อายุความ",
    "obligation": "หนี้",
    "specific performance": "การชำระหนี้เฉพาะเจาะจง",
    "default": "ผิดนัด",
    "breach of contract": "ผิดสัญญา",
    "compensation for damages": "ค่าสินไหมทดแทน",
    "force majeure": "เหตุสุดวิสัย",
    "good faith": "สุจริต",
    "joint and several debtors": "ลูกหนี้ร่วม",
    "subrogation": "รับช่วงสิทธิ",
    "exercising debtor's claims": "การใช้สิทธิเรียกร้องของลูกหนี้",
    "cancellation of fraudulent acts": "การเพิกถอนการฉ้อฉล",
    "right of retention": "สิทธิยึดหน่วง",
    "preferential right": "บุริมสิทธิ",
    "earnest": "มัดจำ",
    "stipulated penalty": "เบี้ยปรับ",
    "rescission of contract": "การเลิกสัญญา",
    "wrongful act": "ละเมิด",
    "tort": "ละเมิด",
    "undue enrichment": "ลาภมิควรได้",
    "unjust enrichment": "ลาภมิควรได้",
    "management of affairs without mandate": "จัดการงานนอกสั่ง",
    "hire of property": "เช่าทรัพย์",
    "hire of work": "จ้างทำของ",
    "hire-purchase": "เช่าซื้อ",
    "loan for use": "ยืมใช้คงรูป",
    "loan for consumption": "ยืมใช้สิ้นเปลือง",
    "suretyship": "ค้ำประกัน",
    "mortgage": "จำนอง",
    "pledge": "จำนำ",
    "servitude": "ภาระจำยอม",
    "superficies": "สิทธิเหนือพื้นดิน",
    "usufruct": "สิทธิเก็บกิน",
    "habitation": "อาศัย",
    "charge on immovable property": "ภาระติดพันในอสังหาริมทรัพย์",
    "possessory right": "สิทธิครอบครอง",
    "adverse possession": "การครอบครองปรปักษ์",
    "public domain of state": "สาธารณสมบัติของแผ่นดิน",
    "statutory heir": "ทายาทโดยธรรม",
    "legatee": "ผู้รับพินัยกรรม",
    "administrator of an estate": "ผู้จัดการมรดก",
    "devolution of an estate": "การตกทอดแห่งทรัพย์มรดก",
}

DEFAULT_CIVIL_LAW_GLOSSARY_TH2EN: dict[str, str] = {}
for _en_term, _th_term in DEFAULT_CIVIL_LAW_GLOSSARY_EN2TH.items():
    DEFAULT_CIVIL_LAW_GLOSSARY_TH2EN.setdefault(_th_term, _en_term)

# 3. Thai Criminal / Penal Code glossary (หมวดกฎหมายอาญาของไทย)
DEFAULT_CRIMINAL_LAW_GLOSSARY_EN2TH: dict[str, str] = {
    "criminal offense": "ความผิดอาญา",
    "criminal liability": "ความรับผิดในทางอาญา",
    "criminal intent": "เจตนา",
    "transferred intent": "เจตนาโดยผลของกฎหมาย",
    "negligence": "ประมาท",
    "premeditation": "ไตร่ตรองไว้ก่อน",
    "attempt to commit an offense": "พยายามกระทำความผิด",
    "principal": "ตัวการ",
    "instigator": "ผู้ใช้ให้กระทำความผิด",
    "supporter": "ผู้สนับสนุน",
    "accomplice": "ผู้ร่วมกระทำความผิด",
    "lawful defense": "การป้องกันโดยชอบด้วยกฎหมาย",
    "offense committed out of necessity": "การกระทำความผิดด้วยความจำเป็น",
    "provocation": "บันดาลโทสะ",
    "extenuating circumstances": "เหตุบรรเทาโทษ",
    "recidivism": "การกระทำความผิดซ้ำ",
    "concurrence of offenses": "การกระทำความผิดหลายบทหรือหลายกระทง",
    "compoundable offense": "ความผิดอันยอมความได้",
    "non-compoundable offense": "ความผิดอาญาแผ่นดิน",
    "petty offense": "ความผิดลหุโทษ",
    "capital punishment": "ประหารชีวิต",
    "imprisonment": "จำคุก",
    "detention": "กักขัง",
    "fine": "ปรับ",
    "forfeiture of property": "ริบทรัพย์สิน",
    "confinement": "กักกัน",
    "measure of safety": "วิธีการเพื่อความปลอดภัย",
    "suspension of punishment": "การรอการลงโทษ",
    "suspension of the determination of punishment": "การรอการกำหนดโทษ",
    "malfeasance in office": "ความผิดต่อตำแหน่งหน้าที่ราชการ",
    "perjury": "เบิกความเท็จ",
    "false accusation": "ฟ้องเท็จ",
    "forgery of documents": "ปลอมเอกสาร",
    " bodily harm": "ทำร้ายร่างกาย",
    "grievous bodily harm": "อันตรายสาหัส",
    "homicide": "ความผิดต่อชีวิต",
    "manslaughter": "ฆ่าผู้อื่นโดยไม่เจตนา",
    "defamation": "หมิ่นประมาท",
    "theft": "ลักทรัพย์",
    "snatching": "วิ่งราวทรัพย์",
    "extortion": "รีดเอาทรัพย์",
    "blackmail": "กรรโชก",
    "robbery": "ชิงทรัพย์",
    "gang-robbery": "ปล้นทรัพย์",
    "fraud": "ฉ้อโกง",
    "cheating and fraud": "ฉ้อโกง",
    "Defrauding creditors": "โกงเจ้าหนี้",
    "embezzlement": "ยักยอก",
    "misappropriation": "ยักยอกทรัพย์",
    "receiving stolen property": "รับของโจร",
    "mischief": "ทำให้เสียทรัพย์",
    "trespass": "บุกรุก",
}

# Normalize any accidental leading/casing keys in criminal law glossary
DEFAULT_CRIMINAL_LAW_GLOSSARY_EN2TH = {
    k.strip().lower(): v for k, v in DEFAULT_CRIMINAL_LAW_GLOSSARY_EN2TH.items()
}

DEFAULT_CRIMINAL_LAW_GLOSSARY_TH2EN: dict[str, str] = {}
for _en_term, _th_term in DEFAULT_CRIMINAL_LAW_GLOSSARY_EN2TH.items():
    DEFAULT_CRIMINAL_LAW_GLOSSARY_TH2EN.setdefault(_th_term, _en_term)

# 4. Thai Civil Procedure Code glossary (หมวดกฎหมายวิธีพิจารณาความแพ่งของไทย)
DEFAULT_CIVIL_PROCEDURE_GLOSSARY_EN2TH: dict[str, str] = {
    "plaintiff": "โจทก์",
    "defendant": "จำเลย",
    "plaint": "คำฟ้อง",
    "answer": "คำให้การ",
    "counterclaim": "ฟ้องแย้ง",
    "petition": "คำร้อง",
    "motion": "คำแถลง",
    "interpleader": "ร้องสอด",
    "joinder of parties": "การร่วมเป็นโจทก์หรือจำเลย",
    "court of first instance": "ศาลชั้นต้น",
    "court of appeal": "ศาลอุทธรณ์",
    "supreme court": "ศาลฎีกา",
    "jurisdiction": "เขตอำนาจศาล",
    "settlement of issues": "การชี้สองสถาน",
    "burden of proof": "ภาระการพิสูจน์",
    "admissibility of evidence": "การรับฟังพยานหลักฐาน",
    "documentary evidence": "พยานเอกสาร",
    "material evidence": "พยานวัตถุ",
    "expert witness": "พยานผู้เชี่ยวชาญ",
    "subpoena": "หมายเรียก",
    "summons": "หมายเรียก",
    "default of appearance": "การขาดนัดพิจารณา",
    "default of filing an answer": "การขาดนัดยื่นคำให้การ",
    "compromise": "การประนีประนอมยอมความ",
    "judgment by consent": "คำพิพากษาตามยอม",
    "interlocutory order": "คำสั่งระหว่างพิจารณา",
    "injunction": "คำสั่งคุ้มครองชั่วคราว",
    "temporary protection before judgment": "การคุ้มครองชั่วคราวก่อนพิพากษา",
    "stay of execution": "การทุเลาการบังคับ",
    "writ of execution": "หมายบังคับคดี",
    "executing officer": "เจ้าพนักงานบังคับคดี",
    "judgment debtor": "ลูกหนี้ตามคำพิพากษา",
    "judgment creditor": "เจ้าหนี้ตามคำพิพากษา",
    "seizure of property": "การยึดทรัพย์",
    "attachment of claims": "การอายัดสิทธิเรียกร้อง",
    "public auction": "การขายทอดตลาด",
    "res judicata": "ดำเนินกระบวนพิจารณาซ้ำ",
    "parallel proceedings": "ฟ้องซ้อน",
    "arbitration award": "คำชี้ขาดของอนุญาโตตุลาการ",
}

DEFAULT_CIVIL_PROCEDURE_GLOSSARY_TH2EN: dict[str, str] = {}
for _en_term, _th_term in DEFAULT_CIVIL_PROCEDURE_GLOSSARY_EN2TH.items():
    DEFAULT_CIVIL_PROCEDURE_GLOSSARY_TH2EN.setdefault(_th_term, _en_term)

# 5. Thai Criminal Procedure Code glossary (หมวดกฎหมายวิธีพิจารณาความอาญาของไทย)
DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_EN2TH: dict[str, str] = {
    "injured person": "ผู้เสียหาย",
    "alleged offender": "ผู้ต้องหา",
    "accused": "จำเลย",
    "public prosecutor": "พนักงานอัยการ",
    "inquiry official": "พนักงานสอบสวน",
    "administrative or police official": "พนักงานฝ่ายปกครองหรือตำรวจ",
    "regular complaint": "คำร้องทุกข์",
    "denunciation": "คำกล่าวโทษ",
    "inquiry": "การสอบสวน",
    "inquiry file": "สำนวนการสอบสวน",
    "preliminary examination": "การไต่สวนมูลฟ้อง",
    "prosecution order": "คำสั่งฟ้อง",
    "non-prosecution order": "คำสั่งไม่ฟ้อง",
    "final non-prosecution order": "คำสั่งเด็ดขาดไม่ฟ้องคดี",
    "warrant of arrest": "หมายจับ",
    "search warrant": "หมายค้น",
    "warrant of detention": "หมายขัง",
    "warrant of imprisonment": "หมายจำคุก",
    "warrant of release": "หมายปล่อย",
    "provisional release": "การปล่อยชั่วคราว",
    "bail": "หลักประกันในการปล่อยชั่วคราว",
    "flagrant offense": "ความผิดซึ่งหน้า",
    "hearsay evidence": "พยานบอกเล่า",
    "circumstantial evidence": "พยานแวดล้อมกรณี",
    "eyewitness": "ประจักษ์พยาน",
    "benefit of the doubt": "ยกประโยชน์แห่งความสงสัยให้จำเลย",
    "beyond a reasonable doubt": "ปราศจากข้อสงสัยตามสมควร",
    "post-mortem inquest": "การชันสูตรพลิกศพ",
    "dismissal of the case": "พิพากษายกฟ้อง",
    "acquittal": "การปล่อยตัวจำเลยพ้นข้อหา",
    "conviction": "คำพิพากษาลงโทษ",
    "withdrawal of prosecution": "การถอนฟ้อง",
}

DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_TH2EN: dict[str, str] = {}
for _en_term, _th_term in DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_EN2TH.items():
    DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_TH2EN.setdefault(_th_term, _en_term)

# Combined Thai Procedural Law Glossary (Civil Procedure + Criminal Procedure)
DEFAULT_PROCEDURAL_LAW_GLOSSARY_EN2TH: dict[str, str] = {
    **DEFAULT_CIVIL_PROCEDURE_GLOSSARY_EN2TH,
    **DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_EN2TH,
}
DEFAULT_PROCEDURAL_LAW_GLOSSARY_TH2EN: dict[str, str] = {
    **DEFAULT_CIVIL_PROCEDURE_GLOSSARY_TH2EN,
    **DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_TH2EN,
}

# Combined Thai Legal Glossary (Civil + Criminal + Civil Procedure + Criminal Procedure)
DEFAULT_LEGAL_GLOSSARY_EN2TH: dict[str, str] = {
    **DEFAULT_CIVIL_LAW_GLOSSARY_EN2TH,
    **DEFAULT_CRIMINAL_LAW_GLOSSARY_EN2TH,
    **DEFAULT_PROCEDURAL_LAW_GLOSSARY_EN2TH,
}
DEFAULT_LEGAL_GLOSSARY_TH2EN: dict[str, str] = {
    **DEFAULT_CIVIL_LAW_GLOSSARY_TH2EN,
    **DEFAULT_CRIMINAL_LAW_GLOSSARY_TH2EN,
    **DEFAULT_PROCEDURAL_LAW_GLOSSARY_TH2EN,
}

# 6. FIDIC Construction Contract glossary (หมวดสัญญาจ้างก่อสร้างสากล FIDIC)
DEFAULT_FIDIC_CONTRACT_GLOSSARY_EN2TH: dict[str, str] = {
    "employer": "ผู้ว่าจ้าง",
    "contractor": "ผู้รับจ้าง",
    "the engineer": "วิศวกรผู้ควบคุมงาน",
    "subcontractor": "ผู้รับจ้างช่วง",
    "nominated subcontractor": "ผู้รับจ้างช่วงที่ผู้ว่าจ้างระบุชื่อ",
    "dispute avoidance/adjudication board": "คณะกรรมการป้องกันและวินิจฉัยข้อพิพาท",
    "commencement date": "วันเริ่มงาน",
    "time for completion": "ระยะเวลาทำงานให้แล้วเสร็จ",
    "extension of time": "การขยายระยะเวลาก่อสร้าง",
    "taking-over certificate": "หนังสือรับรองการรับมอบงาน",
    "defects notification period": "ระยะเวลารับประกันความชำรุดบกพร่อง",
    "performance certificate": "หนังสือรับรองการปฏิบัติงานตามสัญญาแล้วเสร็จ",
    "performance security": "หลักประกันการปฏิบัติตามสัญญา",
    "advance payment": "เงินล่วงหน้า",
    "interim payment certificate": "หนังสือรับรองการจ่ายเงินงวดระหว่างกาล",
    "final payment certificate": "หนังสือรับรองการจ่ายเงินงวดสุดท้าย",
    "statement at completion": "ใบแจ้งยอดงานเมื่อก่อสร้างแล้วเสร็จ",
    "retention money": "เงินประกันผลงานหักสะสม",
    "delay damages": "ค่าเสียหายจากความล่าช้า",
    "liquidated damages": "ค่าเสียหายที่กำหนดไว้ล่วงหน้า",
    "variation": "งานเปลี่ยนแปลงแก้ไข",
    "variation order": "คำสั่งเปลี่ยนแปลงงาน",
    "value engineering": "วิศวกรรมคุณค่า",
    "provisional sum": "เงินเผื่อจ่าย",
    "daywork": "งานคิดค่าแรงและวัสดุรายวัน",
    "bill of quantities": "บัญชีแสดงปริมาณงานและราคา",
    "schedule of rates": "ตารางอัตราค่างาน",
    "unforeseeable physical conditions": "สภาพทางกายภาพที่ไม่อาจคาดหมายได้",
    "exceptional event": "เหตุการณ์พิเศษสุดวิสัย",
    "suspension of work": "การสั่งพักงานชั่วคราว",
    "notice of claim": "หนังสือแจ้งการใช้สิทธิเรียกร้อง",
    "engineer's determination": "คำวินิจฉัยของวิศวกร",
    "notice of dissatisfaction": "หนังสือแจ้งความไม่เห็นพ้อง",
    "amicable settlement": "การระงับข้อพิพาทฉันมิตร",
    "latent defect": "ความชำรุดบกพร่องที่ซ่อนเร้น",
    "taking over of the works": "การรับมอบงานก่อสร้าง",
    "remedying defects": "การแก้ไขความชำรุดบกพร่อง",
}

DEFAULT_FIDIC_CONTRACT_GLOSSARY_TH2EN: dict[str, str] = {}
for _en_term, _th_term in DEFAULT_FIDIC_CONTRACT_GLOSSARY_EN2TH.items():
    DEFAULT_FIDIC_CONTRACT_GLOSSARY_TH2EN.setdefault(_th_term, _en_term)

# 7. Computer Science, Software Engineering & AI glossary (หมวดคอมพิวเตอร์และวิศวกรรมซอฟต์แวร์)
DEFAULT_COMPUTER_GLOSSARY_EN2TH: dict[str, str] = {
    # Data Structures, Algorithms & Systems (โครงสร้างข้อมูล ขั้นตอนวิธี และระบบปฏิบัติการ)
    "algorithm": "ขั้นตอนวิธี",
    "data structure": "โครงสร้างข้อมูล",
    "time complexity": "ความซับซ้อนด้านเวลา",
    "space complexity": "ความซับซ้อนด้านพื้นที่",
    "binary search tree": "ต้นไม้ค้นหาแบบทวิภาค",
    "hash table": "ตารางแฮช",
    "linked list": "รายการโยง",
    "directed acyclic graph": "กราฟระบุทิศทางแบบไม่มีวงจร",
    "recursion": "การเรียกซ้ำ",
    "dynamic programming": "กำหนดการพลวัต",
    "operating system": "ระบบปฏิบัติการ",
    "process scheduling": "การจัดตารางงานโพรเซส",
    "concurrency": "การทำงานพร้อมกัน",
    "parallelism": "การประมวลผลแบบขนาน",
    "deadlock": "ภาวะติดตาย",
    "race condition": "ภาวะแข่งขัน",
    "mutual exclusion": "การกีดกันร่วม",
    "virtual memory": "หน่วยความจำเสมือน",
    "garbage collection": "การเก็บขยะหน่วยความจำ",
    # Software Engineering, Databases & Security (วิศวกรรมซอฟต์แวร์ ฐานข้อมูล และความปลอดภัย)
    "software architecture": "สถาปัตยกรรมซอฟต์แวร์",
    "object-oriented programming": "การเขียนโปรแกรมเชิงวัตถุ",
    "functional programming": "การเขียนโปรแกรมเชิงฟังก์ชัน",
    "inheritance": "การสืบทอดคุณสมบัติ",
    "polymorphism": "พหุสัณฐาน",
    "encapsulation": "การห่อหุ้มข้อมูล",
    "abstraction": "ภาวะนามธรรม",
    "dependency injection": "การฉีดพึ่งพา",
    "continuous integration": "การผสานรวมอย่างต่อเนื่อง",
    "continuous deployment": "การนำขึ้นใช้งานอย่างต่อเนื่อง",
    "version control": "การควบคุมรุ่น",
    "unit test": "การทดสอบระดับหน่วย",
    "integration test": "การทดสอบการผสานรวม",
    "regression test": "การทดสอบการถดถอย",
    "relational database": "ฐานข้อมูลเชิงสัมพันธ์",
    "primary key": "คีย์หลัก",
    "foreign key": "คีย์นอก",
    "database transaction": "ธุรกรรมฐานข้อมูล",
    "atomicity": "ภาวะแบ่งแยกไม่ได้",
    "eventual consistency": "ความสอดคล้องในท้ายที่สุด",
    "load balancer": "ตัวกระจายภาระงาน",
    "reverse proxy": "พร็อกซีย้อนกลับ",
    "latency": "ความหน่วง",
    "throughput": "ปริมาณงานผ่าน",
    "bandwidth": "แบนด์วิดท์",
    "encryption": "การเข้ารหัสลับ",
    "decryption": "การถอดรหัสลับ",
    "authentication": "การพิสูจน์ตัวตน",
    "authorization": "การกำหนดสิทธิ์",
    "two-factor authentication": "การพิสูจน์ตัวตนแบบสองปัจจัย",
    "vulnerability": "ช่องโหว่ด้านความปลอดภัย",
    # Artificial Intelligence & Machine Learning (ปัญญาประดิษฐ์และการเรียนรู้ของเครื่อง)
    "artificial intelligence": "ปัญญาประดิษฐ์",
    "machine learning": "การเรียนรู้ของเครื่อง",
    "deep learning": "การเรียนรู้เชิงลึก",
    "neural network": "โครงข่ายประสาทเทียม",
    "large language model": "โมเดลภาษาขนาดใหญ่",
    "natural language processing": "การประมวลผลภาษาธรรมชาติ",
    "machine translation": "การแปลภาษาด้วยเครื่อง",
    "supervised learning": "การเรียนรู้แบบมีผู้สอน",
    "unsupervised learning": "การเรียนรู้แบบไม่มีผู้สอน",
    "reinforcement learning": "การเรียนรู้แบบเสริมกำลัง",
    "fine-tuning": "การปรับจูนละเอียด",
    "quantization": "การลดทอนความละเอียดของค่าถ่วงน้ำหนัก",
    "inference": "การอนุมานผล",
    "overfitting": "ภาวะเรียนรู้เกินพอดี",
    "underfitting": "ภาวะเรียนรู้ต่ำกว่าพอดี",
    "gradient descent": "การลดตามความชัน",
    "loss function": "ฟังก์ชันการสูญเสีย",
    "attention mechanism": "กลไกความสนใจ",
    "context window": "หน้าต่างบริบท",
    "retrieval-augmented generation": "การสร้างข้อความเสริมด้วยการค้นคืน",
    "vector embedding": "เวกเตอร์ฝังตัว",
    "hallucination": "การสร้างข้อมูลหลอน",
}

DEFAULT_COMPUTER_GLOSSARY_TH2EN: dict[str, str] = {}
for _en_term, _th_term in DEFAULT_COMPUTER_GLOSSARY_EN2TH.items():
    DEFAULT_COMPUTER_GLOSSARY_TH2EN.setdefault(_th_term, _en_term)

# 8. Architecture, Building Design & Building Code glossary (หมวดงานสถาปัตยกรรมและกฎหมายควบคุมอาคาร)
DEFAULT_ARCHITECTURE_GLOSSARY_EN2TH: dict[str, str] = {
    # Architectural Drawings & Design Phases (แบบสถาปัตยกรรมและขั้นตอนการออกแบบ)
    "schematic design": "การออกแบบขั้นร่าง",
    "design development": "การพัฒนาแบบ",
    "construction document": "แบบเพื่อการก่อสร้าง",
    "master plan": "ผังแม่บท",
    "site plan": "ผังบริเวณ",
    "floor plan": "แปลนพื้น",
    "reflected ceiling plan": "แปลนฝ้าเพดาน",
    "roof plan": "แปลนหลังคา",
    "elevation": "รูปด้าน",
    "section": "รูปตัด",
    "perspective rendering": "ภาพทัศนียภาพ",
    "isometric view": "ภาพไอโซเมตริก",
    "scale model": "หุ่นจำลองมาตราส่วน",
    "building information modeling": "แบบจำลองสารสนเทศอาคาร",
    # Building Envelope, Facade & Architectural Elements (เปลือกอาคาร รูปด้าน และองค์ประกอบอาคาร)
    "building envelope": "เปลือกอาคาร",
    "facade": "รูปด้านอาคาร",
    "curtain wall": "ผนังกระจกเคอร์เทนวอลล์",
    "cladding": "วัสดุกรุผิวอาคาร",
    "sunshading device": "แผงบังแดด",
    "louver": "บานเกล็ดระบายอากาศ",
    "canopy": "กันสาด",
    "eaves": "ชายคา",
    "soffit": "ท้องฝ้าชายคา",
    "gable roof": "หลังคาทรงจั่ว",
    "hip roof": "หลังคาทรงปั้นหยา",
    "flat roof": "หลังคาดาดฟ้า",
    "parapet wall": "ผนังกันตกดาดฟ้า",
    "skylight": "ช่องแสงหลังคา",
    "atrium": "โถงกลางอาคาร",
    "foyer": "โถงต้อนรับ",
    "mezzanine": "ชั้นลอย",
    "corridor": "ทางเดินร่วม",
    "balustrade": "ราวกันตก",
    "handrail": "ราวจับ",
    "stairwell": "ปล่องบันได",
    "stair tread": "ลูกนอนบันได",
    "stair riser": "ลูกตั้งบันได",
    "stair landing": "ชานพักบันได",
    "fenestration": "การจัดช่องเปิดอาคาร",
    "mullion": "กรอบตั้งหน้าต่าง",
    "transom": "กรอบขวางช่องแสงเหนือประตู",
    "window sill": "ธรณีหน้าต่าง",
    "threshold": "ธรณีประตู",
    # Interior Architecture & Finishing Materials (สถาปัตยกรรมภายในและวัสดุตกแต่ง)
    "suspended ceiling": "ฝ้าเพดานแขวน",
    "acoustic ceiling": "ฝ้าเพดานซับเสียง",
    "gypsum board partition": "ผนังเบาแผ่นยิปซัม",
    "masonry wall": "ผนังก่ออิฐฉาบปูน",
    "terrazzo": "หินขัด",
    "skirting board": "บัวเชิงผนัง",
    "cornice": "บัวฝ้าเพดาน",
    "architrave": "บัวซับวงกบ",
    "waterproofing membrane": "แผ่นกันซึม",
    "flashing": "แผ่นครอบกันน้ำซึม",
    "thermal insulation": "ฉนวนกันความร้อน",
    "vapor barrier": "แผ่นกั้นความชื้น",
    # Building Control Act, Zoning & Sustainability (กฎหมายควบคุมอาคาร ผังเมือง และอาคารเขียว)
    "floor area ratio": "อัตราส่วนพื้นที่อาคารรวมต่อพื้นที่ดิน",
    "open space ratio": "อัตราส่วนพื้นที่ว่างต่อพื้นที่อาคารรวม",
    "building coverage ratio": "อัตราส่วนพื้นที่คลุมดินของอาคาร",
    "setback distance": "ระยะร่นแนวอาคาร",
    "building line": "แนวร่นอาคาร",
    "ceiling height": "ความสูงภายในชั้น",
    "headroom clearance": "ระยะดิ่งพ้นศีรษะ",
    "means of egress": "ทางหนีไฟและทางออกฉุกเฉิน",
    "fire-enclosed staircase": "บันไดหนีไฟระบบปิด",
    "fire compartment": "พื้นที่ปิดล้อมกันไฟ",
    "fire resistance rating": "อัตราการทนไฟ",
    "universal design": "การออกแบบเพื่อทุกคน",
    "access ramp": "ทางลาดสำหรับผู้พิการ",
    "natural ventilation": "การระบายอากาศตามธรรมชาติ",
    "daylighting": "การใช้แสงธรรมชาติในอาคาร",
    "overall thermal transfer value": "ค่าการถ่ายเทความร้อนรวมของผนังอาคาร",
    "roof thermal transfer value": "ค่าการถ่ายเทความร้อนรวมของหลังคา",
}

DEFAULT_ARCHITECTURE_GLOSSARY_TH2EN: dict[str, str] = {}
for _en_term, _th_term in DEFAULT_ARCHITECTURE_GLOSSARY_EN2TH.items():
    DEFAULT_ARCHITECTURE_GLOSSARY_TH2EN.setdefault(_th_term, _en_term)


def get_domain_glossary(
    domains: GlossaryDomain | Sequence[GlossaryDomain] = "all",
    direction: Direction = "en2th",
) -> dict[str, str]:
    """Return a merged glossary for one or more specialized domains."""
    domain_list: list[GlossaryDomain] = [domains] if isinstance(domains, str) else list(domains)
    merged: dict[str, str] = {}
    for dom in domain_list:
        if dom in ("engineering", "civil_engineering", "all"):
            merged.update(
                DEFAULT_CIVIL_ENGINEERING_GLOSSARY_EN2TH
                if direction == "en2th"
                else DEFAULT_CIVIL_ENGINEERING_GLOSSARY_TH2EN
            )
        if dom in ("architecture", "all"):
            merged.update(
                DEFAULT_ARCHITECTURE_GLOSSARY_EN2TH
                if direction == "en2th"
                else DEFAULT_ARCHITECTURE_GLOSSARY_TH2EN
            )
        if dom in ("computer", "all"):
            merged.update(
                DEFAULT_COMPUTER_GLOSSARY_EN2TH
                if direction == "en2th"
                else DEFAULT_COMPUTER_GLOSSARY_TH2EN
            )
        if dom in ("civil_law", "legal", "all"):
            merged.update(
                DEFAULT_CIVIL_LAW_GLOSSARY_EN2TH
                if direction == "en2th"
                else DEFAULT_CIVIL_LAW_GLOSSARY_TH2EN
            )
        if dom in ("criminal_law", "legal", "all"):
            merged.update(
                DEFAULT_CRIMINAL_LAW_GLOSSARY_EN2TH
                if direction == "en2th"
                else DEFAULT_CRIMINAL_LAW_GLOSSARY_TH2EN
            )
        if dom in ("civil_procedure", "procedural_law", "legal", "all"):
            merged.update(
                DEFAULT_CIVIL_PROCEDURE_GLOSSARY_EN2TH
                if direction == "en2th"
                else DEFAULT_CIVIL_PROCEDURE_GLOSSARY_TH2EN
            )
        if dom in ("criminal_procedure", "procedural_law", "legal", "all"):
            merged.update(
                DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_EN2TH
                if direction == "en2th"
                else DEFAULT_CRIMINAL_PROCEDURE_GLOSSARY_TH2EN
            )
        if dom in ("fidic", "all"):
            merged.update(
                DEFAULT_FIDIC_CONTRACT_GLOSSARY_EN2TH
                if direction == "en2th"
                else DEFAULT_FIDIC_CONTRACT_GLOSSARY_TH2EN
            )
    return merged


_TONE_RULES: dict[tuple[ToneOfVoice, Direction], list[str]] = {
    ("formal", "en2th"): [
        "Use a formal, polite written Thai register suitable for official documents "
        "(ใช้ภาษาเขียนระดับทางการ สละสลวย และเป็นธรรมชาติ)",
    ],
    ("formal", "th2en"): [
        "Use a formal, professional written English register suitable for official documents",
    ],
    ("official_report", "en2th"): [
        "Use an official Thai engineering and government report register "
        "(ใช้สำนวนรายงานทางวิชาการและวิศวกรรมที่เป็นทางการ กระชับ ชัดเจน)",
        "Keep all numerals in Arabic digits",
    ],
    ("official_report", "th2en"): [
        "Use an official engineering and technical report register with objective phrasing",
        "Keep all numerals in Arabic digits",
    ],
    ("academic", "en2th"): [
        "Use an academic Thai register suitable for research papers and technical standards "
        "(ใช้ภาษาวิชาการตามมาตรฐานงานวิจัย)",
    ],
    ("academic", "th2en"): [
        "Use a rigorous academic English register suitable for peer-reviewed publications",
    ],
    ("legal", "en2th"): [
        "Use formal Thai statutory and judicial register aligned with the Thai Civil and "
        "Commercial Code, the Penal Code, and the Procedural Codes "
        "(ใช้สำนวนภาษากฎหมายไทยตามประมวลกฎหมายแพ่งและพาณิชย์ ประมวลกฎหมายอาญา "
        "และประมวลกฎหมายวิธีพิจารณาความอย่างเคร่งครัด)",
    ],
    ("legal", "th2en"): [
        "Use formal English statutory and legal register aligned with official English "
        "translations of the Thai Civil, Criminal, and Procedural Codes",
    ],
    ("fidic_contract", "en2th"): [
        "Use formal Thai construction contract and engineering consultancy register "
        "aligned with FIDIC Conditions of Contract and Thai public/private works contracts "
        "(ใช้สำนวนภาษาสัญญาจ้างก่อสร้างและวิศวกรรมที่ปรึกษาตามมาตรฐานสัญญา FIDIC)",
        "Keep all numerals and clause references in Arabic digits",
    ],
    ("fidic_contract", "th2en"): [
        "Use formal international construction contract register aligned with FIDIC "
        "Conditions of Contract (Red/Yellow/Silver Book terminology)",
        "Keep all numerals and clause references in Arabic digits",
    ],
    ("technical_software", "en2th"): [
        "Use formal Thai computer science and software engineering terminology "
        "aligned with the Royal Society of Thailand while keeping code identifiers intact "
        "(ใช้ศัพท์บัญญัติสาขาคอมพิวเตอร์และเทคโนโลยีสารสนเทศของราชบัณฑิตยสภา "
        "และคงชื่อตัวแปร/โค้ดไว้ตามเดิม)",
    ],
    ("technical_software", "th2en"): [
        "Use precise computer science, software architecture, and AI/ML English terminology",
    ],
    ("architectural_spec", "en2th"): [
        "Use formal Thai architectural design, specification, and building control register "
        "aligned with the Association of Siamese Architects (ASA) and the Building Control Act "
        "(ใช้สำนวนรายการประกอบแบบสถาปัตยกรรมและกฎหมายควบคุมอาคารตามมาตรฐานสมาคมสถาปนิกสยาม)",
        "Keep all dimensions, scales, and numerals in Arabic digits",
    ],
    ("architectural_spec", "th2en"): [
        "Use formal international architectural specification and building code register",
        "Keep all dimensions, scales, and numerals in Arabic digits",
    ],
}

_ANTI_TRANSLITERATION_RULE: dict[Direction, str] = {
    "en2th": (
        "Do not phonetically transliterate English technical terms into Thai; "
        "use established Thai technical terminology (ห้ามแปลทับศัพท์คำศัพท์เฉพาะทาง)"
    ),
    "th2en": (
        "Do not romanize or transliterate Thai technical terms; "
        "translate them into standard English technical terminology"
    ),
}

_HTML_PRESERVATION_RULE = (
    'Preserve all HTML tags, attributes, and structure (e.g., <div class="...">, '
    '<a href="...">, <strong>, <code>, <br/>) exactly as written; '
    "translate only the human-readable text content"
)

_MARKDOWN_PRESERVATION_RULE = (
    "Preserve all Markdown syntax (headings #, bold **, links [text](url), "
    "inline code `...`, tables, and fenced code blocks ```...```) without "
    "modifying code identifiers, URLs, or markup delimiters"
)

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(?:\s+[^<>]*?)?>")
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_MARKDOWN_LINK_URL_RE = re.compile(r"\]\(([^)\s]+)\)")


@dataclass(frozen=True)
class MarkupValidationResult:
    """Result of checking whether HTML tags and Markdown elements survived translation."""

    passed: bool
    missing_html_tags: list[str]
    missing_code_blocks: list[str]
    missing_inline_codes: list[str]
    missing_link_urls: list[str]


@dataclass
class TranslatorConfig:
    """Inference settings aligned with the official ChindaMT-4B card.

    Decoding values match the paper's test-time setup:
    temperature 0.01, top-p 0.7, top-k 20, repetition penalty 1.05,
    max_new_tokens 1024, thinking mode off. Weights stay in bfloat16
    by default with no quantization (comfortable on a 16 GB GPU), or
    can optionally use float16 / 4-bit quantization (`load_in_4bit=True`).
    """

    # Model
    model_id: str = "iapp/ChindaMT-4B"
    torch_dtype: str = "bfloat16"  # maps to torch.bfloat16 or torch.float16
    device_map: str = "auto"
    load_in_4bit: bool = False

    # Decoding (official reported settings)
    temperature: float = 0.01
    top_p: float = 0.7
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_new_tokens: int = 1024
    do_sample: bool = True
    enable_thinking: bool = False

    # Prompt / translation
    direction: Direction = "en2th"
    # Default rule from the model-card example; override or pass [] for none.
    rules: list[str] = field(default_factory=lambda: ["Return only the translated text"])

    # Glossary & Anti-transliteration (คำศัพท์เฉพาะทาง / ห้ามแปลทับศัพท์)
    glossary: dict[str, str] = field(default_factory=dict)
    do_not_transliterate: bool = False
    match_glossary_in_text_only: bool = True

    # Tone of voice (โทนภาษาทางการ / รายงานวิศวกรรม / วิชาการ / กฎหมาย / FIDIC / คอมพิวเตอร์)
    tone: ToneOfVoice = "default"

    # HTML & Markdown tag handling (การจัดการแท็ก HTML / Markdown ระหว่างแปล)
    preserve_html_tags: bool = False
    preserve_markdown: bool = False

    # Long documents: model trained with a 1024-token cutoff.
    # Prefer paragraph-by-paragraph translation for long inputs.
    chunk_by_paragraph: bool = True
    # Soft character budget per chunk (keeps prompts well under the cutoff).
    max_chars_per_chunk: int = 1500

    # How many independent texts to send through generate() at once.
    batch_size: int = 4

    @classmethod
    def formal_engineering_preset(
        cls,
        direction: Direction = "en2th",
        *,
        extra_glossary: Mapping[str, str] | None = None,
        tone: ToneOfVoice = "official_report",
        preserve_html_tags: bool = True,
        preserve_markdown: bool = True,
        load_in_4bit: bool = False,
    ) -> TranslatorConfig:
        """Create a formal preset with Glossary, anti-transliteration, and markup safety."""
        base_glossary = get_domain_glossary("civil_engineering", direction=direction)
        if extra_glossary:
            base_glossary.update(extra_glossary)

        return cls(
            direction=direction,
            glossary=base_glossary,
            do_not_transliterate=True,
            tone=tone,
            preserve_html_tags=preserve_html_tags,
            preserve_markdown=preserve_markdown,
            load_in_4bit=load_in_4bit,
        )

    @classmethod
    def formal_legal_preset(
        cls,
        direction: Direction = "en2th",
        *,
        domain: GlossaryDomain = "legal",
        extra_glossary: Mapping[str, str] | None = None,
        tone: ToneOfVoice = "legal",
        preserve_html_tags: bool = True,
        preserve_markdown: bool = True,
        load_in_4bit: bool = False,
    ) -> TranslatorConfig:
        """Create a formal Thai Civil, Criminal & Procedural Law preset."""
        base_glossary = get_domain_glossary(domain, direction=direction)
        if extra_glossary:
            base_glossary.update(extra_glossary)

        return cls(
            direction=direction,
            glossary=base_glossary,
            do_not_transliterate=True,
            tone=tone,
            preserve_html_tags=preserve_html_tags,
            preserve_markdown=preserve_markdown,
            load_in_4bit=load_in_4bit,
        )

    @classmethod
    def formal_fidic_preset(
        cls,
        direction: Direction = "en2th",
        *,
        include_engineering_and_civil_law: bool = True,
        extra_glossary: Mapping[str, str] | None = None,
        tone: ToneOfVoice = "fidic_contract",
        preserve_html_tags: bool = True,
        preserve_markdown: bool = True,
        load_in_4bit: bool = False,
    ) -> TranslatorConfig:
        """Create a formal FIDIC Construction Contract preset."""
        domains: Sequence[GlossaryDomain] = (
            ("fidic", "civil_engineering", "civil_law")
            if include_engineering_and_civil_law
            else ("fidic",)
        )
        base_glossary = get_domain_glossary(domains, direction=direction)
        if extra_glossary:
            base_glossary.update(extra_glossary)

        return cls(
            direction=direction,
            glossary=base_glossary,
            do_not_transliterate=True,
            tone=tone,
            preserve_html_tags=preserve_html_tags,
            preserve_markdown=preserve_markdown,
            load_in_4bit=load_in_4bit,
        )

    @classmethod
    def formal_computer_preset(
        cls,
        direction: Direction = "en2th",
        *,
        extra_glossary: Mapping[str, str] | None = None,
        tone: ToneOfVoice = "technical_software",
        preserve_html_tags: bool = True,
        preserve_markdown: bool = True,
        load_in_4bit: bool = False,
    ) -> TranslatorConfig:
        """Create a formal Computer Science, Software Engineering & AI preset."""
        base_glossary = get_domain_glossary("computer", direction=direction)
        if extra_glossary:
            base_glossary.update(extra_glossary)

        return cls(
            direction=direction,
            glossary=base_glossary,
            do_not_transliterate=True,
            tone=tone,
            preserve_html_tags=preserve_html_tags,
            preserve_markdown=preserve_markdown,
            load_in_4bit=load_in_4bit,
        )

    @classmethod
    def formal_architecture_preset(
        cls,
        direction: Direction = "en2th",
        *,
        include_civil_engineering: bool = False,
        extra_glossary: Mapping[str, str] | None = None,
        tone: ToneOfVoice = "architectural_spec",
        preserve_html_tags: bool = True,
        preserve_markdown: bool = True,
        load_in_4bit: bool = False,
    ) -> TranslatorConfig:
        """Create a formal Architecture, Building Design & Building Code preset."""
        domains: Sequence[GlossaryDomain] = (
            ("architecture", "civil_engineering")
            if include_civil_engineering
            else ("architecture",)
        )
        base_glossary = get_domain_glossary(domains, direction=direction)
        if extra_glossary:
            base_glossary.update(extra_glossary)

        return cls(
            direction=direction,
            glossary=base_glossary,
            do_not_transliterate=True,
            tone=tone,
            preserve_html_tags=preserve_html_tags,
            preserve_markdown=preserve_markdown,
            load_in_4bit=load_in_4bit,
        )


class ChindaTranslator:
    """Thin wrapper around iapp/ChindaMT-4B.

    Responsibilities:
    1. Load the model in bfloat16/float16 (or optional 4-bit quantization).
    2. Build prompts that match the model card (optional Rules: block,
       Glossary terms, Tone of Voice, and HTML/Markdown preservation rules).
    3. Chunk long text by paragraph while keeping fenced code blocks intact.
    4. Batch several short texts in one generate() call.
    """

    def __init__(self, config: TranslatorConfig | None = None) -> None:
        self.config: TranslatorConfig = config or TranslatorConfig()
        self.tokenizer: Any = None
        self.model: Any = None
        self._torch_module: Any = None

    # ------------------------------------------------------------------ load
    def load(
        self,
        *,
        torch_module: Any = None,
        transformers_module: Any = None,
    ) -> ChindaTranslator:
        """Download / cache weights and move them onto the GPU."""
        cfg = self.config
        if torch_module is None:
            torch_module = importlib.import_module("torch")
        if transformers_module is None:
            transformers_module = importlib.import_module("transformers")

        self._torch_module = torch_module
        dtype = getattr(torch_module, cfg.torch_dtype)
        auto_tokenizer = transformers_module.AutoTokenizer
        auto_model = transformers_module.AutoModelForCausalLM

        quant_desc = "4-bit quantization" if cfg.load_in_4bit else "no quantization"
        print(f"Loading {cfg.model_id} ({cfg.torch_dtype}, {quant_desc})...")
        self.tokenizer = auto_tokenizer.from_pretrained(cfg.model_id)
        # Left-pad so batched causal decoding aligns on the right edge.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": cfg.device_map,
        }
        if cfg.load_in_4bit:
            bnb_config_cls = getattr(transformers_module, "BitsAndBytesConfig", None)
            if bnb_config_cls is not None:
                model_kwargs["quantization_config"] = bnb_config_cls(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            else:
                model_kwargs["load_in_4bit"] = True

        self.model = auto_model.from_pretrained(cfg.model_id, **model_kwargs)
        self.model.eval()
        print("Model ready.")
        return self

    # -------------------------------------------------------------- prompting
    def compile_rules(
        self,
        text: str = "",
        *,
        direction: Direction | None = None,
        rules: Sequence[str] | None = None,
        glossary: Mapping[str, str] | None = None,
        do_not_transliterate: bool | None = None,
        tone: ToneOfVoice | None = None,
        preserve_html_tags: bool | None = None,
        preserve_markdown: bool | None = None,
    ) -> list[str]:
        """Assemble the effective list of auxiliary translation rules."""
        cfg = self.config
        active_direction: Direction = direction or cfg.direction
        base_rules = list(cfg.rules if rules is None else rules)
        active_glossary = dict(cfg.glossary if glossary is None else glossary)
        active_no_translit = (
            cfg.do_not_transliterate if do_not_transliterate is None else do_not_transliterate
        )
        active_tone: ToneOfVoice = cfg.tone if tone is None else tone
        active_html = cfg.preserve_html_tags if preserve_html_tags is None else preserve_html_tags
        active_md = cfg.preserve_markdown if preserve_markdown is None else preserve_markdown

        compiled: list[str] = []

        def _add_rule(rule_text: str) -> None:
            if rule_text and rule_text not in compiled:
                compiled.append(rule_text)

        for r in base_rules:
            _add_rule(r)

        # 1. Tone of voice rules
        for tone_rule in _TONE_RULES.get((active_tone, active_direction), []):
            _add_rule(tone_rule)

        # 2. Anti-transliteration & Glossary rules
        if active_no_translit or active_glossary:
            _add_rule(_ANTI_TRANSLITERATION_RULE[active_direction])

        if active_glossary:
            text_lower = text.lower()
            matched_pairs: list[str] = []
            for src_term, tgt_term in active_glossary.items():
                if (
                    not cfg.match_glossary_in_text_only
                    or not text
                    or src_term.lower() in text_lower
                ):
                    matched_pairs.append(f'"{src_term}" = "{tgt_term}"')
            if matched_pairs:
                _add_rule(
                    "Strictly use this glossary (do not transliterate): " + ", ".join(matched_pairs)
                )

        # 3. HTML & Markdown tag preservation rules
        if active_html:
            _add_rule(_HTML_PRESERVATION_RULE)
        if active_md:
            _add_rule(_MARKDOWN_PRESERVATION_RULE)

        return compiled

    def build_prompt(
        self,
        text: str,
        direction: Direction | None = None,
        rules: Sequence[str] | None = None,
        *,
        glossary: Mapping[str, str] | None = None,
        do_not_transliterate: bool | None = None,
        tone: ToneOfVoice | None = None,
        preserve_html_tags: bool | None = None,
        preserve_markdown: bool | None = None,
    ) -> str:
        """Build the exact user prompt shown on the Hugging Face model card."""
        active_direction: Direction = direction or self.config.direction
        active_rules = self.compile_rules(
            text,
            direction=active_direction,
            rules=rules,
            glossary=glossary,
            do_not_transliterate=do_not_transliterate,
            tone=tone,
            preserve_html_tags=preserve_html_tags,
            preserve_markdown=preserve_markdown,
        )

        src_name, tgt_name, tag = _LANG[active_direction]
        lines = [f"Translate {src_name} to {tgt_name}."]

        if active_rules:
            lines.append("Rules:")
            lines.extend(f"- {rule}" for rule in active_rules)

        lines.append("")
        lines.append(f"{tag}: {text}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- chunking
    @staticmethod
    def _split_paragraphs_preserving_fences(text: str) -> list[str]:
        """Split text on `\\n\\n` outside of Markdown fenced code blocks (```...```)."""
        placeholder_map: dict[str, str] = {}

        def _replace_fence(match: re.Match[str]) -> str:
            block = match.group(0)
            if "\n\n" not in block:
                return block
            key = f"__CHINDA_FENCE_BLOCK_{len(placeholder_map)}__"
            placeholder_map[key] = block
            return key

        masked = _FENCED_CODE_RE.sub(_replace_fence, text)
        raw_paragraphs = masked.split("\n\n")
        restored: list[str] = []
        for para in raw_paragraphs:
            for key, block in placeholder_map.items():
                if key in para:
                    para = para.replace(key, block)
            restored.append(para)
        return restored

    def chunk_text(self, text: str) -> list[str]:
        """Split on blank lines, then pack paragraphs under max_chars_per_chunk.

        Empty paragraphs are kept as empty strings so joining later
        preserves the original document structure. Fenced code blocks
        containing blank lines are kept intact as atomic units.
        """
        if not self.config.chunk_by_paragraph:
            return [text]

        paragraphs = self._split_paragraphs_preserving_fences(text)
        limit = self.config.max_chars_per_chunk
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            # Blank separators stay as their own chunk so we can restore them.
            if not para.strip():
                if current:
                    chunks.append("\n\n".join(current))
                    current, current_len = [], 0
                chunks.append(para)
                continue

            # +2 accounts for the "\n\n" that will sit between paragraphs.
            extra = len(para) + (2 if current else 0)
            if current and current_len + extra > limit:
                chunks.append("\n\n".join(current))
                current, current_len = [para], len(para)
            else:
                current.append(para)
                current_len += extra

        if current:
            chunks.append("\n\n".join(current))
        return chunks or [text]

    # ------------------------------------------------------- markup validation
    @staticmethod
    def validate_markup_preservation(source: str, translated: str) -> MarkupValidationResult:
        """Verify that HTML tags, Markdown code spans/blocks, and link URLs were preserved."""
        src_html_tags = _HTML_TAG_RE.findall(source)
        missing_html = [tag for tag in src_html_tags if tag not in translated]

        src_fenced = _FENCED_CODE_RE.findall(source)
        missing_fenced = [block for block in src_fenced if block not in translated]

        src_without_fences = _FENCED_CODE_RE.sub("", source)
        tgt_without_fences = _FENCED_CODE_RE.sub("", translated)
        src_inline = _INLINE_CODE_RE.findall(src_without_fences)
        missing_inline = [
            f"`{code}`" for code in src_inline if f"`{code}`" not in tgt_without_fences
        ]

        src_urls = _MARKDOWN_LINK_URL_RE.findall(source)
        missing_urls = [url for url in src_urls if f"]({url})" not in translated]

        passed = not (missing_html or missing_fenced or missing_inline or missing_urls)
        return MarkupValidationResult(
            passed=passed,
            missing_html_tags=missing_html,
            missing_code_blocks=missing_fenced,
            missing_inline_codes=missing_inline,
            missing_link_urls=missing_urls,
        )

    # --------------------------------------------------------------- generate
    def _generate_batch(
        self,
        prompts: list[str],
        *,
        torch_module: Any = None,
    ) -> list[str]:
        """Run one padded generate() call for a list of ready-made prompts."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load() before translating.")

        active_torch = torch_module or self._torch_module
        if active_torch is None:
            active_torch = importlib.import_module("torch")

        cfg = self.config
        chat_texts: list[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            chat_texts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=cfg.enable_thinking,
                )
            )

        raw_inputs = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = {k: v.to(self.model.device) for k, v in raw_inputs.items()}
        # With left padding, every sequence's new tokens start after the full
        # padded prompt length (same for all rows in the batch).
        prompt_len = int(inputs["input_ids"].shape[1])

        with active_torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repetition_penalty,
                do_sample=cfg.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results: list[str] = []
        for seq in outputs:
            new_tokens = seq[prompt_len:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return results

    # -------------------------------------------------------------- translate
    def translate(
        self,
        text: str,
        direction: Direction | None = None,
        rules: Sequence[str] | None = None,
        *,
        glossary: Mapping[str, str] | None = None,
        do_not_transliterate: bool | None = None,
        tone: ToneOfVoice | None = None,
        preserve_html_tags: bool | None = None,
        preserve_markdown: bool | None = None,
    ) -> str:
        """Translate one string (auto-chunks long documents by paragraph)."""
        chunks = self.chunk_text(text)
        translated: list[str] = []

        # Skip empty chunks without calling the model.
        work_indices = [i for i, c in enumerate(chunks) if c.strip()]
        work_texts = [chunks[i] for i in work_indices]

        outputs = self.translate_batch(
            work_texts,
            direction=direction,
            rules=rules,
            glossary=glossary,
            do_not_transliterate=do_not_transliterate,
            tone=tone,
            preserve_html_tags=preserve_html_tags,
            preserve_markdown=preserve_markdown,
        )

        # Re-insert empty chunks so "\n\n".join restores spacing.
        result_map = dict(zip(work_indices, outputs, strict=False))
        for i, chunk in enumerate(chunks):
            translated.append(result_map.get(i, chunk))

        return "\n\n".join(translated)

    def translate_batch(
        self,
        texts: Sequence[str],
        direction: Direction | None = None,
        rules: Sequence[str] | None = None,
        *,
        glossary: Mapping[str, str] | None = None,
        do_not_transliterate: bool | None = None,
        tone: ToneOfVoice | None = None,
        preserve_html_tags: bool | None = None,
        preserve_markdown: bool | None = None,
    ) -> list[str]:
        """Translate many independent strings, micro-batched for the GPU."""
        if not texts:
            return []

        prompts = [
            self.build_prompt(
                t,
                direction=direction,
                rules=rules,
                glossary=glossary,
                do_not_transliterate=do_not_transliterate,
                tone=tone,
                preserve_html_tags=preserve_html_tags,
                preserve_markdown=preserve_markdown,
            )
            for t in texts
        ]
        batch_size = max(1, self.config.batch_size)
        results: list[str] = []

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            results.extend(self._generate_batch(batch))

        return results


def run_chinda_demo(translator: ChindaTranslator | None = None) -> None:
    """Execute the ChindaMT-4B Thai-English demo workflow."""
    if translator is None:
        config = TranslatorConfig.formal_engineering_preset(
            direction="en2th",
            tone="official_report",
            preserve_html_tags=True,
            preserve_markdown=True,
        )
        config.batch_size = 2
        translator = ChindaTranslator(config).load()

    # --- single sentence ---
    english = "The weather is nice today."
    thai = translator.translate(english)
    print("EN → TH")
    print(f"  src: {english}")
    print(f"  out: {thai}\n")

    # --- reverse direction ---
    back = translator.translate(thai, direction="th2en", rules=["Return only the translated text"])
    print("TH → EN")
    print(f"  src: {thai}")
    print(f"  out: {back}\n")

    # --- glossary + formal tone + HTML/Markdown preservation ---
    formatted_src = (
        "## Inspection Summary\n\n"
        "The **shear wall** and <code>prestressed concrete</code> passed "
        "[seismic load](https://example.com/spec) checks."
    )
    formatted_out = translator.translate(
        formatted_src,
        glossary=DEFAULT_ENGINEERING_GLOSSARY_EN2TH,
        do_not_transliterate=True,
        tone="official_report",
        preserve_html_tags=True,
        preserve_markdown=True,
    )
    print("Formal + Glossary + HTML/Markdown EN → TH")
    print(f"  src: {formatted_src}")
    print(f"  out: {formatted_out}\n")

    # --- batch of short strings ---
    batch = [
        "Please submit the report by Friday.",
        "The bridge inspection is scheduled for next week.",
    ]
    outs = translator.translate_batch(batch)
    print("Batch EN → TH")
    for src, out in zip(batch, outs, strict=False):
        print(f"  src: {src}")
        print(f"  out: {out}")

    # --- long text (paragraph chunking) ---
    long_text = (
        "ChindaMT-4B is a Thai-English translation model.\n\n"
        "It follows auxiliary rules such as terminology, register, "
        "length, and output format.\n\n"
        "For long documents, translate paragraph by paragraph."
    )
    long_out = translator.translate(long_text)
    print("\nLong document EN → TH")
    print(long_out)
