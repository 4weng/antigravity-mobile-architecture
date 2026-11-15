# สรุปสถาปัตยกรรม Google Antigravity บนมือถือ Android (2 โหมดหลัก)

เอกสารสรุปเปรียบเทียบการทำงานตามคู่มือฉบับรวมสมบูรณ์ ทั้งมุมมอง **Software Engineering** และ **Civil / Structural Engineering**:

## ตารางเปรียบเทียบโหมดการทำงานทั้ง 2 โหมด

| หัวข้อเปรียบเทียบ | Remote Conversations (Pure Cloud / Remote Agents) | Remote Control (Windows 11 PC Workstation) |
| --- | --- | --- |
| **ตำแหน่งประมวลผล** | Google Isolated Cloud Servers 100% | เครื่องคอมพิวเตอร์ Windows 11 ที่บ้าน |
| **สถานะเครื่องพีซีที่บ้าน** | ปิดเครื่องได้สนิท (Home PC Off) | ต้องเปิดเครื่องและรัน Desktop App หรือ `agy --remote-agent` |
| **การเชื่อมต่อไฟล์และทรัพยากร** | ผูกผ่าน OAuth Connectors ใน Resources (**GitHub** + **Google Drive**) | เข้าถึงไฟล์ในฮาร์ดดิสก์เครื่องบ้านโดยตรง (เช่น `D:\...`) |
| **ช่องทางสื่อสารจากมือถือ** | Android Chrome PWA (`antigravity.google.com`) + Service Worker Push | Android Chrome PWA + **Web Relay** น้ำหนักเบา (ไม่ใช่ RDP) |
| **ฮาร์ดแวร์ที่ใช้งาน** | Cloud Compute Engine (แบ่งบทบาท **Argon**, **Hydrogen**, **Claude Opus 5.5**) | Local Workstation Hardware (เช่น **RAM 128 GB**, **NVIDIA 16 GB VRAM**) |
| **ตัวอย่างงานที่เหมาะสม** | วางแผนโครงสร้างโค้ด, แตก Branch, Code Review, งานที่ไม่ต้องใช้ไฟล์ในเครื่องบ้าน | รันโมเดล **ChindaMT-4B**, ประมวลผลภาพโดรนตรวจรอยร้าวอาคาร |
| **การสลับโหมดบนมือถือ** | Top Bar **Instance Switcher** -> `Remote Conversations` | Top Bar **Instance Switcher** -> `Windows PC Workstation` |

## บทบาท Model Tiers (ส่วนที่ 3)

1. **Argon** — *วางแผนโครงสร้างและแตกงานย่อย (Planning)* | เทียบเท่า **Project Director / Lead Structural Architect**
2. **Hydrogen** — *เขียนโค้ดแยกตาม Branch (Branch Coding)* | เทียบเท่า **Production Engineer**
3. **Claude Opus 5.5** — *ตรวจสอบความถูกต้องของโค้ด (Code Review)* | เทียบเท่า **Independent Peer Reviewer / Checker**
