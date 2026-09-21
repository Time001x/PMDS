# 📋 เอกสารสรุปสูตรคำนวณและเกณฑ์การประเมิน MDS-UPDRS
**โครงการ:** ระบบประเมินและคัดกรองความเสี่ยงโรคพาร์กินสันด้วยปัญญาประดิษฐ์แบบหลายมิติ (PMDS)  
**มาตรฐานอ้างอิง:** Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS)

---

## 1. เกณฑ์ระดับคะแนนมาตรฐาน MDS-UPDRS (0 – 4 Rating Scale)

แพทย์ผู้เชี่ยวชาญด้านระบบประสาท (Neurologists) ใช้เกณฑ์ประเมินความรุนแรง 5 ระดับ ดังนี้:

| คะแนน (Score) | สเกลทางคลินิก | ข้อความแสดงผลบนแอป | สีประจำระดับ | นิยามทางคลินิก (Clinical Definition) |
| :---: | :---: | :---: | :---: | :--- |
| **0** | Normal | **ไม่มีอาการ** | 🟢 `#05C134` | ปกติ สมบูรณ์ ไม่มีอาการแสดงของโรคพาร์กินสัน |
| **1** | Slight | **เล็กน้อย** | 🟢 `#10B981` | มีความผิดปกติเพียงเล็กน้อย ไม่กระทบต่อการดำเนินชีวิต |
| **2** | Mild | **เสี่ยงปานกลาง** | 🟡 `#F59E0B` | มีอาการชัดเจนขึ้น เริ่มส่งผลกระทบต่อบางกิจกรรม |
| **3** | Moderate | **เสี่ยงมาก** | 🟠 `#F97316` | มีอาการชัดเจน รบกวนการทำกิจกรรมประจำวันอย่างมาก |
| **4** | Severe | **อาการรุนแรง** | 🔴 `#C10508` | มีอาการรุนแรงมาก ไม่สามารถพึ่งพาตนเองได้ |

---

## 2. สูตรคำนวณแยกตามมิติการตรวจ (Modality Mathematical Formulas)

### 🎤 2.1 มิติทดสอบเสียงพูด (Speech Test — MDS-UPDRS Item 3.1)
วิเคราะห์จากสัญญาณเสียงพูดถดถอย (Acoustic Voice Degradation) ด้วย 3 ดัชนีทางอคูสติก:

1. **Pitch Jitter (local) % (ความแปรผันความถี่พื้นฐาน):**
   $$Jitter = \frac{\frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N} \sum_{i=1}^{N} T_i} \times 100\%$$
   *(เมื่อ $T_i$ คือระยะเวลาคาบความถี่ Fundamental Frequency $F_0$)*

2. **Shimmer (local) % (ความแปรผันแอมพลิจูดความดัง):**
   $$Shimmer = \frac{\frac{1}{N-1} \sum_{i=1}^{N-1} |A_i - A_{i+1}|}{\frac{1}{N} \sum_{i=1}^{N} A_i} \times 100\%$$
   *(เมื่อ $A_i$ คือแอมพลิจูดสูงสุดของแต่ละคาบเสียง)*

3. **Harmonics-to-Noise Ratio (HNR):**
   $$HNR = 10 \times \log_{10} \left( \frac{E_{\text{harmonics}}}{E_{\text{noise}}} \right) \text{ (dB)}$$

**เกณฑ์การตัดเกรด Item 3.1:**
- **Score 0 (ไม่มีอาการ):** $Jitter < 1.04\%$, $Shimmer < 3.81\%$, $HNR > 20 \text{ dB}$
- **Score 1 (เล็กน้อย):** $Jitter \in [1.04\%, 1.50\%]$, $Shimmer \in [3.81\%, 5.00\%]$, $HNR \in [16, 20] \text{ dB}$
- **Score 2 (เสี่ยงปานกลาง):** $Jitter \in [1.50\%, 2.50\%]$, $Shimmer \in [5.00\%, 7.50\%]$, $HNR \in [12, 16) \text{ dB}$
- **Score 3 (เสี่ยงมาก):** $Jitter \in [2.50\%, 4.00\%]$, $Shimmer \in [7.50\%, 10.00\%]$, $HNR \in [8, 12) \text{ dB}$
- **Score 4 (อาการรุนแรง):** $Jitter > 4.00\%$, $Shimmer > 10.00\%$, $HNR < 8 \text{ dB}$

---

### 📳 2.2 มิติทดสอบอาการสั่นขณะพัก (Rest Tremor — MDS-UPDRS Item 3.17)
วิเคราะห์จากเซ็นเซอร์ตรวจจับความเร่ง (Accelerometer) บนสมาร์ทโฟนขณะพักมือ:

1. **การคำนวณ Dynamic Acceleration Magnitude (ขจัดแรงโน้มถ่วง):**
   $$a_{\text{mag}}(t) = \sqrt{(a_x(t) - \bar{a}_x)^2 + (a_y(t) - \bar{a}_y)^2 + (a_z(t) - \bar{a}_z)^2}$$

2. **Root Mean Square (RMS Dynamic Amplitude):**
   $$a_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} a_{\text{mag}}(t_i)^2} \text{ (m/s}^2\text{)}$$

3. **FFT Pathological Tremor Band Energy (ความถี่สั่นพาร์กินสัน 4.0 – 6.5 Hz):**
   $$E_{\text{tremor}} = \int_{4.0}^{6.5} P(f) \, df$$

**เกณฑ์การตัดเกรด Item 3.17:**
- **Score 0 (ไม่มีอาการ):** $a_{\text{rms}} < 0.35 \text{ m/s}^2$ (ระยะสั่น $< 0.5 \text{ cm}$)
- **Score 1 (เล็กน้อย):** $a_{\text{rms}} \in [0.35, 0.55) \text{ m/s}^2$ (ระยะสั่น $0.5 - 1.0 \text{ cm}$)
- **Score 2 (เสี่ยงปานกลาง):** $a_{\text{rms}} \in [0.55, 0.85) \text{ m/s}^2$ (ระยะสั่น $1.0 - 3.0 \text{ cm}$)
- **Score 3 (เสี่ยงมาก):** $a_{\text{rms}} \in [0.85, 1.30) \text{ m/s}^2$ (ระยะสั่น $3.0 - 10.0 \text{ cm}$)
- **Score 4 (อาการรุนแรง):** $a_{\text{rms}} \ge 1.30 \text{ m/s}^2$ (ระยะสั่น $> 10.0 \text{ cm}$)

---

### 👆 2.3 มิติทดสอบการแตะนิ้ว (Finger Tapping — MDS-UPDRS Item 3.4)
วิเคราะห์ภาวะเคลื่อนไหวช้า (Bradykinesia) จากการรบกวนจังหวะการแตะหน้าจอ:

1. **ความเร็วในการแตะ (Tapping Frequency Rate):**
   $$f_{\text{tap}} = \frac{N_{\text{taps}}}{T_{\text{duration}}} \text{ (Hz)}$$

2. **ความไม่สม่ำเสมอของจังหวะ (Inter-Tap Interval Coefficient of Variation):**
   $$\text{CV}_{\text{ITI}} = \left( \frac{\text{SD}_{\text{ITI}}}{\text{Mean}_{\text{ITI}}} \right) \times 100\%$$

**เกณฑ์การตัดเกรด Item 3.4:**
- **Score 0 (ไม่มีอาการ):** $f_{\text{tap}} \ge 3.0 \text{ Hz}$, $\text{CV}_{\text{ITI}} < 15\%$
- **Score 1 (เล็กน้อย):** $f_{\text{tap}} \in [2.4, 3.0) \text{ Hz}$, $\text{CV}_{\text{ITI}} \in [15\%, 25\%]$
- **Score 2 (เสี่ยงปานกลาง):** $f_{\text{tap}} \in [1.8, 2.4) \text{ Hz}$, $\text{CV}_{\text{ITI}} \in [25\%, 35\%]$
- **Score 3 (เสี่ยงมาก):** $f_{\text{tap}} \in [1.2, 1.8) \text{ Hz}$, $\text{CV}_{\text{ITI}} \in [35\%, 50\%]$
- **Score 4 (อาการรุนแรง):** $f_{\text{tap}} < 1.2 \text{ Hz}$, $\text{CV}_{\text{ITI}} > 50\%$

---

### 🚶 2.4 มิติทดสอบการเดิน (Gait Assessment — MDS-UPDRS Item 3.10)
วิเคราะห์จากวงรอบการเดิน (Gait Cycle) ด้วยเซ็นเซอร์ความเร่งขณะเดิน 10 วินาที:

1. **ความเร็วในการเดิน (Walking Velocity):**
   $$v_{\text{walk}} = \frac{d_{\text{total}}}{T_{\text{walk}}} \text{ (m/s)}$$

2. **ความแปรผันของระยะเวลาแต่ละก้าว (Stride Time CV):**
   $$\text{CV}_{\text{stride}} = \left( \frac{\text{SD}_{\text{stride}}}{\text{Mean}_{\text{stride}}} \right) \times 100\%$$

3. **ภาวะเดินชะงัก (Freezing of Gait Pauses):**
   $$\text{FOG}_{\text{count}} = \sum \text{Gaps}_{> 2.0\text{s}}$$

**เกณฑ์การตัดเกรด Item 3.10:**
- **Score 0 (ไม่มีอาการ):** $v_{\text{walk}} \ge 1.1 \text{ m/s}$, $\text{CV}_{\text{stride}} < 12\%$, $\text{FOG} = 0$
- **Score 1 (เล็กน้อย):** $v_{\text{walk}} \in [0.95, 1.1) \text{ m/s}$, $\text{CV}_{\text{stride}} \in [12\%, 20\%]$, $\text{FOG} = 0$
- **Score 2 (เสี่ยงปานกลาง):** $v_{\text{walk}} \in [0.75, 0.95) \text{ m/s}$, $\text{CV}_{\text{stride}} \in [20\%, 30\%]$, $\text{FOG} \le 1$
- **Score 3 (เสี่ยงมาก):** $v_{\text{walk}} \in [0.50, 0.75) \text{ m/s}$, $\text{CV}_{\text{stride}} \in [30\%, 45\%]$, $\text{FOG} \in [1, 2]$
- **Score 4 (อาการรุนแรง):** $v_{\text{walk}} < 0.50 \text{ m/s}$, $\text{CV}_{\text{stride}} > 45\%$, $\text{FOG} \ge 3$

---

### 📝 2.5 มิติทดสอบแบบสอบถาม (MDS-UPDRS Part II Questionnaire)
ประเมินผลกระทบต่อกิจกรรมในชีวิตประจำวัน (Motor Experiences of Daily Living) จำนวน 20 ข้อ (ข้อละ 0 – 3 คะแนน):

$$\text{UPDRS}_{\text{total}} = \sum_{i=1}^{20} Q_i \quad (\text{คะแนนเต็ม } 60 \text{ คะแนน})$$

**เกณฑ์การตัดเกรด Part II:**
- **Score 0 (ไม่มีอาการ):** $\text{UPDRS}_{\text{total}} < 6$
- **Score 1 (เล็กน้อย):** $\text{UPDRS}_{\text{total}} \in [6, 15]$
- **Score 2 (เสี่ยงปานกลาง):** $\text{UPDRS}_{\text{total}} \in [16, 28]$
- **Score 3 (เสี่ยงมาก):** $\text{UPDRS}_{\text{total}} \in [29, 42]$
- **Score 4 (อาการรุนแรง):** $\text{UPDRS}_{\text{total}} > 42$

---

## 3. เกณฑ์สรุปผลรวม (Overall Ensemble & Majority Vote Rule)

ระบบใช้ **กฎเกณฑ์เสียงข้างมาก 3 ใน 5 มิติ (3/5 Majority Decision Rule)** เพื่อให้การสรุปผลรวมมีความเที่ยงตรงและป้องกันผลผิดพลาด:

$$\text{Decision} = \begin{cases} 
\text{ไม่มีอาการ (Level 0)}, & \text{ถ้า } N(\text{Score} \le 1) \ge 3 \\
\text{เสี่ยงมาก (Level 3)}, & \text{ถ้า } N(\text{Score} \ge 3) \ge 3 \\
\text{ตามค่าถ่วงน้ำหนัก Ensemble}, & \text{กรณีอื่นๆ}
\end{cases}$$

---
*เอกสารฉบับนี้รวบรวมขึ้นเพื่อใช้อ้างอิงทางวิชาการและการนำเสนอต่อคณะกรรมการ/อาจารย์ที่ปรึกษาโครงการ*
