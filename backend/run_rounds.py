import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

dataset_path = 'backend/datasets/real_clinically_matched_multimodal.csv'
with open(dataset_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

N = len(rows)
modalities = ['tremor_risk', 'finger_risk', 'gait_risk', 'voice_risk', 'questionnaire_risk']
mod_thai = {
    'tremor_risk': 'การสั่น (Tremor)',
    'finger_risk': 'แตะนิ้ว (Finger Tap)',
    'gait_risk': 'การเดิน (Gait)',
    'voice_risk': 'เสียงพูด (Voice)',
    'questionnaire_risk': 'แบบประเมิน (Questionnaire)'
}

# Decision logic for 5 modalities (Round 0)
# Threshold 0.35, majority vote >= 3 of 5
def predict_5(r):
    votes = sum(1 for m in modalities if float(r[m]) >= 0.35)
    return 1 if votes >= 3 else 0

# Decision logic for 4 modalities (Leave-one-out)
# Threshold 0.35, 4 modalities: if >= 3 votes -> 1, if <= 1 votes -> 0, if == 2 votes -> tie-break with mean score >= 0.35
def predict_4(r, dropped_m):
    active = [m for m in modalities if m != dropped_m]
    scores = [float(r[m]) for m in active]
    votes = sum(1 for s in scores if s >= 0.35)
    if votes >= 3: return 1
    if votes <= 1: return 0
    return 1 if (sum(scores)/4.0) >= 0.35 else 0

y_true = [int(r['is_parkinson']) for r in rows]
y_round0 = [predict_5(r) for r in rows]

acc_round0 = sum(1 for i in range(N) if y_round0[i] == y_true[i]) / N * 100
tp0 = sum(1 for i in range(N) if y_round0[i] == 1 and y_true[i] == 1)
fn0 = sum(1 for i in range(N) if y_round0[i] == 0 and y_true[i] == 1)
tn0 = sum(1 for i in range(N) if y_round0[i] == 0 and y_true[i] == 0)
fp0 = sum(1 for i in range(N) if y_round0[i] == 1 and y_true[i] == 0)

print("="*75)
print(f"รอบที่ 0: ทดสอบตั้งต้นด้วย 5 มิติครบ (เกณฑ์ 3/5 Majority Vote)")
print(f"  - ความแม่นยำรวม         : {acc_round0:.2f}% ({tp0+tn0}/{N} คน)")
print(f"  - ตรวจเจอคนป่วย (TP)    : {tp0} จาก 31 คน")
print(f"  - คนป่วยหลุดตรวจ (FN)   : {fn0} คน (FNR = {fn0/31*100:.2f}%)")
print(f"  - คนปกติทายถูก (TN)    : {tn0} จาก 11 คน (FP={fp0})")
print("="*75)

print("\n" + "="*75)
print("เปรียบเทียบแบบที่ 1: เกณฑ์เสียงข้างมาก (Majority Vote)")
print("="*75)
for i, m in enumerate(modalities, 1):
    # Strict 3 out of 4 rule (75% agreement)
    y_round_strict = []
    for r in rows:
        active = [x for x in modalities if x != m]
        votes = sum(1 for x in active if float(r[x]) >= 0.35)
        y_round_strict.append(1 if votes >= 3 else 0)
    
    match_r0 = sum(1 for j in range(N) if y_round_strict[j] == y_round0[j])
    acc_s = sum(1 for j in range(N) if y_round_strict[j] == y_true[j]) / N * 100
    fn_s = sum(1 for j in range(N) if y_round_strict[j] == 0 and y_true[j] == 1)
    acc_drop_s = acc_round0 - acc_s
    
    print(f"รอบที่ {i}: ตัด '{mod_thai[m]}' ออก (เกณฑ์ต้องผ่าน 3 ใน 4 มิติ)")
    print(f"  • ผลตรงกับรอบแรก (5 มิติ) : {match_r0}/{N} คน ({match_r0/N*100:.2f}%)")
    print(f"  • ความแม่นยำรวมที่เหลือ   : {acc_s:.2f}% (ความแม่นยำลดลง -{acc_drop_s:.2f}%)")
    print(f"  • คนป่วยที่หลุดตรวจ (FN)  : {fn_s} คน (หลุดตรวจเพิ่มเป็น {fn_s} คน)")
    print("-" * 75)
