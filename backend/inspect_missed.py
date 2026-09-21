import csv
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

dataset_path = 'backend/datasets/real_clinically_matched_multimodal.csv'
with open(dataset_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

modalities = ['tremor_risk', 'finger_risk', 'gait_risk', 'voice_risk', 'questionnaire_risk']
mod_short = {'tremor_risk': 'สั่น', 'finger_risk': 'แตะนิ้ว', 'gait_risk': 'เดิน', 'voice_risk': 'เสียง', 'questionnaire_risk': 'แบบประเมิน'}

for m in modalities:
    rem = [x for x in modalities if x != m]
    missed = []
    for r in rows:
        if r['is_parkinson'] == '1':
            v0 = sum(1 for x in modalities if float(r[x]) >= 0.35) >= 3
            v_rem = sum(1 for x in rem if float(r[x]) >= 0.35) >= 3
            if v0 and not v_rem:
                pos_in_rem = [mod_short[x] for x in rem if float(r[x]) >= 0.35]
                neg_in_rem = [mod_short[x] for x in rem if float(r[x]) < 0.35]
                missed.append(f"{r['subject_id']} (เหลือผ่านแค่ {len(pos_in_rem)} มิติคือ {pos_in_rem})")
    print(f"เมื่อตัด '{mod_short[m]}' ออก -> คนป่วยหลุดตรวจ {len(missed)} คน:")
    for x in missed:
        print(f"   - {x}")
    print()
