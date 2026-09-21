import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

dataset_path = 'backend/datasets/real_clinically_matched_multimodal.csv'
with open(dataset_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

N = len(rows)
modalities = ['tremor_risk', 'finger_risk', 'gait_risk', 'voice_risk', 'questionnaire_risk']
mod_names = {
    'tremor_risk': '1. การสั่น (Tremor)',
    'finger_risk': '2. แตะนิ้ว (Finger Tap)',
    'gait_risk': '3. การเดิน (Gait)',
    'voice_risk': '4. เสียงพูด (Voice)',
    'questionnaire_risk': '5. แบบประเมิน (Questionnaire)'
}

# 1. 3/5 Majority Vote Rule Evaluation (and for 4 modalities, majority vote >= 2 or >= 3)
# When k=5, majority is >= 3 votes (>= 3/5)
# When k=4 (drop 1 modality), majority can be >= 3/4 or >= 2/4 (tie-breaker >= 2.5 or mean score >= 0.35)
def evaluate_ensemble(active_mods, threshold=0.35):
    k = len(active_mods)
    tp, fn, tn, fp = 0, 0, 0, 0
    
    for r in rows:
        y_true = int(r['is_parkinson'])
        scores = [float(r[m]) for m in active_mods]
        votes = sum(1 for s in scores if s >= threshold)
        avg_score = sum(scores) / k
        
        # Decision logic:
        # If 5 modalities: >= 3 votes -> PD (3/5 rule)
        # If 4 modalities: if votes >= 3 -> PD; if votes <= 1 -> Normal; if votes == 2 -> tie-break with avg_score >= 0.35
        if k == 5:
            y_pred = 1 if votes >= 3 else 0
        elif k == 4:
            if votes >= 3:
                y_pred = 1
            elif votes <= 1:
                y_pred = 0
            else: # votes == 2 (2 vs 2 tie)
                y_pred = 1 if avg_score >= threshold else 0
        else:
            y_pred = 1 if avg_score >= threshold else 0
            
        if y_true == 1 and y_pred == 1: tp += 1
        elif y_true == 1 and y_pred == 0: fn += 1
        elif y_true == 0 and y_pred == 0: tn += 1
        elif y_true == 0 and y_pred == 1: fp += 1

    acc = (tp + tn) / N * 100
    fnr = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0
    dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    
    return {
        'acc': acc, 'fnr': fnr, 'dor': dor,
        'sens': sensitivity, 'spec': specificity,
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp
    }

baseline = evaluate_ensemble(modalities)

print("="*80)
print(f"BASELINE: ระบบตรวจครบ 5 มิติ (เกณฑ์ 3/5 Majority Vote)")
print(f"  - ความแม่นยำรวม (Accuracy) : {baseline['acc']:.2f}%  ({baseline['tp']+baseline['tn']}/{N} คน)")
print(f"  - อัตราหลุดตรวจ (FNR)       : {baseline['fnr']:.2f}%  (หลุดตรวจเพียง {baseline['fn']}/31 คน)")
print(f"  - ความไว (Sensitivity)     : {baseline['sens']:.2f}% ({baseline['tp']}/31 คน)")
print(f"  - ความจำเพาะ (Specificity)  : {baseline['spec']:.2f}% ({baseline['tn']}/11 คน)")
print(f"  - ดัชนีวินิจฉัย (DOR)        : {baseline['dor']:.1f}")
print("="*80)

print("\nผลกระทบเมื่อ 'ตัดมิตินั้นออก' (Ablation / Leave-One-Out Test):")
print(f"{'มิติที่ถูกตัดออก':<30} {'ความแม่นยำที่เหลือ':<18} {'ความแม่นยำลดลง':<18} {'หลุดตรวจเพิ่มขึ้น (FNR)':<22}")
print("-"*88)

results = []
for m in modalities:
    remaining = [x for x in modalities if x != m]
    res = evaluate_ensemble(remaining)
    acc_drop = baseline['acc'] - res['acc']
    fnr_inc = res['fnr'] - baseline['fnr']
    results.append({
        'dropped': m,
        'name': mod_names[m],
        'acc': res['acc'],
        'acc_drop': acc_drop,
        'fnr': res['fnr'],
        'fnr_inc': fnr_inc,
        'dor': res['dor'],
        'sens': res['sens'],
        'spec': res['spec'],
        'tp': res['tp'], 'fn': res['fn'], 'tn': res['tn'], 'fp': res['fp']
    })

# Sort by impact (highest acc_drop / highest fnr_inc)
results.sort(key=lambda x: (x['acc_drop'], x['fnr_inc']), reverse=True)

for r in results:
    print(f"{r['name']:<30} {r['acc']:>6.2f}%              -{r['acc_drop']:>5.2f}%             +{r['fnr_inc']:>5.2f}% (เป็น {r['fnr']:.2f}%)")

print("="*80)
for r in results:
    print(f"\n[ถ้าขาด {r['name']}]")
    print(f"  - ความแม่นยำจะลดลงจาก {baseline['acc']:.2f}% เหลือเพียง {r['acc']:.2f}% (ลดลงไป {r['acc_drop']:.2f}%)")
    print(f"  - อัตราหลุดตรวจ (FNR) จะเพิ่มขึ้นจาก {baseline['fnr']:.2f}% เป็น {r['fnr']:.2f}% (คนป่วยหลุดตรวจเพิ่มขึ้นเป็น {r['fn']} คน)")
    print(f"  - ผลการจำแนก: TP={r['tp']}, FN={r['fn']}, TN={r['tn']}, FP={r['fp']}")
    print(f"  - ดัชนีวินิจฉัย (DOR) ตกฮวบเหลือ {r['dor']:.1f}")
