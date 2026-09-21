import csv
import math
import statistics
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

dataset_path = 'backend/datasets/real_clinically_matched_multimodal.csv'

with open(dataset_path, 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))

print(f"Total subjects loaded: {len(reader)}")

modalities = [
    ('questionnaire_risk', '5. แบบประเมิน (Questionnaire)'),
    ('gait_risk', '4. การเดิน (Gait)'),
    ('finger_risk', '3. แตะนิ้ว (Finger Tap)'),
    ('tremor_risk', '2. การสั่น (Tremor)'),
    ('voice_risk', '1. เสียงพูด (Voice)')
]

def pearson(x, y):
    n = len(x)
    mx = statistics.mean(x)
    my = statistics.mean(y)
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    sx = math.sqrt(sum((x[i] - mx)**2 for i in range(n)))
    sy = math.sqrt(sum((y[i] - my)**2 for i in range(n)))
    return cov / (sx * sy) if sx * sy != 0 else 0

def rank(vals):
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0] * len(vals)
    for r, (i, v) in enumerate(indexed):
        ranks[i] = r + 1
    return ranks

def spearman(x, y):
    return pearson(rank(x), rank(y))

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    m1, m2 = statistics.mean(group1), statistics.mean(group2)
    v1 = statistics.variance(group1) if n1 > 1 else 0
    v2 = statistics.variance(group2) if n2 > 1 else 0
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled_sd if pooled_sd != 0 else 0

def calc_auc(scores, targets):
    pos = [scores[i] for i in range(len(targets)) if targets[i] == 1]
    neg = [scores[i] for i in range(len(targets)) if targets[i] == 0]
    wins = 0
    for p in pos:
        for n in neg:
            if p > n: wins += 1.0
            elif p == n: wins += 0.5
    return wins / (len(pos) * len(neg))

y_pd = [float(r['is_parkinson']) for r in reader]
y_updrs = [float(r['total_UPDRS']) for r in reader]
y_motor = [float(r['motor_UPDRS']) for r in reader]

results = []
for col, name in modalities:
    vals = [float(r[col]) for r in reader]
    pd_vals = [float(r[col]) for r in reader if float(r['is_parkinson']) == 1]
    hc_vals = [float(r[col]) for r in reader if float(r['is_parkinson']) == 0]
    
    r_pd = pearson(vals, y_pd)
    rho_pd = spearman(vals, y_pd)
    r_updrs = pearson(vals, y_updrs)
    r_motor = pearson(vals, y_motor)
    d = cohens_d(pd_vals, hc_vals)
    auc = calc_auc(vals, y_pd)
    
    # 0.35 threshold evaluation
    tp = sum(1 for v in pd_vals if v >= 0.35)
    fn = sum(1 for v in pd_vals if v < 0.35)
    tn = sum(1 for v in hc_vals if v < 0.35)
    fp = sum(1 for v in hc_vals if v >= 0.35)
    fnr = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
    sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    specificity = (tn / (fp + tn) * 100) if (fp + tn) > 0 else 0
    accuracy = ((tp + tn) / len(reader) * 100)
    
    # Adjusted DOR to avoid div-by-zero
    dor = ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))
    raw_dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else None
    
    results.append({
        'key': col,
        'name': name,
        'mean_pd': statistics.mean(pd_vals),
        'sd_pd': statistics.stdev(pd_vals),
        'mean_hc': statistics.mean(hc_vals),
        'sd_hc': statistics.stdev(hc_vals),
        'cohen_d': d,
        'auc': auc,
        'r_pd': r_pd,
        'rho_pd': rho_pd,
        'r_updrs': r_updrs,
        'r_motor': r_motor,
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
        'fnr': fnr,
        'fpr': fpr,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'dor': dor,
        'raw_dor': raw_dor
    })

# Logistic regression or multiple linear regression simulation to find standardized regression coefficients (Beta)
# Standardize features
def mean_sd(lst):
    m = statistics.mean(lst)
    s = statistics.stdev(lst)
    return m, s

# Standardized covariance / correlation matrix
all_features = [col for col, _ in modalities]
feat_data = {col: [float(r[col]) for r in reader] for col in all_features}

print("\n" + "="*80)
print(f"{'มิติการตรวจ':<25} {'Cohen d':<9} {'AUC-ROC':<9} {'r (is_PD)':<11} {'r (UPDRS)':<11} {'FNR (%)':<9} {'DOR':<8}")
print("="*80)

# Sort by Cohen's d (or correlation with is_parkinson / UPDRS)
results.sort(key=lambda x: x['cohen_d'], reverse=True)

for i, r in enumerate(results, 1):
    dor_str = f"{r['raw_dor']:.1f}" if r['raw_dor'] is not None else f"{r['dor']:.1f}*"
    print(f"{r['name']:<25} {r['cohen_d']:<9.3f} {r['auc']:<9.3f} {r['r_pd']:<11.3f} {r['r_updrs']:<11.3f} {r['fnr']:<9.2f} {dor_str:<8}")

print("="*80)

for r in results:
    print(f"\n[{r['name']}]")
    print(f"  - ค่าเฉลี่ยกลุ่มผู้ป่วย (PD): {r['mean_pd']:.4f} (±{r['sd_pd']:.4f})")
    print(f"  - ค่าเฉลี่ยกลุ่มปกติ (HC):  {r['mean_hc']:.4f} (±{r['sd_hc']:.4f})")
    print(f"  - Cohen's d: {r['cohen_d']:.3f} (ขนาดอิทธิพล / Effect size)")
    print(f"  - AUC-ROC: {r['auc']:.3f}")
    print(f"  - สหสัมพันธ์กับสถานะโรค r(is_PD): {r['r_pd']:.3f} (Spearman rho: {r['rho_pd']:.3f})")
    print(f"  - สหสัมพันธ์กับความรุนแรง r(total_UPDRS): {r['r_updrs']:.3f}")
    print(f"  - สหสัมพันธ์กับ motor_UPDRS: {r['r_motor']:.3f}")
    print(f"  - Confusion (เกณฑ์ 0.35): TP={r['tp']}, FN={r['fn']}, TN={r['tn']}, FP={r['fp']}")
    print(f"  - Sensitivity: {r['sensitivity']:.2f}%, Specificity: {r['specificity']:.2f}%, FNR: {r['fnr']:.2f}%")
