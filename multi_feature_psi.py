"""Per-feature PSI drift report. Run on master."""
import json, subprocess
import pandas as pd, numpy as np
from io import StringIO
from datetime import datetime


def load_csv_from_hdfs(path):
    proc = subprocess.run(['hdfs', 'dfs', '-cat', path], capture_output=True, text=True)
    return pd.read_csv(StringIO(proc.stdout))


baseline = load_csv_from_hdfs('/data/movielens/ratings.csv')
# 實務上 recent 從 HBase user_ratings 表或最近一天的 streaming history Parquet 拿
recent = baseline.sample(2000, random_state=int(datetime.now().timestamp()) % 1000)

for df in (baseline, recent):
    df['hour_of_day'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)


def psi(b, r, kind='continuous', bins=10):
    eps = 1e-4
    if kind == 'continuous':
        edges = np.histogram_bin_edges(b, bins=bins)
        b_pct, _ = np.histogram(b, bins=edges)
        r_pct, _ = np.histogram(r, bins=edges)
    else:
        cats = sorted(set(b) | set(r))
        b_pct = np.array([(b == c).sum() for c in cats], dtype=float)
        r_pct = np.array([(r == c).sum() for c in cats], dtype=float)
    b_pct = b_pct / max(b_pct.sum(), 1)
    r_pct = r_pct / max(r_pct.sum(), 1)
    b_pct = np.clip(b_pct, eps, None)
    r_pct = np.clip(r_pct, eps, None)
    return float(np.sum((r_pct - b_pct) * np.log(r_pct / b_pct)))


features = {
    'rating':      ('continuous',  5),
    'hour_of_day': ('continuous',  6),
    'day_of_week': ('categorical', None),
    'is_weekend':  ('categorical', None),
}
report = {}
for col, (kind, bins) in features.items():
    p_val = psi(baseline[col].values, recent[col].values, kind, bins or 10)
    report[col] = {
        'psi': round(p_val, 4),
        'status': 'STABLE' if p_val < 0.1 else ('WARNING' if p_val < 0.2 else 'ALERT'),
    }

overall_psi = max(v['psi'] for v in report.values())
overall_status = 'STABLE' if overall_psi < 0.1 else ('WARNING' if overall_psi < 0.2 else 'ALERT')

print('\n=== Multi-Feature Drift Report ===')
print(f"{'Feature':<20}{'PSI':>10}{'Status':>15}")
print('-' * 45)
for col, info in sorted(report.items(), key=lambda x: -x[1]['psi']):
    marker = ' <--' if info['status'] == 'ALERT' else ''
    print(f"{col:<20}{info['psi']:>10.4f}{info['status']:>15}{marker}")
print('-' * 45)
print(f"{'OVERALL':<20}{overall_psi:>10.4f}{overall_status:>15}")

out = {
    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'features': report,
    'overall_psi': overall_psi,
    'overall_status': overall_status,
}
with open('/tmp/multi_drift_report.json', 'w') as f:
    json.dump(out, f, indent=2)
subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', '/data/drift_reports'])
subprocess.run(['hdfs', 'dfs', '-put', '-f', '/tmp/multi_drift_report.json',
                f"/data/drift_reports/drift_{datetime.now().strftime('%Y%m%d_%H%M')}.json"])
print('Report saved to /data/drift_reports/')
