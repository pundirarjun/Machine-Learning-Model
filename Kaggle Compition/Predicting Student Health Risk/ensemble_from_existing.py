"""
Ensemble existing submission CSVs by majority vote (mode) per id using Python stdlib
Writes: submissions/submission_ensemble_majority_existing.csv

This script does not use pandas, only csv and collections.
"""
import csv
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent
SUB_DIR = ROOT / 'submissions'
OUT_PATH = SUB_DIR / 'submission_ensemble_majority_existing.csv'

# Collect CSV files in submissions dir
csv_files = sorted([p for p in SUB_DIR.iterdir() if p.suffix.lower() == '.csv'])
if not csv_files:
    raise SystemExit('No CSV files found in submissions/')

# Read sample submission to get the id order
sample_path = ROOT / 'sample_submission.csv'
if not sample_path.exists():
    # try to infer IDs from first CSV
    with open(csv_files[0], 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        ids = [row[0] for row in reader]
else:
    with open(sample_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        ids = [row[0] for row in reader]

preds_per_id = defaultdict(list)

for p in csv_files:
    # skip files that look like not full submissions (heuristic)
    name = p.name.lower()
    if 'eda' in name:
        continue
    try:
        with open(p, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            h = next(reader)
            # assume first column is id, second is target
            for row in reader:
                if not row:
                    continue
                idx = row[0]
                pred = row[1]
                preds_per_id[idx].append(pred)
    except Exception as e:
        print(f'Warning: failed to read {p.name}: {e}')

# For ids not present in preds_per_id, fill with empty list
for idx in ids:
    preds_per_id.setdefault(idx, [])

# Compute majority vote; if tie or no preds, fall back to the most common across all models or to '0'
# Determine global most common class
global_counter = Counter()
for v in preds_per_id.values():
    global_counter.update(v)
most_common_global = global_counter.most_common(1)[0][0] if global_counter else '0'

ensemble_preds = []
for idx in ids:
    votes = preds_per_id[idx]
    if not votes:
        ensemble_preds.append(most_common_global)
        continue
    ctr = Counter(votes)
    top = ctr.most_common()
    if len(top) == 1:
        ensemble_preds.append(top[0][0])
    else:
        # check tie
        if top[0][1] > top[1][1]:
            ensemble_preds.append(top[0][0])
        else:
            # tie -> choose class with higher global frequency
            tied = [c for c,count in top if count == top[0][1]]
            tied.sort(key=lambda x: global_counter.get(x,0), reverse=True)
            ensemble_preds.append(tied[0])

# Write output CSV
with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # header: use sample header if available
    if sample_path.exists():
        with open(sample_path, 'r', newline='', encoding='utf-8') as sf:
            sreader = csv.reader(sf)
            sheader = next(sreader)
            writer.writerow(sheader)
    else:
        writer.writerow(['id','health_condition'])
    for idx, pred in zip(ids, ensemble_preds):
        writer.writerow([idx, pred])

print('Wrote ensemble submission to', OUT_PATH)
