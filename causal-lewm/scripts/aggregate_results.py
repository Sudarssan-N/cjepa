"""Aggregate run summaries (results.jsonl) into a mean±std table.

Each training run appends one JSON line via train.py. This collapses a
multi-seed / ablation sweep into a compact table grouped by the SAVi flag.

Usage:
  python scripts/aggregate_results.py [results.jsonl]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
if not rows:
    sys.exit(f"no records in {path}")

# Group by the experimental condition (everything except seed / out_dir).
def key(r):
    return (r["slot_propagate"], r.get("max_episodes"), r["mask_target"], r["steps"])

groups = defaultdict(list)
for r in rows:
    groups[key(r)].append(r)

metrics = ["nmse", "pcos", "pred", "recon", "slot_sim"]
print(f"{'SAVi':>5} {'eps':>5} {'mask':>5} {'steps':>6} {'n':>3} | "
      + " | ".join(f"{m:>15}" for m in metrics))
for (prop, eps, mask, steps), rs in sorted(groups.items()):
    seeds = [r["seed"] for r in rs]
    cells = []
    for m in metrics:
        vals = np.array([r[m] for r in rs], dtype=float)
        cells.append(f"{vals.mean():.3f}±{vals.std():.3f}")
    print(f"{str(prop):>5} {str(eps):>5} {mask:>5.2f} {steps:>6} {len(rs):>3} | "
          + " | ".join(f"{c:>15}" for c in cells)
          + f"   seeds={seeds}")
