#!/usr/bin/env bash
# Runs the planning evals that complete the paper's §5.6 table, in one go.
# LeWM-matched protocol (50 episodes, seed 42, goal_offset 25, budget 50,
# CEM 300x30 top-10%) — all defaults of scripts/eval_planning.py.
#
# Colab usage (GPU runtime, Drive mounted):
#   %cd /content/cjepa/causal-lewm
#   !git pull
#   !bash scripts/colab_planning_evals.sh
#
# Overridable via env vars: DATA, DRIVE, OUT, NUM_EVAL.
set -euo pipefail

DATA=${DATA:-/content/pusht_expert_train.h5}
DRIVE=${DRIVE:-/content/drive/MyDrive/DataPoints/cjepa/causal-lewm/outputs}
OUT=${OUT:-planning_results.jsonl}
NUM_EVAL=${NUM_EVAL:-50}

pip install -q stable-worldmodel hdf5plugin

# Fail fast if there's no GPU — the "ours" CEM evals are ~100x slower on CPU
# (tens of minutes per chunk). The random baseline is fine on CPU.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo
  echo "########################################################################"
  echo "# NO GPU DETECTED. The 'ours' planning evals will be unusably slow on   #"
  echo "# CPU. In Colab: Runtime > Change runtime type > GPU (T4), then rerun.  #"
  echo "# (Set ALLOW_CPU=1 to run anyway — not recommended.)                    #"
  echo "########################################################################"
  echo
  [ "${ALLOW_CPU:-0}" = "1" ] || exit 1
fi

# Fetch the dataset if neither the .h5 nor its .zst is present
# (eval_planning.py auto-decompresses the .zst on first use).
if [ ! -f "$DATA" ] && [ ! -f "$DATA.zst" ]; then
  echo "downloading pusht_expert_train.h5.zst (13.1 GB) ..."
  python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("quentinll/lewm-pusht", "pusht_expert_train.h5.zst",
                repo_type="dataset", local_dir="/content")
PY
fi

run_ckpt() {  # run_ckpt <label> <checkpoint>
  echo
  echo "================  $1  ================"
  if [ ! -f "$2" ]; then
    echo "SKIP: checkpoint not found: $2"
    return 0
  fi
  python scripts/eval_planning.py \
      --data "$DATA" --policy ours --num-eval "$NUM_EVAL" \
      --ckpt "$2" --out "$OUT"
}

# Random baseline (cheap; recorded at 2.0% — rerun only if OUT lacks it)
if ! grep -qs '"policy": "random"' "$OUT"; then
  echo "================  random baseline  ================"
  python scripts/eval_planning.py \
      --data "$DATA" --policy random --num-eval "$NUM_EVAL" --out "$OUT"
fi

run_ckpt "frozen SAVi-ON, seed 0 (ours, headline)" "$DRIVE/2026-05-26_14-56-02/final.pt"
run_ckpt "SAVi-OFF ablation, seed 0"               "$DRIVE/2026-05-26_16-39-27/final.pt"
# End-to-end Run 13b — runs only if you've copied its final.pt to Drive:
run_ckpt "end-to-end DINOv2, Run 13b"              "$DRIVE/2026-06-11_15-27-51/final.pt"

echo
echo "================  paste these back  ================"
cat "$OUT"
