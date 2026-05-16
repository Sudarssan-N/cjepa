# Causal-LeWM

End-to-end object-centric world model combining **C-JEPA**'s object-level
masking with **LeWM**'s SIGReg regularization. See
[`../research-gap_cjepa-sigreg.md`](../research-gap_cjepa-sigreg.md) for the
full research proposal.

## Layout

- `src/`        — model code (slot attention wrapper, object-level masking, diversity losses)
- `configs/`   — Hydra configs (will mirror `le-wm/config/`)
- `scripts/`   — env + data setup helpers
- `notes/`     — running design notes and ablation logs

## Upstream repos (cloned siblings)

- `../cjepa/`  — reference for object-level masking, VideoSAUR encoder, ALOE eval
- `../le-wm/` — starting point for end-to-end training loop, SIGReg, MPC planning

## Setup

```bash
bash scripts/setup_env.sh
```

This creates a single conda env `causal-lewm` (Python 3.10) and installs
LeWM's dependencies plus the slot-attention extras we need from C-JEPA.

## First milestone (Phase 1)

Per §7 of the research plan:

1. Fork LeWM training loop (`../le-wm/train.py`, `../le-wm/jepa.py`).
2. Insert Slot Attention after the ViT encoder; replace `[CLS]` extraction
   with `N` slot vectors.
3. Implement object-level masking inside the predictor.
4. Add per-slot SIGReg + a slot-diversity term (start with VICReg-style
   per-slot variance).
5. Train + evaluate on Push-T against a frozen-encoder C-JEPA baseline.
