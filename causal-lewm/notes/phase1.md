# Phase 1 — Push-T prototype

Goal: cheapest path to a Causal-LeWM run that we can compare against a
frozen-encoder C-JEPA baseline on Push-T.

## Concrete TODO

- [ ] Run `scripts/setup_env.sh`; confirm `python -c "import stable_worldmodel"` works.
- [ ] Download Push-T expert HDF5 to `$STABLEWM_HOME` (see `../le-wm/README.md`).
- [ ] Pull C-JEPA's VideoSAUR slot attention into `src/slot_attention.py`
      (reference: `../cjepa/src/third_party/videosaur/videosaur/modules/`).
- [ ] Subclass `JEPA` in `../le-wm/jepa.py` to:
      - replace `[CLS]` token with slot tensor
      - apply `sample_object_mask` to history + future slots
      - add `per_slot_sigreg` and `slot_variance_diversity` to the loss
- [ ] Mirror `le-wm/config/train/lewm.yaml` into `configs/causal_lewm.yaml`
      with new keys: `num_slots`, `lambda_div`, `mask_ratio`, `warmup_steps`.
- [ ] Train w/ frozen DINOv2 first (sanity check: should match LeWM-Push-T baseline).
- [ ] Unfreeze encoder, enable masking curriculum, re-run.

## Open questions

- Per-slot vs joint-flattened SIGReg (§4.3) — start A, log both.
- N (slot count) for Push-T: C-JEPA uses 7 — likely fine.
- Does slot attention need its own warmup before masking kicks in?
  Plan says yes (§4.4 Phase 1).
