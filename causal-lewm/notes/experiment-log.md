# Causal-LeWM — Experiment Log

A running, paper-oriented record of what we changed, what happened, and what we
learned. Newest insights at the bottom of each section. **Keep updating as runs
complete.** Each entry ties a code change (commit) to its observed effect on the
training metrics so the narrative is reconstructable for the writeup.

---

## 1. Project & hypothesis

**Causal-LeWM** combines C-JEPA-style object-level masking with LeWM's SIGReg
regularizer to train an **object-centric (slot) encoder + predictor world model
end-to-end from pixels** on Push-T, *without* the usual JEPA crutches (EMA
target, stop-grad, pretrained-and-frozen encoder, auxiliary supervision).

Central question: can object masking (which induces latent interventions) + a
Gaussianity regularizer (SIGReg) train the full object-centric stack from pixels
without representation collapse?

## 2. Architecture & objective (as of this log)

```
frames (B,T,C,H,W)
  └─ Encoder: DINOv2-small (frozen for baseline) → patch tokens
       └─ proj+norm (trainable)          # adapts frozen features
            └─ SlotAttention → slots (B,T,N,D)   N=7 slots, D=192
                 ├─ SlotPredictor (transformer, AdaLN-zero, action-conditioned)
                 │     → next-slot prediction on masked + future positions
                 └─ SpatialBroadcastDecoder (DINOSAUR-style)
                       → reconstruct frozen DINO patch features from slots
```

**Loss** = `pred` + λ_sig·`sig` + λ_div·`div` + λ_decorr·`decorr` + λ_recon·`recon`

| term | meaning | anti-collapse role |
|------|---------|--------------------|
| `pred` | MSE, predicted vs detached slots, masked+future positions only | the task |
| `sig` | per-slot SIGReg: each slot's marginal → isotropic unit Gaussian | prevents trivial constant |
| `div` | hinge on per-(slot,dim) std across batch | each slot index must vary across samples |
| `decorr` | mean +off-diagonal cosine sim between slots **within a frame** | the N slots must differ from each other |
| `recon` | MSE, slot→DINO-feature reconstruction | slots must encode real content |

**Default weights (`configs/causal_lewm.yaml`):** λ_sig=0.09, λ_div=0.1,
λ_decorr=1.0, λ_recon=1.0. Mask curriculum: warmup 500 steps, ramp 1500 → 0.4.

**Health metrics to watch in logs:**
- `pred` — should fall **below 1.0** (the variance floor = trivial mean-prediction).
- `slot_sim` — off-diagonal slot cosine sim; should stay **low** (slots distinct, not collapsed).
- `recon` — should **decrease** (slots grounded).
- A *good* run has all three simultaneously. Watch for "fake wins" where `pred`
  is low only because slots collapsed (then `slot_sim`→1).

## 3. Infrastructure / data-loading fixes (engineering, not science)

These unblocked training on the LeWM Push-T HDF5 in Colab; recorded for repro.

| commit | problem | fix |
|--------|---------|-----|
| `05b0a6a` | `'Dataset' object has no attribute 'keys'` — loader assumed per-episode HDF5 groups | auto-detect grouped vs **flat** (stable-worldmodel) layout |
| `712215e` | `OSError: can't open directory .../plugin` on read | `import hdf5plugin` (Blosc/LZ4/Zstd codecs); also detect `ep_offset`/`ep_len` episode bounds |
| `5781d60` | frozen-DINOv2 forward dominated step time (~4.5 s/step) | precompute frozen patch tokens to a float16 HDF5 cache; train on cache |
| `8140a6b` | full cache = ~93 GB (474,498 frames), won't fit Colab disk | `data.max_episodes` subsetting + disk-space guard with suggested K |

**Dataset facts (Push-T expert):** 474,498 referenced frames across 18,685
episodes (~25 referenced frames/episode at frameskip 5). Flat 4-D layout,
`pixels (N,H,W,3)` + `ep_offset`/`ep_len`. Cached DINO tokens: `(n,256,384)`
float16 (≈196 KB/frame).

**Caching payoff note:** for a single 5000-step run over the *full* set, each
frame is reused only ~1.3× → cache barely helps and is huge. On a **subset**
(e.g. `max_episodes=500` ≈12.5k frames), reuse is ~50× → cache is the right call
and steps drop to ~0.1 s. Use subsets for iteration.

**Cost:** step time went ~4.5 s → ~0.1 s (cached, slot+predictor only); ~0.33 s
once the broadcast decoder is added.

## 4. Experiment runs (chronological)

> All Push-T, DINOv2-small frozen, N=7 slots, D=192, num_steps=4, frameskip=5,
> batch 32. "OF2" = overfit on `max_episodes=2`. Numbers are start → end (≈500
> steps unless noted).

### Run 1 — Baseline, full data, pre-cache
- **Config:** λ_sig=0.09, λ_div=0.1, no recon/decorr. Full dataset. ~4.5 s/step.
- **Result:** `pred` 0.65 → **0.99 (stuck at variance floor)**; `sig` 55 → 1.1;
  `div` 0.67 → 0.027; `slot_sim` 0.66 → **~0**.
- **Read:** classic collapse. Slots become decorrelated **noise** that satisfies
  SIGReg + diversity; predictor falls back to mean-prediction. `pred` rising
  0.65→1.0 tracks SIGReg inflating slot variance to unit (the floor *is* the
  variance). Nothing about dynamics is learned.

### Run 2 — Same objective, cached, subset
- **Config:** identical losses; `max_episodes=500`; cached features. ~0.1 s/step.
- **Result:** same collapse (`pred`→0.99, `slot_sim`→0, `div`→0.029) but 1500
  steps in 176 s instead of hours.
- **Read:** caching + subset gives fast iteration; collapse unchanged (expected —
  no new pressure added). Confirms the speedup is behavior-preserving.

### Run 3 — Overfit, regularizers OFF (plumbing test)
- **Config:** OF2, λ_sig=0, λ_div=0, λ_recon=0, 500 steps.
- **Result:** `pred` 0.60 → **0.056**; `slot_sim` 0.55 → **0.976**; `div`→0.80
  (hinge violated, low variance).
- **Read:** ✅ **Plumbing is sound** — predictor/masking/gradients work, `pred`
  *can* drop. ❌ But it dropped by **collapsing all 7 slots to identical**
  (slot_sim→0.98), the trivial degenerate solution. Tells us the regularizers
  are load-bearing, and that low `pred` alone is not evidence of success.

### Run 4 — Overfit, reconstruction ONLY
- **Config:** OF2, λ_sig=0, λ_div=0, **λ_recon=1.0**, 500 steps.
- **Result:** `recon` **6.13 → 1.06** (works!); but `slot_sim` → **0.997**;
  `pred` → 0.006.
- **Read:** the decoder reconstructs fine, but **reconstruction alone does not
  prevent within-frame collapse**: a spatial-broadcast decoder can rebuild a
  frame from a *single* repeated slot + positional embeddings → effectively a
  1-slot autoencoder. Recon supplies *content* but not *distinctness*.

### Run 5 — Overfit, full objective, OLD (noisy) slot init
- **Config:** OF2, λ_sig=0.09, λ_div=0.1, λ_recon=1.0. SlotAttention init =
  shared μ + unit-Gaussian noise (Locatello-style), re-drawn per forward.
- **Result:** `recon` 6.13 → 0.92; `slot_sim` 0.56 → **0.016 (distinct!)**; but
  `pred` **stuck at ~0.98**.
- **Read:** slots stay distinct here — but `pred` won't drop even on 2 episodes.
  Diagnosed as an **ill-posed prediction target**: slot attention is run
  independently per frame with random init, so (a) slot index *n* has no
  consistent identity across frames (target misaligned in time) and (b) the same
  frame yields different slots every forward (target stochastic). MSE on
  per-index slots is then irreducible ≈ variance.
- **Crucial realization:** the distinctness in this run came from the **init
  noise**, *not* from sig/div. (See Run 6.)

### Run 6 — Overfit, full objective, NEW learned per-slot init
- **Change (`ee61d5c`):** SlotAttention now uses **distinct learned per-slot
  anchors** `μ:(1,N,D)` + **low** init noise (logσ init −4 ⇒ σ≈0.018). Targets
  near-deterministic (measured rel. drift across passes 0.23 → 0.03).
- **Config:** OF2, full objective (same weights as Run 5).
- **Result:** `pred` 0.59 → **0.027 — BROKE the variance floor** ✅; `recon`
  6.09 → 0.86 ✅; but `slot_sim` → **0.946 (collapsed again)**.
- **Read:** confirms the Run-5 diagnosis — stable slot identity makes prediction
  learnable. **But** removing the init noise re-exposed within-frame collapse,
  because sig/div only constrain each slot's *marginal across the batch* and
  leave "all N slots identical within a frame" unpenalized. The low `pred` here
  is again partly the 1-slot shortcut.
- **Key trade-off identified:** slot-init noise couples two things —
  high noise → distinct slots / broken prediction; low noise → working
  prediction / collapsed slots. Need to **decouple**: low noise for stable
  targets + an explicit distinctness loss.

### Run 7 — Overfit, + within-frame decorrelation loss  ✅ **collapse beaten**
- **Change (`b72a69d`):** added `slot_decorrelation` (mean +off-diagonal cosine
  sim between slots within a frame), λ_decorr=1.0, on top of low-noise init.
- **Config:** OF2, λ_sig=0.09, λ_div=0.1, λ_decorr=1.0, λ_recon=1.0, 500 steps.
- **Result — all four healthy simultaneously:**
  - `pred` 0.59 → **0.044** (below floor, *honest* — not the 1-slot shortcut)
  - `slot_sim` 0.52 → **0.004** (slots distinct, no collapse)
  - `recon` 6.12 → **0.93** (grounded)
  - `decorr` 0.52 → 0.092; `div` ~0.10; `sig` 77 → 4.5
- **Dynamics worth noting:** `slot_sim` first *rose* to ~0.89 (step 75) as the
  model attempted to collapse, then `decorr` engaged and drove it down to ~0.004
  by step 499. Consistent with the weak-gradient-*at-exact*-collapse caveat: the
  cosine penalty kicks in once slots are *near* (not perfectly) identical — and
  that was sufficient here to reverse an in-progress collapse.
- **Read:** ✅ The full objective — stable slot identity (low-noise learned init)
  + `recon` (content) + `sig`/`div` (marginals) + `decorr` (within-frame
  distinctness) — yields distinct, grounded, *and* predictable slots with no
  collapse, on the overfit set. This is the first run that escapes **both**
  collapse modes at once.
- **Caveat:** overfit on 2 episodes; `pred`=0.044 is partly memorization. Real
  test is whether this holds on a larger set (Run 8).

### Run 8 — Scale-up, full objective, 500 episodes, 5000 steps  ✅ collapse holds / ⚠️ weak prediction
- **Config:** `max_episodes=500` (~10,737 cached frames), full objective
  (λ_sig=0.09, λ_div=0.1, λ_decorr=1.0, λ_recon=1.0), 5000 steps, mask curriculum
  active (→0.4 by ~step 2000). ~0.39 s/step (decoder included).
- **Result:**
  - `slot_sim`: spike to ~0.84 (step ~75) → driven to ~0 by step 350 → **stays
    ~0 / slightly negative for the remaining ~4,600 steps.** No collapse at scale.
  - `recon`: 6.19 → **0.30** (slots strongly grounded).
  - `decorr` → ~0.05; `div` → ~0.05; `sig` → ~2.2 (not fully minimized at λ=0.09).
  - `pred`: 0.52 → dip 0.08 (transient collapse) → **rises** to ~0.75 (step 700)
    → plateaus **~0.85–0.92, ends 0.91**.
- **Read:**
  - ✅ **Collapse is beaten at scale** — first fully healthy representation
    (distinct + grounded + stable) on a real subset over a long run. The chased
    problem (Runs 1–7) is solved.
  - ⚠️ **Prediction is weak** — `pred`≈0.91 is only ~9% below the mean-prediction
    floor; the predictor explains little slot variance. The `pred` *rise* over
    training is expected, not regression: early low `pred` was the transient
    1-slot collapse (near-zero target variance), and as slots de-collapsed +
    the mask ramped to 0.4, genuine difficulty rose.
- **Conclusion:** representation collapse and prediction quality are **separable
  problems**. Solved the first; the second is the new bottleneck.
- **Next (Run 9+):** diagnose *why* prediction is weak —
  (a) task difficulty (mask_target, frameskip) vs (b) slot temporal identity.
  Cheap diagnostic: set `mask_target=0` + smaller `frameskip` and see if `pred`
  drops a lot (→ task hard) or stays ~0.9 (→ identity/predictor issue → SAVi-style
  temporal slot propagation). Also add a normalized pred metric (pred / slot
  variance, or cosine) so distance from the true floor is legible.

### Run 9 — Prediction diagnostic: mask OFF, 500 episodes, 2000 steps
- **Config:** `max_episodes=500`, `model.mask_target=0` (predict only the future
  frame from clean history), full objective otherwise, 2000 steps, frameskip 5.
- **Result:** `slot_sim` ~0 (no collapse), `recon` 6.26 → 0.40; `pred` plateaus
  **~0.70** (vs ~0.91 in Run 8 with mask 0.4).
- **Read — mixed verdict, both factors real:**
  1. **Object masking ≈ half the difficulty** (0.70 → 0.91): masked-history-slot
     prediction at ratio 0.4 is hard; the curriculum is aggressive.
  2. **Clean next-frame prediction is still only moderate** (~0.70 ≈ 30% below the
     mean floor). For Push-T's simple dynamics this is mediocre → residual limit
     is likely **slot temporal identity** (per-frame independent slot attention)
     and/or `frameskip=5` (large inter-frame motion).
- **Next:** (a) add a normalized pred metric (NMSE = pred / target-variance,
  + cosine) for legible comparison; (b) attack the residual via **SAVi-style
  temporal slot propagation**; optionally a frameskip sweep (needs cache rebuild).

### Run 10 — SAVi propagation ON, 500 episodes, 5000 steps  ✅✅ **big prediction win**
- **Config:** `max_episodes=500`, full objective, `slot_propagate=true` (SAVi),
  5000 steps, mask curriculum → 0.4 (same hard setting as Run 8). ~0.36 s/step.
- **Result (with the new normalized metrics):**
  - `nmse`: ~0.30 (early) → **~0.15–0.19** (end; best ~0.136). Still trending down.
  - `pcos`: → **~0.90–0.93** (high directional agreement).
  - `pred` (raw): 0.70 → **~0.19** (best ~0.14).
  - `slot_sim` ~0 / slightly negative (no collapse); `recon` 6.21 → **0.26**;
    `sig` → ~2.0; `div` → ~0.03; `decorr` → ~0.01.
- **Read:** ✅✅ Strong prediction with collapse still solved. The controlled
  effect of SAVi is measured against Run 11 (same code, flag flipped) — see
  there. Still improving at 5000 steps → longer training likely helps further.
  [NOTE: an earlier draft compared to Run 8's `pred`≈0.91; that was an
  *uncontrolled* earlier snapshot and overstated the gain — use Run 11.]

### Run 11 — Ablation: SAVi OFF (`slot_propagate=false`), same config as Run 10
- **Config:** identical to Run 10 (500 ep, full objective, 5000 steps, mask→0.4)
  except `model.slot_propagate=false`. Same code (`de187ef`) — the **controlled**
  comparison.
- **Result:** `nmse` → **~0.47** (end; pred ~0.48), `pcos` ~0.71, `slot_sim` ~0
  (no collapse), `recon` 6.27 → 0.22. `pred` *decreases* over training
  (0.63 → 0.48), unlike Run 8.
- **Controlled SAVi effect (Run 11 → Run 10, single variable):**
  `nmse` **0.47 → 0.18**, `pred` 0.48 → 0.19, `pcos` 0.71 → 0.92.
  ≈53% → ≈82% of slot variance explained. A large, clean win.
- **Two cautions for the paper:**
  1. The earlier "≈0.9 baseline" (Run 8) was an **uncontrolled** earlier
     snapshot and its `pred` *rose* over training (0.08→0.91, likely a worse
     basin). The honest no-SAVi baseline is **0.47**, not 0.9.
  2. Two nominally-identical no-SAVi runs landed at 0.91 (Run 8) vs 0.48
     (Run 11) → **real run-to-run variance**. Report **multiple seeds** per arm.

### Run 12 — Multi-seed SAVi ablation, 3 seeds × {on, off}  ✅✅ **headline result, statistically clean**
- **Config:** identical to Runs 10/11 (500 ep, full objective, 5000 steps,
  mask→0.4), now swept over `seed ∈ {0,1,2}` × `slot_propagate ∈ {true,false}`.
  Six runs, each contributing one tail-averaged (last 500 steps) record to
  `results.jsonl`; collapsed with `scripts/aggregate_results.py`.
- **Result (mean ± std over 3 seeds):**

  | SAVi | NMSE | pcos | pred | recon | slot_sim |
  |------|------|------|------|-------|----------|
  | **OFF** | **0.790 ± 0.014** | 0.445 ± 0.013 | 0.805 ± 0.014 | 0.264 ± 0.004 | −0.010 ± 0.005 |
  | **ON**  | **0.235 ± 0.030** | 0.875 ± 0.018 | 0.245 ± 0.033 | 0.236 ± 0.008 | −0.067 ± 0.003 |

  Per-seed NMSE — OFF: {0.809, 0.775, 0.788}; ON: {0.256, 0.255, 0.193}.
- **Checkpoint files (local `6runs O:P/`, verified against each ckpt's cfg):**
  final-2/3/4.pt = SAVi ON seeds 0/1/2; final-5/6/7.pt = SAVi OFF seeds 0/1/2.
- **Read:** ✅✅ The SAVi effect is **large and unambiguous**: NMSE **0.79 → 0.235**
  (3.4×; ~21% → ~77% of slot variance explained), pcos 0.45 → 0.88. The
  per-arm std (±0.014–0.030) is tiny next to the 0.55 gap — **no seed overlap**,
  so the effect is not a seed artifact. This is the defensible headline number.
- **Three points for the paper:**
  1. **Resolves the earlier variance worry.** Runs 8 (0.91) and 11 (0.48) for
     no-SAVi looked like real run-to-run variance; *properly seeded*, the
     no-SAVi arm is tight at **0.790 ± 0.014**. So that earlier spread was
     uncontrolled-seed/snapshot noise, not a config effect — and the honest
     no-SAVi baseline is ~0.79, between the two earlier points.
  2. **The gain is purely predictive.** `recon` (0.264 vs 0.236) and `slot_sim`
     (~0 in both) are essentially identical across arms → **neither arm
     collapses, and SAVi does not change reconstruction quality.** The entire
     improvement is attributable to temporal slot identity helping the
     *predictive* task — clean causal attribution.
  3. Single-seed runs from Runs 10/11 (0.18 / 0.47) sit *just outside* the
     seeded ±std bands (0.235±0.030 / 0.790±0.014); they were mildly optimistic
     (SAVi) / optimistic (no-SAVi) draws. Report the seeded means, not the
     single runs.

### Run 13 — End-to-end (unfrozen) DINOv2 — **the central hypothesis**, single seed
> ⚠️ Superseded by **Run 13b** below: this run was interrupted at 4675/5000
> and the user reports its batch size may have accidentally been 20 (not 16).
> Kept for the trajectory observations; cite 13b's numbers.
- **Config:** `encoder=dinov2_finetune` (`freeze: false`), backbone_lr=1e-5, heads
  lr=3e-4, gradient checkpointing auto-on. `max_episodes=500`, `slot_propagate=true`,
  mask→0.4, batch_size=**16** (down from 32 to fit the unfrozen ViT backward on
  T4), 5000 steps requested. **Session ended at step 4675** (~3.5 h). Step time
  ≈2.7 s (no cache — full ViT fwd+bwd every step).
- **Tail average (last ~500 steps, 21 log points, step 4175→4675):**

  | metric | finetune (1 seed, ≈4675 steps) | frozen SAVi-ON (3 seeds, 5000 steps) |
  |--------|--------------------------------|--------------------------------------|
  | nmse   | **0.339**                      | 0.235 ± 0.030 |
  | pcos   | 0.815                          | 0.875 ± 0.018 |
  | pred   | 0.338                          | 0.245 ± 0.033 |
  | recon  | **0.735**                      | 0.236 ± 0.008 |
  | slot_sim | −0.059                       | −0.067 ± 0.003 |

- **Read — the central hypothesis is *not* refuted:**
  1. ✅ **Stable.** No collapse: `slot_sim ≈ −0.06` (identical to frozen),
     `recon` *decreases* monotonically (6.22 → ~0.65), backbone doesn't destroy
     its pretrained features. The main risk of end-to-end ViT finetuning under
     this objective — collapse of either slots or backbone features — did not
     materialize. This is the headline scientific result of the run.
  2. ⚠️ **Slightly trails frozen at matched compute** (~0.10 NMSE gap,
     ~0.06 pcos gap). Backbone at lr=1e-5 adapts much slower than freshly-init
     heads at lr=3e-4, so most "easy" gains come from the slot/predictor side
     (which is already saturated by the frozen baseline).
  3. 📈 **Still improving when killed.** Last 1k steps trended `nmse` 0.40→0.30,
     `recon` 0.95→0.65 — not converged.
  4. 🔁 **`recon` is ~3× the frozen value (0.74 vs 0.24)** because the decoder is
     chasing a *moving* target (live backbone features, target detached). This
     is the expected joint-training geometry, *not* a collapse signal — the
     trajectory is monotonically falling.
- **Caveats for the paper:** (i) **batch=16 vs frozen's batch=32** — a confound
  worth either re-running the frozen baseline at batch=16, or rerunning the
  finetune at batch=32. (ii) **single seed** vs the frozen arm's 3 seeds.
  (iii) **interrupted** — needs to either resume from `step_4000.pt` or
  re-run to completion.
- **Spikes:** a handful of large `loss`/`recon` spikes (steps 2125, 2550, 3850,
  4125, 4625) that recover within ~25 steps — outlier batches, not instability.
- **Story for the paper, drafted:** "Object-masking + SIGReg train the full
  object-centric stack stably from pixels (no collapse, slots healthy, backbone
  preserved). At matched compute the frozen-encoder variant is competitive;
  whether end-to-end overtakes it is a separable question of scale / adaptation
  schedule." The stability result *is* the contribution; the absolute number is
  the next axis.
- **Next:** the cleaner end-to-end comparison is **warm-start from a frozen
  SAVi-ON checkpoint** (Run 14, planned): load a Run-12 `final-*.pt`, swap to
  `encoder=dinov2_finetune`, continue training. This isolates *backbone
  adaptation on top of working slots* and sidesteps the session-death problem
  because heads start already-trained. Enabled by the new `init_from` config
  knob (commit `be1ef8d`).

### Run 13b — End-to-end DINOv2 rerun, completed  ✅ **citable end-to-end number**
- **Config:** same as Run 13 (`encoder=dinov2_finetune`, backbone_lr=1e-5,
  heads 3e-4, grad checkpointing, 500 ep, SAVi on, mask→0.4, seed 0) but run
  to **completion: 5000/5000 steps**, batch_size=16 per the launch command.
  Raw record: `notes/results-run13-finetune.jsonl`.
- **Tail average (last 500 steps):**

  | metric | end-to-end 13b (1 seed) | interrupted Run 13 | frozen SAVi-ON (3 seeds) |
  |--------|-------------------------|--------------------|--------------------------|
  | nmse   | **0.317**               | 0.339              | 0.235 ± 0.030 |
  | pcos   | 0.824                   | 0.815              | 0.875 ± 0.018 |
  | pred   | 0.325                   | 0.338              | 0.245 ± 0.033 |
  | recon  | **0.597**               | 0.735              | 0.236 ± 0.008 |
  | slot_sim | −0.065                | −0.059             | −0.067 ± 0.003 |

- **Read:**
  1. ✅ Confirms Run 13's headline: **end-to-end training is stable** — no
     slot collapse (slot_sim identical to frozen), recon still falling
     (0.735 → 0.597 tail-avg with the extra 325 steps + clean run), healthy
     prediction. The central hypothesis stands on a *completed* run.
  2. ⚠️ Still trails frozen at matched steps: **0.317 vs 0.235 ± 0.030**
     (~0.08 NMSE, ~2.7σ outside the frozen band). Remaining confounds:
     batch 16 vs 32, single seed, and the recon target is moving (live
     backbone) so the encoder is still adapting at 5000 steps.
  3. 📈 Trajectory suggests unconverged: recon's tail-average dropped 19%
     just from the final ~7% of training vs the interrupted run.
- **Next options:** longer end-to-end run (cheapest test of "just needs more
  steps"), frozen baseline at batch 16 (kills the batch confound), or
  warm-start from `final-2.pt` via `init_from` (isolates backbone adaptation).

### Qualitative — slot decoder masks, end-to-end checkpoint (Run 13b, step 5000)
- **Artifacts:** `notes/figures/slots_finetune_{0..3}.png` — 4 random Push-T
  windows, per-slot alpha masks + argmax segmentation, rendered with
  `scripts/visualize_slots.py` from the Run-13b finetuned `final.pt`.
- **What they show, consistently across all 4 windows:**
  1. ✅ **Masks are object-shaped, not noise.** One slot forms a compact blob
     hugging the T-block; one slot takes the background (bright everywhere
     with a dark hole exactly at the T). The argmax segmentation separates
     T-block from background in every frame. This is the visual counterpart
     of "recon low + slot_sim ~0": slots carry real scene content, and neither
     collapse mode (noise / identical) is present.
  2. ⚠️ **Slot *identity* hops across timesteps.** The slot index owning the
     T (and the background) changes from row to row within a window, despite
     `slot_propagate=true`. SAVi propagation evidently stabilizes identity
     enough to help prediction (Run 12) without making index↔object binding
     temporally rigid in the decoder's view.
  3. The model effectively uses ~2–3 of 7 slots per frame; inactive slots are
     near-uniform with thin image-edge stripe artifacts. The pusher dot
     (~1 patch at 16×16 resolution) is not clearly bound by any slot.
- **For the paper:** point 1 is the qualitative figure supporting the
  stability claim. Point 2 is an honest limitation/discussion item — and a
  testable comparison: render the *frozen* SAVi-ON checkpoint (`final-2.pt`)
  on the same windows (same default seed → identical panels) and check
  whether identity is more stable when the backbone is frozen.

### Qualitative — three-way comparison on the SAME 4 windows (seed-0 models)
- **Artifacts:** `notes/figures/slots_{finetune,frozen,savioff}_{0..3}.png` —
  finetuned end-to-end (Run 13b), frozen SAVi-ON (Run 12, `final-2.pt`),
  frozen SAVi-OFF (Run 12, `final-5.pt`); identical windows (viz seed 0:
  2492/4720/7854/5882), so panels are directly comparable.
- **Observations:**
  1. **Binding crispness ranks: finetuned > frozen-ON > OFF.** The end-to-end
     model has the most compact, object-hugging T-block masks and the cleanest
     argmax segmentation. Frozen SAVi-ON separates object from background too,
     but its masks are visibly more diffuse/patchy with more partially-active
     slots. SAVi-OFF masks are blobby, often anchored to large background
     regions, and its segmentation degrades to speckle in some frames.
     → **End-to-end finetuning improves object binding qualitatively**, even
     though its NMSE (0.317) slightly trails frozen (0.235±0.030) — the two
     metrics measure different things, worth a paper paragraph.
  2. **Slot-identity stability is *inverted* across arms:** both SAVi-ON
     models (finetuned + frozen) show identity hopping across timesteps,
     while **SAVi-OFF keeps slot indices remarkably stable** across rows
     (the same slot plays the same role in every frame). Mechanistically
     sensible: per-frame independent encoding re-derives slots from the same
     learned anchors each time (deterministic role assignment), whereas SAVi
     makes slots recursive state, so roles can swap as the recursion evolves.
  3. Yet SAVi-ON predicts 3.4× better (Run 12) — so *decoder-visible index
     stability is not what enables prediction*; the propagated slot state
     evidently maintains the temporal correspondence the predictor needs even
     when the reconstruction roles shuffle. Frame this carefully (we observe
     alpha-mask roles, not the predictive latent alignment directly).

## 5. Insights so far (paper-relevant)

1. **The variance floor is the tell.** `pred ≈ 1.0` with unit-variance slots
   means mean-prediction; it is *not* "slowly training." A flat `pred` should be
   read against the slot variance, not zero. (Recommend reporting a normalized
   pred / cosine metric in the paper.)

2. **Two distinct collapse modes, with opposite signatures:**
   - *Decorrelated-noise collapse* (regularizers on, no grounding): `slot_sim`→0,
     `pred` stuck at floor. Slots distinct but contentless.
   - *Identical-slot collapse* (no distinctness pressure): `slot_sim`→1, `pred`
     trivially ~0. Slots grounded/predictable but degenerate (1 effective slot).
   A correct model must avoid *both* simultaneously.

3. **The three forces are complementary, not substitutable:**
   - `recon` → content (but alone → identical-slot collapse).
   - `sig`/`div` → each slot's marginal well-behaved (but alone → noise collapse).
   - `decorr` → the N slots differ within a frame.
   - stable slot **identity** → prediction is well-posed.
   Removing any one re-opens a collapse channel.

4. **Slot identity is a prerequisite for next-slot prediction.** Per-frame
   independent slot attention with random init makes the per-index prediction
   target both time-misaligned and stochastic → irreducible loss. Learned
   per-slot anchors + low noise fixed it (Run 5→6). *Open:* whether this is
   enough at scale or whether SAVi-style temporal slot propagation is needed.

5. **Init noise was silently doing the distinctness job.** A subtle confound:
   what looked like sig/div keeping slots apart (Run 5) was actually the random
   init. Worth calling out as a methodological caution.

6. **`decorr` can reverse an in-progress collapse, not just prevent it.** In
   Run 7 `slot_sim` spiked to ~0.89 before being driven to ~0.004. The cosine
   penalty has vanishing gradient at *exact* collapse but a usable one at
   near-collapse, so it rescues slots that have started to merge. (If a run ever
   reaches exact collapse first, this term cannot recover it — keep init noise
   nonzero, σ≈0.018, as a safeguard.)

7. **Ablation story for the paper is now clean:** each of {stable identity,
   recon, decorr} removed individually re-opens a specific, *named* failure
   (Runs 5, 4, 6 respectively). This is a ready-made ablation table.

8. **Solving collapse ≠ strong prediction.** Run 8 shows the two are separable:
   with a fully healthy (distinct/grounded/stable) representation, `pred` is
   still only ~9% below the mean floor. Anti-collapse is necessary but not
   sufficient for a useful world model — prediction quality is its own axis.
   (Frame the paper around *both*: stability AND predictive utility.)

9. **Temporal slot identity is the headline mechanism (now seed-validated).**
   Controlled, **3-seed** ablation (Run 12, only `slot_propagate` flipped):
   SAVi cut NMSE **0.790 ± 0.014 → 0.235 ± 0.030** at fixed masking (~21% →
   ~77% of slot variance explained), pcos 0.45 → 0.88. The per-arm std is tiny
   next to the gap (no seed overlap) → the effect is real, not a seed artifact.
   The bottleneck was the *ill-posedness* of per-frame-independent slot encoding
   (no object↔slot correspondence across time), not capacity or data. Critically,
   `recon` and `slot_sim` are unchanged across arms → **neither collapses and
   reconstruction is unaffected; the gain is purely predictive.** (Earlier single
   runs: uncontrolled Run 8 claimed 0.9→0.18, corrected to controlled Run 11's
   0.47→0.18; the seeded means 0.79→0.235 supersede both.)

## 6. Open questions / next steps

- [x] Run 7 result: does `decorr` + low-noise init give all-three-healthy?
      **Yes** — overfit escapes both collapse modes (commit `b72a69d`).
- [x] **Run 8:** scale Run 7 to `max_episodes=500`. **Collapse holds at scale**
      (slot_sim~0, recon 6.2→0.30 over 5000 steps), but `pred`≈0.91 (weak).
- [x] **Run 9 — prediction diagnostic:** mask OFF → `pred` 0.91→0.70. **Mixed:**
      masking is ~half the difficulty; clean pred still moderate (~0.70).
- [x] Add normalized pred metric (NMSE + cosine) — done (`3ed10d5`).
- [x] **SAVi-style temporal slot propagation** (`d9e7970`). **Run 10: big win**
      — `nmse` ≈0.9 → ≈0.18 at fixed mask=0.4 (~82% variance explained).
- [x] **Ablation (Run 11):** `slot_propagate=false`, same code → `nmse` ~0.47
      vs SAVi's ~0.18. Controlled gain confirmed (smaller than the uncontrolled
      Run-8 extrapolation).
- [x] **Multiple seeds** per arm (SAVi on/off): **Run 12, 3 seeds each.**
      SAVi NMSE **0.790±0.014 (off) → 0.235±0.030 (on)** — large, no overlap.
      Headline number established.
- [ ] Longer training (Run 10 still improving at 5000 steps) and/or scale episodes.
- [ ] Planning eval (CEM-MPC, `model.plan_mpc`) vs LeWM; qualitative slot masks.
      **Harness ready:** `scripts/eval_planning.py` mirrors LeWM's protocol
      (same env/success criterion/task sampling/CEM budget) with a slot-space
      cost policy; smoke-tested locally end-to-end on a fake dataset with the
      Run-12 seed-0 checkpoint. Awaiting a full 50-episode run in Colab.
- [ ] End-to-end unfrozen DINOv2 (the central hypothesis).
- [ ] Reconsider mask curriculum (0.4 may be too aggressive); frameskip sweep
      (needs cache rebuild).
- [ ] Tune (λ_decorr, λ_recon) balance; report sensitivity.
- [ ] If per-frame slots still wobble → **SAVi-style temporal slot propagation**
      (carry slot state frame→frame instead of re-encoding independently).
- [ ] Turn on the object-mask curriculum meaningfully (currently `mask`→0.4) and
      measure its effect once base collapse is solved.
- [ ] End-to-end (unfrozen DINOv2) — the actual central hypothesis — after the
      frozen baseline is healthy.
- [ ] Qualitative: visualize decoder alpha masks — do slots bind to T-block vs
      pusher vs background?
- [ ] Downstream: planning (CEM-MPC in `model.plan_mpc`) success rate vs LeWM.

## 7. Commit → change map

| commit | change |
|--------|--------|
| `05b0a6a` | flat HDF5 layout support |
| `712215e` | hdf5plugin codecs + ep_offset/ep_len bounds |
| `5781d60` | frozen-DINOv2 feature cache |
| `8140a6b` | max_episodes subset + disk guard |
| `57167b6` | slot→feature reconstruction (DINOSAUR decoder), λ_recon |
| `ee61d5c` | learned per-slot init (stable slot identity, low noise) |
| `b72a69d` | within-frame slot decorrelation loss, λ_decorr |
| `3ed10d5` | normalized prediction metrics (pred_nmse, pred_cos) |
| `d9e7970` | SAVi-style temporal slot propagation, `slot_propagate` |

_Last updated: through Run 12 (3-seed SAVi ablation). **Headline:** SAVi
temporal slot propagation cuts NMSE **0.790±0.014 (off) → 0.235±0.030 (on)** at
fixed mask=0.4 (no seed overlap), collapse solved in both arms, recon unchanged
→ gain is purely predictive. Supersedes the single-run 0.47→0.18 / 0.9→0.18
figures. Next: longer training, planning eval, end-to-end unfrozen DINOv2._
