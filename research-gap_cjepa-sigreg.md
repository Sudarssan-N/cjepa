# Research Gap: Causal LeWM — Combining C-JEPA with SIGReg for End-to-End Object-Centric World Models

**Date:** 2026-05-04
**Status:** Gap confirmed — unexplored

---

## 1. Key Papers

### C-JEPA: Causal-JEPA — Learning World Models through Object-Level Latent Interventions

- **Paper:** [arXiv:2602.11389](https://arxiv.org/abs/2602.11389) (Feb 2026)
- **Authors:** Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, Randall Balestriero
- **Code:** [github.com/galilai-group/cjepa](https://github.com/galilai-group/cjepa)
- **Core idea:** Extends masked JEPA from image patches to object-centric (slot) representations. Object-level masking forces the predictor to infer an object's state from other objects, creating latent interventions with counterfactual-like effects and preventing shortcut solutions.
- **Key results:**
  - ~20% absolute improvement in counterfactual reasoning on CLEVRER vs. same architecture without object-level masking (68.81% vs. 47.68%)
  - Uses only 1% of latent tokens compared to patch-based world models (e.g., DINO-WM) while achieving comparable planning performance
  - 8× faster MPC planning than DINO-WM (673s vs. 5,763s for 50 trajectories on Push-T)
- **Critical limitation:** Uses a **frozen** object-centric encoder (VideoSAUR with frozen DINOv2 ViT-S/14 backbone). Performance is ceiling-limited by encoder quality.
- **Stated future work:** *"Jointly refining object-centric encoders using strong pretrained backbones without representational collapse."*

### LeWorldModel (LeWM): Stable End-to-End JEPA from Pixels

- **Paper:** [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) (Mar 2026)
- **Authors:** Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero
- **Code:** [github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- **Core idea:** First JEPA that trains stably end-to-end from raw pixels using only two loss terms: next-embedding prediction loss + SIGReg (Sketched Isotropic Gaussian Regularization). No EMA, no stop-gradient, no pretrained encoder, no auxiliary supervision.
- **Key results:**
  - 15M parameters, trains on a single GPU in hours
  - 48× faster planning than foundation-model-based world models
  - 18% higher success rate than PLDM on Push-T
  - Emergent physical understanding: detects physically implausible events, latent trajectories straighten over training
  - Reduces tunable hyperparameters from 6 to 1 (λ, the SIGReg weight)
- **Critical limitation:** Operates on **patch-level ViT tokens** (single [CLS] embedding), not object-centric slot representations. No explicit object-level structure or interaction modeling.

### Shared Context

Both papers come from the same research group (LeCun/Balestriero at Meta AI / NYU / Mila) and share co-authors (Maes, Le Lidec). They were developed concurrently — neither cites the other.

---

## 2. The Gap

**No paper combines C-JEPA's object-level causal masking with LeWM's SIGReg for end-to-end training of an object-centric world model.**

### Search Methodology

Extensive searches conducted across:
- **arXiv** — Queries for "causal JEPA" + "SIGReg", "object-centric JEPA" + "end-to-end", "slot attention" + "SIGReg" — all returned zero results
- **Google Scholar** — Queries for C-JEPA + SIGReg / LeWorldModel combination, filtered from 2024 — zero matching papers
- **Semantic Scholar API** — Searches for C-JEPA, SIGReg regularization, and combination terms — no intersection found
- **Citation search** — No papers citing C-JEPA (2602.11389) reference SIGReg or LeWM

### Closest Existing Work (Not the Same)

| Paper | What it does | Why it's different |
|---|---|---|
| **HCLSM** (Jaber & Jaber, 2026) | Object-centric + end-to-end + JEPA prediction loss | Does NOT use SIGReg or C-JEPA's object-level masking for causal interventions |
| **ReCoRe** (Poudel et al., CVPR 2024) | Intervention-invariant regularization for world models | Pre-dates both SIGReg and C-JEPA; different regularization mechanism |
| **VJEPA** (Huang, 2026) | Variational JEPA | References C-JEPA but doesn't address end-to-end training or SIGReg |
| **DSeq-JEPA** (He et al., 2025) | Discriminative sequential JEPA | Uses causal masking but doesn't address object-centric structure or end-to-end training |

---

## 3. Refined Problem Statement

> **Causal Object-Centric World Models via End-to-End JEPA with Structured Regularization**
>
> Current object-centric world models that induce causal structure through latent interventions (e.g., C-JEPA) rely on frozen pretrained encoders, creating a performance ceiling determined by encoder quality and preventing the encoder from adapting to task-relevant object dynamics. Meanwhile, end-to-end JEPA training methods (e.g., LeWM/SIGReg) have been demonstrated only on unstructured patch tokens, leaving object-centric inductive biases unexploited.
>
> This work investigates whether C-JEPA's object-level masking — which induces latent interventions and yields ~20% absolute improvement in counterfactual reasoning — can be combined with SIGReg-style regularization to train the full object-centric encoder + predictor stack end-to-end from pixels, without representational collapse.
>
> **Central hypothesis:** A suitably adapted regularization scheme can simultaneously prevent JEPA representational collapse and slot attention collapse, enabling the encoder to learn object representations that are both causally structured (via object-level masking) and task-adapted (via end-to-end gradient flow).

---

## 4. Technical Approach

### 4.1 The Dual Collapse Problem

Training end-to-end introduces two simultaneous collapse risks:

| Collapse Type | Mechanism | Current Prevention |
|---|---|---|
| **JEPA collapse** | Encoder maps all inputs to constant/trivial representations | SIGReg — forces isotropic Gaussian structure on latent distribution |
| **Slot collapse** | All slot attention heads bind to the same object/region | Slot competition (softmax over slots) — but this can fail under joint optimization pressure |

When the JEPA prediction loss and slot attention are trained jointly, a perverse incentive emerges: the encoder can make prediction trivially easy by collapsing all slots to identical representations, since predicting identical values requires zero capacity. SIGReg alone may not prevent this — it enforces a Gaussian *distribution*, not slot *distinctness*.

### 4.2 Proposed Solution: Slot-Diversity Regularization

Add an explicit inter-slot diversity term to the training objective:

```
ℒ_total = ℒ_pred + λ₁ · SIGReg(Z) + λ₂ · ℒ_diversity(S)
```

Where `ℒ_diversity` could take one of several forms:

- **Minimum pairwise distance:** `ℒ_div = -min_{i≠j} ||s_i - s_j||₂` — penalizes the closest pair of slots
- **Contrastive (InfoNCE):** Treat each slot as a separate class, maximize agreement between a slot and its temporally adjacent counterpart while repelling other slots
- **VICReg-style variance term:** Apply variance regularization *per slot index* across the batch, encouraging each slot position to encode different information
- **Orthogonality constraint:** `ℒ_div = ||SS^T - I||_F` where S is the normalized slot matrix — encourages slots to span orthogonal directions

**Recommendation:** Start with the VICReg-style per-slot variance term since it integrates naturally with the batch-level statistics SIGReg already computes.

### 4.3 Adapting SIGReg for Slot-Structured Representations

SIGReg was designed for a single global embedding (the ViT [CLS] token). Slot attention produces an **N × d** matrix per frame. Three options:

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A: Per-slot SIGReg** | Apply SIGReg independently to each slot's distribution across the batch | Simple, minimal change | Doesn't model inter-slot relationships |
| **B: Joint-flattened SIGReg** | Concatenate/flatten all slots into one distribution, match to Gaussian mixture | Captures joint structure | More complex target distribution |
| **C: Slot-conditional SIGReg** | Apply SIGReg to slot embeddings conditioned on slot index | Maintains slot identity | Assumes fixed slot roles (may not hold) |

**Recommendation:** Start with **Option A** (independent SIGReg per slot, averaged) combined with the diversity term from §4.2. This keeps the hyperparameter count low while explicitly handling both collapse modes.

### 4.4 Gradient Isolation: Scheduled Masking

In C-JEPA, the frozen encoder means masking doesn't affect encoder learning. With a trainable encoder, the masking pattern influences gradient flow through the backbone.

**Proposed curriculum:**
1. **Phase 1 (warmup):** No object-level masking — train encoder + predictor with only future prediction + SIGReg + diversity loss. Encoder learns stable slot decomposition.
2. **Phase 2 (introduce masking):** Gradually increase masking ratio from 0 to target (e.g., 25-50% of objects). Encoder adapts to produce representations robust to partial observability.
3. **Phase 3 (full training):** Full C-JEPA objective with object-level masking on history + future tokens.

### 4.5 Architecture Blueprint

```
Raw Pixels (X_t)
     │
     ▼
┌─────────────────────────┐
│  Trainable ViT Backbone  │  ← initialized from DINOv2 weights (not frozen)
│  (patch embeddings)      │     ~5M params (ViT-Tiny from LeWM)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Slot Attention          │  ← trainable, iteratively groups patches
│  (N slots × d dim)       │     into N object-centric slot vectors
└───────────┬─────────────┘
            │
     ┌──────┴──────────────┐
     │                     │
     ▼                     ▼
┌──────────────┐   ┌──────────────┐
│ Object-Level │   │   SIGReg     │  ← prevents JEPA collapse
│   Masking    │   │   + diversity│  ← prevents slot collapse
│ (history +   │   │   losses     │
│  future)     │   │              │
└──────┬───────┘   └──────────────┘
       │
       ▼
┌─────────────────────────┐
│  Masked Transformer      │  ← bidirectional attention over slots
│  Predictor (~10M params) │     predicts masked + future slot states
└───────────┬─────────────┘
            │
            ▼
       ℒ_pred (mask reconstruction + future prediction)
```

### 4.6 Loss Function Summary

```
ℒ_total = ℒ_mask + λ_sig · SIGReg(S) + λ_div · ℒ_diversity(S)

where:
  ℒ_mask = 𝔼[ Σ_τ Σ_i 𝟙[masked] ||ẑ_τ^i - z_τ^i||₂² ]
    - History term: reconstruct masked object slots from visible ones
    - Future term: predict future slot states (always masked)

  SIGReg(S) = (1/N) Σ_i SIGReg(s^i)     ← per-slot Gaussian matching
    SIGReg(s^i) = (1/M) Σ_m T(proj_m(s^i))

  ℒ_diversity(S) = Σ_i Var_batch(s^i)    ← per-slot variance across batch
    (or contrastive/orthogonality variant)
```

---

## 5. Evaluation Protocol

### 5.1 Primary Benchmarks

| Benchmark | What it tests | C-JEPA baseline (frozen) | Hypothesis |
|---|---|---|---|
| **CLEVRER** | Counterfactual VQA | 68.81% counterfactual per-question | End-to-end training should exceed frozen encoder by learning task-adapted object features |
| **Push-T** | 2D manipulation planning | 88.67% success, 673s/50 trajectories | Comparable or better planning with richer object representations |

### 5.2 Ablation Studies

1. **Encoder quality dependence:** Train with random init vs. ImageNet vs. DINOv2 initialization → measure how end-to-end training reduces dependence on pretrained encoder quality
2. **λ_div sweep:** Vary diversity loss weight → find the minimum needed to prevent slot collapse
3. **Masking curriculum:** Compare scheduled masking vs. full masking from start → quantify training stability benefit
4. **SIGReg variant:** Compare per-slot SIGReg vs. joint-flattened SIGReg → determine which slot regularization strategy works best
5. **Diversity loss variant:** Compare variance-based vs. contrastive vs. orthogonality diversity losses

### 5.3 Collapse Detection Metrics

- **Slot uniqueness score:** Mean pairwise cosine similarity between slots within a frame (low = healthy, high → 1.0 = collapsed)
- **Slot utilization:** Entropy of slot-to-object assignments across frames
- **SIGReg statistic value:** Tracks distribution matching quality during training
- **Prediction loss on held-out interactions:** Should remain high if slots are distinct and informative

---

## 6. Key Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Slot collapse dominates early training | High | Phase 1 warmup without masking; freeze slot attention for first N steps |
| SIGReg Gaussian target incompatible with slot structure | Medium | Test both per-slot and joint-flattened variants; fall back to VICReg-style regularization if SIGReg fails |
| Computational cost: ViT + Slot Attention + Predictor trained jointly | Medium | Use ViT-Tiny (5M), N≤7 slots, single GPU; matches LeWM's demonstrated feasibility |
| Masking gradient noise destabilizes slot attention iterations | Medium | Gradient clipping on slot attention; reduce slot attention iterations during early training |
| End-to-end doesn't outperform frozen encoder | Low | Even matching C-JEPA performance with a smaller/no pretrained encoder would be a meaningful result (reduced dependency) |

---

## 7. Implementation Path

### Phase 1: Minimal Prototype (Push-T)
- Fork LeWM codebase (`github.com/lucas-maes/le-wm`)
- Add Slot Attention module after ViT encoder (replace [CLS] token extraction)
- Implement object-level masking in the predictor
- Add per-slot SIGReg + diversity loss
- Train and evaluate on Push-T with frozen encoder baseline

### Phase 2: Full Training + Ablations
- Unfreeze encoder, implement scheduled masking
- Run ablation studies (§5.2)
- Profile collapse metrics (§5.3)

### Phase 3: CLEVRER + Counterfactual Evaluation
- Port to CLEVRER with ALOE VQA evaluation
- Compare counterfactual reasoning against frozen C-JEPA
- Analyze influence neighborhoods qualitatively

### Phase 4: Extensions (if successful)
- Scale to richer environments (3D, more objects)
- Explore hierarchical slot decomposition
- Investigate whether influence neighborhoods align with ground-truth causal graphs

---

## 8. Relevant Codebases

| Codebase | URL | Purpose |
|---|---|---|
| C-JEPA | `github.com/galilai-group/cjepa` | Reference for object-level masking, predictor, ALOE evaluation |
| LeWM | `github.com/lucas-maes/le-wm` | Starting point — end-to-end training loop, SIGReg, MPC planning |
| VideoSAUR | (within C-JEPA repo) | Slot attention encoder reference implementation |

---

## 9. References

1. Nam et al., "Causal-JEPA: Learning World Models through Object-Level Latent Interventions," arXiv:2602.11389, Feb 2026.
2. Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels," arXiv:2603.19312, Mar 2026.
3. Poudel et al., "ReCoRe: Regularized Contrastive Representation Learning of World Model," CVPR 2024.
4. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture," CVPR 2023. (I-JEPA)
5. Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning," ICLR 2022.
6. Jaber & Jaber, "HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling," arXiv 2026.
