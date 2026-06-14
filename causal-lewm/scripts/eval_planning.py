"""Planning evaluation: goal-conditioned CEM-MPC success rate on Push-T.

Mirrors LeWM's eval protocol (le-wm/eval.py + config/eval/pusht.yaml) so the
success-rate numbers are directly comparable:
  - same env (swm/PushT-v1) and success criterion: block pose within 20 px and
    pi/9 rad of the goal pose at any step within the budget,
  - same task sampling: num_eval (episode, start) pairs drawn from the expert
    dataset (uniform over valid starts, so episodes are weighted by length),
    goal = the dataset state goal_offset steps later, eval_budget env steps,
  - same CEM budget by default: 300 samples, 30 iterations, top-30 elites,
    horizon 5 model steps x action_block 5 env steps = 25-step plans,
    replanned every receding model steps.

Differences from LeWM (by design, matching how Causal-LeWM was trained):
  - Cost is slot-space MSE to the encoded goal (model.plan_mpc), not
    CLS-embedding MSE.
  - One model step = ONE 2-D action repeated action_block times in the env
    (the model is conditioned on the single action at each frameskip-5 frame);
    LeWM plans action_block independent env actions per model step.
  - History frames are taken at stride history_stride (= training frameskip 5).

Usage (Colab, with pusht_expert_train.h5 on disk):
  pip install stable-worldmodel
  python scripts/eval_planning.py --ckpt /content/final-2.pt \
      --data /content/pusht_expert_train.h5 --policy ours --num-eval 50
  python scripts/eval_planning.py \
      --data /content/pusht_expert_train.h5 --policy random --num-eval 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# `python scripts/eval_planning.py` puts scripts/ (not the repo root) on
# sys.path; add the root so `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.policy import BasePolicy, RandomPolicy  # noqa: E402

# swm re-exports HDF5Dataset only when its optional dep `hdf5plugin` is
# installed (a bare `except ImportError: pass` hides the failure otherwise).
if not hasattr(swm.data, "HDF5Dataset"):
    sys.exit(
        "stable_worldmodel.data.HDF5Dataset is unavailable — the optional "
        "dependency 'hdf5plugin' is missing.\nFix:  pip install hdf5plugin"
    )


def load_model(ckpt_path: Path, device: str):
    from omegaconf import OmegaConf

    from src.model import CausalLeWM, CausalLeWMConfig

    ckpt = torch.load(ckpt_path, map_location=device)
    # Older checkpoints carry unresolved Hydra interpolations; re-resolve
    # against the full saved cfg (same fix as scripts/visualize_slots.py).
    full = OmegaConf.create(ckpt["cfg"])
    cfg = CausalLeWMConfig(**OmegaConf.to_container(full.model, resolve=True))
    model = CausalLeWM(cfg).to(device)
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    bad = [k for k in missing if not k.startswith("encoder.backbone")]
    if bad:
        print(f"WARNING: missing non-backbone keys: {bad[:8]} ...")
    model.eval()
    print(
        f"loaded {Path(ckpt_path).name} @ step {ckpt.get('step', '?')} "
        f"(slot_propagate={cfg.slot_propagate}, num_slots={cfg.num_slots})"
    )
    return model


class SlotMPCPolicy(BasePolicy):
    """Receding-horizon MPC over CausalLeWM.plan_mpc (slot-space CEM)."""

    def __init__(
        self,
        model,
        device: str,
        horizon: int = 5,
        receding: int = 5,
        action_block: int = 5,
        history_frames: int = 4,
        history_stride: int = 5,
        n_samples: int = 300,
        n_iters: int = 30,
        elite_frac: float = 0.1,
        env_chunk: int = 8,
    ):
        super().__init__()
        self.type = "world_model"
        self.model = model
        self.device = device
        self.horizon = horizon
        self.receding = receding
        self.action_block = action_block
        self.history_frames = history_frames
        self.history_stride = history_stride
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.elite_frac = elite_frac
        self.env_chunk = env_chunk
        # The policy keeps its own rolling frame history (one frame arrives per
        # env step) so we don't depend on env-side frame stacking, which varies
        # across stable-worldmodel versions.
        self.hist_window = (history_frames - 1) * history_stride + 1
        self._buf: list[deque] | None = None
        self._fhist: list[deque] | None = None
        self._step = 0

    def set_env(self, env) -> None:
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._buf = [deque(maxlen=self.receding * self.action_block) for _ in range(n)]
        self._fhist = [deque(maxlen=self.hist_window) for _ in range(n)]
        self._step = 0

    def _to_frames(self, pix: np.ndarray) -> torch.Tensor:
        """uint8 (b, T, H, W, C) -> float (b, T, C, h, h) in [0,1] at model size."""
        x = torch.from_numpy(np.ascontiguousarray(pix)).float().div_(255.0)
        x = x.permute(0, 1, 4, 2, 3)
        size = self.model.cfg.image_size
        if x.shape[-1] != size or x.shape[-2] != size:
            b, t = x.shape[:2]
            x = torch.nn.functional.interpolate(
                x.flatten(0, 1), size=(size, size), mode="bilinear", align_corners=False
            ).reshape(b, t, 3, size, size)
        return x

    def _history(self, idx: list[int]) -> torch.Tensor:
        # last history_frames frames at history_stride, oldest first; clamp to 0
        # (early in an episode the rolling history is short -> repeat oldest).
        windows = []
        for i in idx:
            d = self._fhist[i]
            sel = [max(0, len(d) - 1 - k * self.history_stride) for k in range(self.history_frames)][::-1]
            windows.append(np.stack([d[s] for s in sel]))
        return self._to_frames(np.stack(windows))

    def _goal(self, info: dict, idx: list[int]) -> torch.Tensor:
        g = np.asarray(info["goal"])
        if g.ndim == 5:  # broadcast over the history axis -> take one
            g = g[:, -1]
        return self._to_frames(g[idx][:, None])[:, 0]

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        assert self.env is not None, "Environment not set for the policy"
        n = self.env.num_envs

        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None:
            for i in range(n):
                if needs_flush[i]:
                    self._buf[i].clear()
                    self._fhist[i].clear()

        pix = np.asarray(info_dict["pixels"])
        if pix.ndim == 5:  # (n, T, H, W, C) -> latest frame
            pix = pix[:, -1]
        for i in range(n):
            self._fhist[i].append(pix[i])

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        replan = [i for i in range(n) if len(self._buf[i]) == 0 and not dead[i]]
        if replan:
            n_chunks = (len(replan) + self.env_chunk - 1) // self.env_chunk
            print(
                f"[mpc] env-step {self._step}: replanning {len(replan)} envs in "
                f"{n_chunks} chunks (CEM {self.n_samples}x{self.n_iters}, "
                f"horizon {self.horizon}); {int(dead.sum())}/{n} envs done",
                flush=True,
            )
            t_plan = time.time()
            frames = self._history(replan)
            goals = self._goal(info_dict, replan)
            plans = []
            for c in range(0, len(replan), self.env_chunk):
                t_chunk = time.time()
                f = frames[c : c + self.env_chunk].to(self.device)
                g = goals[c : c + self.env_chunk].to(self.device)
                mu = self.model.plan_mpc(
                    f,
                    g,
                    action_dim=2,
                    horizon=self.horizon,
                    n_samples=self.n_samples,
                    n_iters=self.n_iters,
                    elite_frac=self.elite_frac,
                )  # (b, horizon, 2) in [-1, 1]
                plans.append(mu.cpu())
                print(
                    f"  chunk {c // self.env_chunk + 1}/{n_chunks} "
                    f"({f.shape[0]} envs): {time.time() - t_chunk:.1f}s",
                    flush=True,
                )
            print(f"[mpc] replan finished in {time.time() - t_plan:.0f}s", flush=True)
            plan = torch.cat(plans)[:, : self.receding]  # (b, receding, 2)
            # one model step = the same action held for action_block env steps
            plan = plan.repeat_interleave(self.action_block, dim=1)
            for row, i in enumerate(replan):
                self._buf[i].extend(plan[row])

        act = np.full((n, 2), np.nan, dtype=np.float32)
        for i in range(n):
            if not dead[i] and self._buf[i]:
                act[i] = self._buf[i].popleft().numpy()
        self._step += 1
        return act


class ExpertReplayPolicy(BasePolicy):
    """Oracle: replay the dataset's own native per-step actions.

    For env i (task = episode episodes_idx[i], start start_steps[i]) this plays
    the recorded action at global index offsets[ep]+start+t at env step t — i.e.
    the exact action sequence that produced the goal state (which is the dataset
    state goal_offset steps after the start). If this does NOT reach the goal,
    the harness/success-criterion/goal-injection is broken; if it DOES, the env
    + criterion are sound and any failure of --policy ours is the model/planner.
    One action per env step (no action_block repeat) — native cadence.
    """

    def __init__(self, dataset, episodes_idx, start_steps):
        super().__init__()
        self.type = "expert"
        self.actions = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
        offsets = np.asarray(dataset.offsets)
        lengths = np.asarray(dataset.lengths)
        self.base = np.array(
            [offsets[e] + s for e, s in zip(episodes_idx, start_steps)], dtype=np.int64
        )
        # last valid global action index within each task's episode
        self.last = np.array(
            [offsets[e] + lengths[e] - 1 for e in episodes_idx], dtype=np.int64
        )
        self._t: np.ndarray | None = None

    def set_env(self, env) -> None:
        self.env = env
        self._t = np.zeros(env.num_envs, dtype=np.int64)

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        n = self.env.num_envs
        nf = info_dict.get("_needs_flush")
        if nf is not None:
            self._t[np.asarray(nf, dtype=bool)] = 0
        term = info_dict.get("terminated")
        dead = (
            np.asarray(term, dtype=bool).ravel()[:n]
            if term is not None
            else np.zeros(n, bool)
        )

        gidx = np.minimum(self.base + self._t, self.last)
        act = self.actions[gidx].astype(np.float32)
        act[dead] = np.nan
        self._t[~dead] += 1
        return act


def sample_tasks(dataset, num_eval: int, goal_offset: int, seed: int):
    """(episode, start) pairs uniform over all valid dataset starts."""
    lengths = np.asarray(dataset.lengths)
    counts = np.maximum(lengths - goal_offset, 0)
    ep_col = np.repeat(np.arange(len(lengths)), counts)
    start_col = np.concatenate([np.arange(c) for c in counts if c > 0])
    if len(ep_col) < num_eval:
        raise ValueError(
            f"Only {len(ep_col)} valid starts for goal_offset={goal_offset}; "
            f"need {num_eval}."
        )
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(ep_col), size=num_eval, replace=False))
    print(f"{len(ep_col)} valid starting points; sampled {num_eval} (seed={seed})")
    return ep_col[pick].tolist(), start_col[pick].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None, help="required for --policy ours")
    ap.add_argument("--data", required=True, help="path to pusht_expert_train.h5")
    ap.add_argument("--policy", choices=["ours", "random", "expert"], default="ours")
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--receding", type=int, default=5)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--history-frames", type=int, default=4)
    ap.add_argument("--history-stride", type=int, default=5)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--elite-frac", type=float, default=0.1)
    ap.add_argument("--env-chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("planning_results.jsonl"))
    args = ap.parse_args()

    if args.policy == "ours":
        assert args.horizon * args.action_block <= args.budget, (
            "plan length (horizon * action_block) must fit within eval budget"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
    elif args.policy == "ours":
        print(
            "\n" + "=" * 70 + "\n"
            "WARNING: no CUDA device — running CEM-MPC on CPU.\n"
            "Each chunk re-runs the DINOv2 encoder + a 300x30 CEM search; on CPU\n"
            "this is ~100x slower (tens of minutes PER CHUNK) and the eval will\n"
            "effectively never finish. In Colab: Runtime > Change runtime type >\n"
            "GPU (T4), then rerun. Continuing on CPU anyway...\n" + "=" * 70 + "\n",
            flush=True,
        )

    # Accept the compressed download directly (.h5.zst), or an .h5 path whose
    # .zst sibling exists — same auto-decompress behavior as src.data.PushTHDF5.
    from src.data import _decompress_zst

    data_path = Path(args.data)
    if data_path.suffix == ".zst":
        data_path = _decompress_zst(data_path)
    elif not data_path.exists() and data_path.with_suffix(data_path.suffix + ".zst").exists():
        data_path = _decompress_zst(data_path.with_suffix(data_path.suffix + ".zst"))

    dataset = swm.data.HDF5Dataset(path=str(data_path))
    episodes_idx, start_steps = sample_tasks(
        dataset, args.num_eval, args.goal_offset, args.seed
    )

    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=args.num_eval,
        max_episode_steps=2 * args.budget,
        image_shape=(224, 224),
    )

    if args.policy == "expert":
        policy = ExpertReplayPolicy(dataset, episodes_idx, start_steps)
    elif args.policy == "ours":
        assert args.ckpt is not None, "--ckpt is required for --policy ours"
        model = load_model(args.ckpt, device)
        policy = SlotMPCPolicy(
            model,
            device,
            horizon=args.horizon,
            receding=args.receding,
            action_block=args.action_block,
            history_frames=args.history_frames,
            history_stride=args.history_stride,
            n_samples=args.samples,
            n_iters=args.iters,
            elite_frac=args.elite_frac,
            env_chunk=args.env_chunk,
        )
    else:
        policy = RandomPolicy()

    world.set_policy(policy)

    # Same state/goal injection as LeWM's config/eval/pusht.yaml callables.
    callables = [
        {"method": "_set_state", "args": {"state": {"value": "state"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
    ]

    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=[int(e) for e in episodes_idx],
        start_steps=[int(s) for s in start_steps],
        goal_offset=args.goal_offset,
        eval_budget=args.budget,
        callables=callables,
        video=str(args.video) if args.video else None,
    )
    dt = time.time() - t0

    print(f"success_rate: {metrics['success_rate']:.1f}%  ({dt:.0f}s)")

    record = {
        "policy": args.policy,
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "num_eval": args.num_eval,
        "goal_offset": args.goal_offset,
        "budget": args.budget,
        "horizon": args.horizon,
        "action_block": args.action_block,
        "samples": args.samples,
        "iters": args.iters,
        "seed": args.seed,
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": [bool(s) for s in metrics["episode_successes"]],
        "eval_seconds": round(dt, 1),
    }
    print("SUMMARY " + json.dumps({k: v for k, v in record.items() if k != "episode_successes"}))
    with args.out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
