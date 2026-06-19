"""Live training progress files for DQN/PPO pilots (tail -f friendly)."""
import csv
import json
import os
import time
from datetime import datetime


def _fmt_float(value, digits=4):
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN
            return "n/a"
    except TypeError:
        return "n/a"
    return "{:.{}f}".format(float(value), digits)


class LiveProgressBoard(object):
    """Writes auto-updated progress files each training epoch."""

    def __init__(self, pilot_dir, cond, seed, total_epochs, use_pav=False):
        os.makedirs(pilot_dir, exist_ok=True)
        self.cond = cond
        self.seed = seed
        self.total_epochs = int(total_epochs)
        self.use_pav = bool(use_pav)
        self._start = time.time()
        self.paths = {
            "live": os.path.join(pilot_dir, "progress_live_{}_seed{}.txt".format(cond, seed)),
            "curve_csv": os.path.join(pilot_dir, "reward_curve_{}_seed{}.csv".format(cond, seed)),
            "meta_json": os.path.join(pilot_dir, "progress_meta_{}_seed{}.json".format(cond, seed)),
        }
        if use_pav:
            self.paths["pav_monitor_csv"] = os.path.join(
                pilot_dir, "pav_monitor_{}_seed{}.csv".format(cond, seed)
            )
            self.paths["pav_state_json"] = os.path.join(
                pilot_dir, "pav_state_{}_seed{}.json".format(cond, seed)
            )

    def print_watch_banner(self):
        lines = [
            "",
            "=" * 72,
            "LIVE PROGRESS — files update automatically each epoch",
            "=" * 72,
            "  Human-readable:  tail -f {}".format(self.paths["live"]),
            "  Reward curve:    tail -f {}".format(self.paths["curve_csv"]),
            "  Machine-readable: cat {}".format(self.paths["meta_json"]),
        ]
        if self.use_pav:
            lines.extend([
                "  PAV monitor CSV: tail -f {}".format(self.paths["pav_monitor_csv"]),
                "  PAV state JSON:  cat {}".format(self.paths["pav_state_json"]),
            ])
        lines.append("=" * 72)
        for line in lines:
            print(line, flush=True)

    def read_pav_state(self):
        path = self.paths.get("pav_state_json")
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return {}

    def save_curve_csv(self, rows):
        if not rows:
            return
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with open(self.paths["curve_csv"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def update(self, epoch, row, pav_state=None):
        elapsed = time.time() - self._start
        per_epoch = elapsed / max(epoch, 1)
        remaining = per_epoch * max(self.total_epochs - epoch, 0)
        pav_state = pav_state or {}

        meta = {
            "updated_at": datetime.now().isoformat(),
            "condition": self.cond,
            "seed": self.seed,
            "epoch": int(epoch),
            "total_epochs": self.total_epochs,
            "progress_pct": round(100.0 * epoch / max(self.total_epochs, 1), 2),
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(remaining, 1),
            "train_episode_reward_mean": row.get("train_episode_reward_mean"),
            "sim_eval_avg_reward": row.get("sim_eval_avg_reward"),
            "timesteps_total": row.get("timesteps_total"),
            "paths": self.paths,
        }
        if self.use_pav:
            meta["pav_alpha_scale"] = pav_state.get("alpha_scale")
            meta["pav_rolling_distinguishability"] = pav_state.get("rolling_distinguishability")
            meta["pav_mean_verifier"] = pav_state.get("mean_verifier")
            meta["pav_transitions"] = pav_state.get("transitions")
            meta["prover_kind"] = pav_state.get("prover_kind")

        with open(self.paths["meta_json"], "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        parts = [
            "[{}/{} {:.1f}%]".format(epoch, self.total_epochs, meta["progress_pct"]),
            "train={}".format(_fmt_float(row.get("train_episode_reward_mean"))),
        ]
        if row.get("sim_eval_avg_reward") is not None:
            parts.append("sim_eval={}".format(_fmt_float(row.get("sim_eval_avg_reward"))))
        if self.use_pav:
            parts.append("alpha_scale={}".format(_fmt_float(pav_state.get("alpha_scale"), 3)))
            parts.append(
                "pav_dist={}".format(_fmt_float(pav_state.get("rolling_distinguishability")))
            )
            parts.append(
                "prover_kind={}".format(pav_state.get("prover_kind") or "n/a")
            )
        parts.append("elapsed={:.0f}s".format(elapsed))
        parts.append("eta={:.0f}s".format(remaining))
        line = " ".join(parts)

        with open(self.paths["live"], "a") as f:
            f.write(line + "\n")
        print("[progress] " + line, flush=True)
        return line

    def format_tqdm_postfix(self, row, pav_state=None):
        pav_state = pav_state or {}
        postfix = {
            "train": _fmt_float(row.get("train_episode_reward_mean"), 2),
        }
        if row.get("sim_eval_avg_reward") is not None:
            postfix["sim"] = _fmt_float(row.get("sim_eval_avg_reward"), 2)
        if self.use_pav:
            postfix["a_scale"] = _fmt_float(pav_state.get("alpha_scale"), 3)
            if pav_state.get("prover_kind"):
                postfix["prover"] = str(pav_state["prover_kind"])
        return postfix
