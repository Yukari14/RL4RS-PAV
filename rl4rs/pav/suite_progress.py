"""Clean live progress for multi-prover PAV suite training (watch / tail friendly)."""
import json
import os
import time
from datetime import datetime

_STATE = {
    "live_path": None,
    "current_path": None,
    "meta_path": None,
    "suite_start": None,
    "variant_start": None,
    "index": 0,
    "total": 0,
    "suffix": "",
    "prover_kind": "",
    "stage": "init",
    "status": "running",
    "extra": {},
}


def default_paths(output_dir):
    pav_dir = os.path.join(output_dir, "pav")
    return {
        "live": os.path.join(pav_dir, "progress_live_prover_suites.txt"),
        "current": os.path.join(pav_dir, "progress_current_prover_suites.txt"),
        "meta": os.path.join(pav_dir, "progress_meta_prover_suites.json"),
    }


def configure(output_dir, index, total, suffix, prover_kind, paths=None):
    paths = paths or default_paths(output_dir)
    _STATE["live_path"] = paths["live"]
    _STATE["current_path"] = paths["current"]
    _STATE["meta_path"] = paths["meta"]
    _STATE["index"] = int(index)
    _STATE["total"] = int(total)
    _STATE["suffix"] = str(suffix)
    _STATE["prover_kind"] = str(prover_kind)
    _STATE["variant_start"] = time.time()
    _STATE["stage"] = "init"
    _STATE["status"] = "running"
    _STATE["extra"] = {}
    if _STATE["suite_start"] is None:
        _STATE["suite_start"] = _STATE["variant_start"]
    for key in ("live", "current", "meta"):
        os.makedirs(os.path.dirname(paths[key]) or ".", exist_ok=True)
    note("start")


def _format_line(stage=None, status=None, **extra):
    now = time.time()
    variant_elapsed = now - (_STATE.get("variant_start") or now)
    suite_elapsed = now - (_STATE.get("suite_start") or now)
    idx = _STATE.get("index") or 0
    total = _STATE.get("total") or 0
    stage = stage or _STATE.get("stage") or "?"
    status = status or _STATE.get("status") or "running"
    merged = dict(_STATE.get("extra") or {})
    merged.update(extra)
    parts = [
        "[{}/{}]".format(idx, total),
        "suffix={}".format(_STATE.get("suffix") or "?"),
        "prover_kind={}".format(_STATE.get("prover_kind") or "?"),
        "stage={}".format(stage),
        "status={}".format(status),
        "variant_elapsed={:.0f}s".format(variant_elapsed),
        "suite_elapsed={:.0f}s".format(suite_elapsed),
    ]
    for key, val in merged.items():
        if val is not None:
            parts.append("{}={}".format(key, val))
    return " ".join(parts), variant_elapsed, suite_elapsed, stage, status, merged


def note(stage, status="running", append_history=True, **extra):
    if not _STATE.get("meta_path"):
        return
    _STATE["stage"] = stage
    _STATE["status"] = status
    if extra:
        merged = dict(_STATE.get("extra") or {})
        merged.update(extra)
        _STATE["extra"] = merged
    line, variant_elapsed, suite_elapsed, stage, status, merged = _format_line()
    if append_history and _STATE.get("live_path"):
        with open(_STATE["live_path"], "a") as f:
            f.write(line + "\n")
    _write_current(line)
    meta = {
        "updated_at": datetime.now().isoformat(),
        "index": _STATE.get("index"),
        "total": _STATE.get("total"),
        "suffix": _STATE.get("suffix"),
        "prover_kind": _STATE.get("prover_kind"),
        "stage": stage,
        "status": status,
        "variant_elapsed_sec": round(variant_elapsed, 1),
        "suite_elapsed_sec": round(suite_elapsed, 1),
        "paths": {
            "live": _STATE.get("live_path"),
            "current": _STATE.get("current_path"),
            "meta": _STATE.get("meta_path"),
        },
    }
    meta.update(merged)
    with open(_STATE["meta_path"], "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return line


def heartbeat():
    """Refresh elapsed time without adding log noise (for long hybrid_mc)."""
    if not _STATE.get("meta_path"):
        return
    line, _, _, _, _, _ = _format_line()
    _write_current(line)
    note(_STATE.get("stage") or "?", status=_STATE.get("status") or "running", append_history=False)


def _write_current(line):
    path = _STATE.get("current_path")
    if path:
        with open(path, "w") as f:
            f.write(line + "\n")


def clear(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").close()


def reset_suite_timer():
    _STATE["suite_start"] = None


def print_watch_banner(output_dir):
    paths = default_paths(output_dir)
    lines = [
        "",
        "=" * 72,
        "PAV SUITE PROGRESS — use these instead of train_prover_suites.log",
        "=" * 72,
        "  One-line status:  watch -n 10 cat {}".format(paths["current"]),
        "  Stage history:    tail -f {}".format(paths["live"]),
        "  JSON detail:      cat {}".format(paths["meta"]),
        "  (Full TF log only for debugging: output/pav/train_prover_suites.log)",
        "=" * 72,
    ]
    for line in lines:
        print(line, flush=True)
    return paths
