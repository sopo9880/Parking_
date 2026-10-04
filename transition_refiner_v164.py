# -*- coding: utf-8 -*-
"""Parking Research Agent v16.4 candidate transition refiner.

This module intentionally does *not* replace the v16.2 SAFE_BASELINE engine.
It post-processes the SAFE_BASELINE per-slot causal state trace and creates a
v16.4 candidate trace for comparison.

Decision rules never read ground truth. Ground truth is used only after the
candidate trace has been produced, for evaluation/reporting.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class RefinerConfig:
    entry_recent_motion_window_sec: float = 8.0
    entry_recent_motion_min_px: float = 45.0
    entry_track_age_target_sec: float = 24.0
    entry_transition_track_age_estimate_sec: float = 3.0
    entry_stable_motion_span_max_px: float = 12.0
    entry_stable_speed_max_px_s: float = 6.0
    entry_recent_hit_ratio_min: float = 0.75

    exit_weak_mean_conf_max: float = 0.20
    exit_motion_span_min_px: float = 35.0
    exit_confirm_sec: float = 6.0
    exit_reacquire_settle_sec: float = 8.0
    exit_require_duplicate_empty_vote: bool = True

    event_eval_window_sec: float = 30.0


def _required_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _last_valid_track_id(df: pd.DataFrame) -> int:
    for x in reversed(df["dominant_track_id"].fillna(-1).tolist()):
        try:
            x = int(x)
        except Exception:
            continue
        if x >= 0:
            return x
    return -1


def apply_transition_refiner(slot_df: pd.DataFrame, cfg: RefinerConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "time_sec", "local_id", "global_id", "state", "phase",
        "dominant_track_id", "track_speed_px_s", "track_motion_span_px",
        "slot_detection_motion_span_px", "recent_hit_ratio",
        "mean_detection_confidence",
    ]
    _required_columns(slot_df, required, "baseline_slot_timeseries")

    base = slot_df.copy().sort_values(["local_id", "time_sec"]).reset_index(drop=True)
    out = base.copy()
    out["refined_state"] = out["state"].astype(str)
    out["refine_reason"] = ""
    events: List[Dict] = []

    source_counts = base.groupby("global_id")["local_id"].nunique().to_dict()

    for local_id, g in base.groupby("local_id", sort=False):
        g = g.sort_values("time_sec")
        prev_state = g["state"].shift(1)
        transition_idx = g.index[(prev_state == "EMPTY") & (g["state"] == "OCCUPIED")]
        for ix in transition_idx:
            row = base.loc[ix]
            t0 = float(row["time_sec"])
            current_track = int(row["dominant_track_id"]) if pd.notna(row["dominant_track_id"]) else -1
            recent = g[(g["time_sec"] >= t0 - cfg.entry_recent_motion_window_sec) & (g["time_sec"] < t0)]
            previous_track = _last_valid_track_id(recent)
            recent_motion = max(
                float(recent["track_motion_span_px"].max()) if len(recent) else 0.0,
                float(recent["slot_detection_motion_span_px"].max()) if len(recent) else 0.0,
            )
            if previous_track < 0 or current_track < 0 or current_track == previous_track:
                continue
            if recent_motion < cfg.entry_recent_motion_min_px:
                continue

            min_release_time = t0 + max(
                0.0,
                cfg.entry_track_age_target_sec - cfg.entry_transition_track_age_estimate_sec,
            )
            future = g[g["time_sec"] >= min_release_time]
            stable = future[
                (future["track_motion_span_px"] <= cfg.entry_stable_motion_span_max_px)
                & (future["track_speed_px_s"] <= cfg.entry_stable_speed_max_px_s)
                & (future["recent_hit_ratio"] >= cfg.entry_recent_hit_ratio_min)
            ]
            release_time = float(stable.iloc[0]["time_sec"]) if len(stable) else min_release_time
            mask = (
                (out["local_id"] == local_id)
                & (out["time_sec"] >= t0)
                & (out["time_sec"] < release_time)
            )
            if mask.any():
                out.loc[mask, "refined_state"] = "EMPTY"
                out.loc[mask, "refine_reason"] = "ENTRY_TRACK_SWITCH_HOLD"
                events.append({
                    "rule": "ENTRY_TRACK_SWITCH_HOLD",
                    "local_id": local_id,
                    "global_id": row["global_id"],
                    "start_sec": t0,
                    "end_sec": release_time,
                    "previous_track_id": previous_track,
                    "current_track_id": current_track,
                    "recent_motion_px": recent_motion,
                    "baseline_mean_conf": float(row["mean_detection_confidence"]),
                })

    for local_id, g in base.groupby("local_id", sort=False):
        g = g.sort_values("time_sec")
        global_id = str(g.iloc[0]["global_id"])
        if source_counts.get(global_id, 1) < 2:
            continue
        prev_phase = g["phase"].shift(1)
        transition_idx = g.index[
            (prev_phase != "LEAVING")
            & (g["phase"] == "LEAVING")
            & (g["state"] == "OCCUPIED")
        ]
        for ix in transition_idx:
            row = base.loc[ix]
            t0 = float(row["time_sec"])
            if float(row["mean_detection_confidence"]) > cfg.exit_weak_mean_conf_max:
                continue
            if float(row["track_motion_span_px"]) < cfg.exit_motion_span_min_px:
                continue
            if cfg.exit_require_duplicate_empty_vote:
                siblings = out[
                    (out["global_id"] == global_id)
                    & (out["local_id"] != local_id)
                    & (out["time_sec"] == t0)
                ]
                if not (len(siblings) and (siblings["refined_state"] == "EMPTY").any()):
                    continue

            release_time = t0 + cfg.exit_confirm_sec
            after_release = g[g["time_sec"] >= release_time]
            baseline_reoccupy = after_release[
                (after_release["phase"] == "OCCUPIED")
                & (after_release["state"] == "OCCUPIED")
            ]
            reacquire_time: Optional[float] = None
            if len(baseline_reoccupy):
                reacquire_time = (
                    float(baseline_reoccupy.iloc[0]["time_sec"])
                    + cfg.exit_reacquire_settle_sec
                )
            end_time = (
                reacquire_time
                if reacquire_time is not None
                else float(g["time_sec"].max()) + 1.0
            )
            mask = (
                (out["local_id"] == local_id)
                & (out["time_sec"] >= release_time)
                & (out["time_sec"] < end_time)
            )
            if mask.any():
                out.loc[mask, "refined_state"] = "EMPTY"
                out.loc[mask, "refine_reason"] = "LEAVING_WEAK_OWNER_RELEASE"
                events.append({
                    "rule": "LEAVING_WEAK_OWNER_RELEASE",
                    "local_id": local_id,
                    "global_id": global_id,
                    "start_sec": release_time,
                    "end_sec": end_time,
                    "previous_track_id": "",
                    "current_track_id": int(row["dominant_track_id"]) if pd.notna(row["dominant_track_id"]) else -1,
                    "recent_motion_px": float(row["track_motion_span_px"]),
                    "baseline_mean_conf": float(row["mean_detection_confidence"]),
                })

    return out, pd.DataFrame(events)


def build_global_trace(refined_slot_df: pd.DataFrame) -> pd.DataFrame:
    _required_columns(
        refined_slot_df,
        ["time_sec", "global_id", "refined_state"],
        "refined_slot_timeseries",
    )
    glob = (
        refined_slot_df.groupby(["time_sec", "global_id"], as_index=False)
        .agg(
            occupied_votes=("refined_state", lambda s: int((s == "OCCUPIED").sum())),
            total_votes=("refined_state", "size"),
        )
    )
    glob["state"] = glob["occupied_votes"].map(
        lambda n: "OCCUPIED" if n > 0 else "EMPTY"
    )
    return glob


def build_count_trace(global_df: pd.DataFrame, baseline_count_path: Path) -> pd.DataFrame:
    pred = (
        global_df.groupby("time_sec")["state"]
        .apply(lambda s: int((s == "OCCUPIED").sum()))
        .reset_index(name="occupied_pred_candidate")
    )
    base_counts = pd.read_csv(baseline_count_path)
    result = base_counts.merge(pred, on="time_sec", how="left")
    result["candidate_error"] = (
        result["occupied_pred_candidate"]
        - result["ground_truth_occupied_space_count"]
    )
    result["candidate_abs_error"] = result["candidate_error"].abs()
    return result


def metrics_from_count_trace(count_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, d in count_df.groupby("split", dropna=False):
        rows.append({
            "split": split,
            "N": int(len(d)),
            "exact_rate": float((d["candidate_abs_error"] == 0).mean()),
            "mae": float(d["candidate_abs_error"].mean()),
            "max_abs_error": int(d["candidate_abs_error"].max()),
            "under_rate": float((d["candidate_error"] < 0).mean()),
            "over_rate": float((d["candidate_error"] > 0).mean()),
        })
    return pd.DataFrame(rows)


def event_balanced_metrics(count_df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    d = count_df.sort_values("time_sec").copy()
    gt = d["ground_truth_occupied_space_count"]
    change_times = d.loc[
        gt.ne(gt.shift(1)) & gt.shift(1).notna(),
        "time_sec",
    ].tolist()
    if not change_times:
        return pd.DataFrame(
            columns=["event_time_sec", "N", "exact_rate", "mae", "max_abs_error"]
        )
    rows = []
    half = max(1.0, float(window_sec))
    for t in change_times:
        w = d[d["time_sec"].between(float(t) - half, float(t) + half)]
        if len(w) == 0:
            continue
        rows.append({
            "event_time_sec": float(t),
            "N": int(len(w)),
            "exact_rate": float((w["candidate_abs_error"] == 0).mean()),
            "mae": float(w["candidate_abs_error"].mean()),
            "max_abs_error": int(w["candidate_abs_error"].max()),
        })
    return pd.DataFrame(rows)


def run(run_dir: Path, cfg: RefinerConfig) -> Dict:
    run_dir = run_dir.resolve()
    slot_path = run_dir / "baseline_slot_timeseries.csv"
    count_path = run_dir / "baseline_count_timeseries.csv"
    if not slot_path.is_file() or not count_path.is_file():
        raise FileNotFoundError(
            "run_dir must contain baseline_slot_timeseries.csv "
            "and baseline_count_timeseries.csv"
        )

    baseline_slots = pd.read_csv(slot_path)
    refined_slots, events = apply_transition_refiner(baseline_slots, cfg)
    globals_ = build_global_trace(refined_slots)
    counts = build_count_trace(globals_, count_path)
    metrics = metrics_from_count_trace(counts)
    event_metrics = event_balanced_metrics(counts, cfg.event_eval_window_sec)

    refined_slots.to_csv(run_dir / "candidate_v164_slot_timeseries.csv", index=False)
    globals_.to_csv(run_dir / "candidate_v164_global_slot_timeseries.csv", index=False)
    counts.to_csv(run_dir / "candidate_v164_count_timeseries.csv", index=False)
    metrics.to_csv(run_dir / "candidate_v164_metrics.csv", index=False)
    events.to_csv(run_dir / "candidate_v164_transition_events.csv", index=False)
    event_metrics.to_csv(
        run_dir / "candidate_v164_event_balanced_metrics.csv",
        index=False,
    )
    (run_dir / "candidate_v164_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    test_row = metrics[metrics["split"] == "TEST"]
    test = test_row.iloc[0].to_dict() if len(test_row) else {}
    baseline = pd.read_csv(count_path)
    baseline_test = baseline[baseline["split"] == "TEST"].copy()
    baseline_exact = (
        float((baseline_test["abs_error"] == 0).mean())
        if len(baseline_test) else None
    )
    baseline_mae = (
        float(baseline_test["abs_error"].mean())
        if len(baseline_test) else None
    )

    summary = {
        "candidate": "v16.4.0-transition-refiner",
        "baseline": "v16.2 SAFE_BASELINE",
        "baseline_test_exact_rate": baseline_exact,
        "baseline_test_mae": baseline_mae,
        "candidate_test_exact_rate": test.get("exact_rate"),
        "candidate_test_mae": test.get("mae"),
        "candidate_test_max_abs_error": test.get("max_abs_error"),
        "candidate_test_under_rate": test.get("under_rate"),
        "candidate_test_over_rate": test.get("over_rate"),
        "transition_events": int(len(events)),
        "decision": "CANDIDATE_ONLY_INDEPENDENT_VALIDATION_REQUIRED",
    }
    (run_dir / "candidate_v164_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "Parking Research Agent v16.4.0 candidate transition refiner",
        "==========================================================",
        "",
        "SAFE_BASELINE is not modified or promoted automatically.",
        "Ground truth is never read by transition decisions; it is evaluation-only.",
        "",
        f"Baseline TEST Exact: {baseline_exact:.4f}" if baseline_exact is not None else "Baseline TEST Exact: n/a",
        f"Baseline TEST MAE: {baseline_mae:.4f}" if baseline_mae is not None else "Baseline TEST MAE: n/a",
        f"Candidate TEST Exact: {test.get('exact_rate', float('nan')):.4f}",
        f"Candidate TEST MAE: {test.get('mae', float('nan')):.4f}",
        f"Candidate TEST Max Error: {test.get('max_abs_error', 'n/a')}",
        f"Candidate transition events: {len(events)}",
        "",
        "Decision: CANDIDATE ONLY - independent video validation required before SAFE_BASELINE promotion.",
    ]
    (run_dir / "CANDIDATE_v16_4_REPORT.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="v16.4 candidate transition refiner")
    ap.add_argument(
        "run_dir",
        nargs="?",
        default="output",
        help="Directory containing baseline CSV outputs",
    )
    ap.add_argument("--config", help="Optional JSON overrides for RefinerConfig")
    args = ap.parse_args()
    cfg = RefinerConfig()
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(cfg, k, v)
    summary = run(Path(args.run_dir), cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
