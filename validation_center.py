# -*- coding: utf-8 -*-
"""v16.5 validation center.

Adds:
- local Dataset Profiles
- 1..N CCTV configuration
- count-GT template + interactive 10-second labeler
- episode/cut-boundary evaluation mask
- repeated-video stability analysis
- MetaPKLot/CNRPark-EXT download + external spatial validation UI

The temporal algorithm is intentionally not tuned from external/independent data.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from localization import tr
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from utils import format_timestamp, load_json, save_json, video_info, open_video, read_frame_at
from public_dataset import prepare_metapklot_cnr, evaluate_external_cnr

APP_DIR = Path(__file__).resolve().parent
USER_DATA = APP_DIR / "user_data"
DATASET_ROOT = USER_DATA / "datasets"
PROFILE_PATH = DATASET_ROOT / "profiles.json"
PUBLIC_ROOT = DATASET_ROOT / "metapklot_cnr"


def _ensure():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def parse_time_list(text: str) -> List[float]:
    out = []
    for token in str(text or "").replace(";", ",").split(","):
        s = token.strip()
        if not s:
            continue
        parts = s.split(":")
        try:
            if len(parts) == 1:
                v = float(parts[0])
            elif len(parts) == 2:
                v = float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                v = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            else:
                raise ValueError
        except Exception:
            raise ValueError(f"Invalid cut boundary: {s}")
        if v > 0:
            out.append(float(v))
    return sorted(set(out))


def format_time_list(values) -> str:
    return ", ".join(format_timestamp(float(x)) for x in (values or []))


def configured_camera_names(settings: Dict, rois: Optional[Dict] = None) -> List[str]:
    validation = settings.get("validation", {}) or {}
    n = max(1, int(validation.get("camera_count", 3)))
    names = [f"cctv{i}" for i in range(1, n + 1)]
    for c in sorted((rois or {}).keys()):
        if c not in names:
            names.append(c)
    return names


def _episode_id(t: float, cuts: List[float]) -> int:
    return 1 + sum(1 for c in cuts if float(t) >= float(c) - 1e-9)


def _eval_valid(t: float, cuts: List[float], warmup: float) -> bool:
    starts = [0.0] + list(cuts)
    for s in starts:
        if float(s) <= float(t) < float(s) + float(warmup) - 1e-9:
            return False
    return True


def build_gt_template(video_path: str, out_path: str | Path, camera_count: int = 3,
                      interval_sec: float = 10.0, cut_boundaries=None,
                      warmup_sec: float = 10.0) -> Path:
    info = video_info(video_path)
    duration = float(info.get("duration_sec", 0.0))
    interval = max(1.0, float(interval_sec))
    cuts = sorted(float(x) for x in (cut_boundaries or []))
    times = np.arange(0.0, duration + 1e-6, interval).tolist()
    rows = []
    for t in times:
        row = {"timestamp": format_timestamp(t)}
        for i in range(1, int(camera_count) + 1):
            row[f"cctv{i}_count"] = ""
        row.update({
            "ground_truth_unique_vehicle_count": "",
            "ground_truth_occupied_space_count": "",
            "tags": "CUT" if any(abs(t-c) <= interval/2 for c in cuts) else "",
            "overlap_cctv": "",
            "notes": "",
            "episode_id": _episode_id(t, cuts),
            "eval_valid": int(_eval_valid(t, cuts, warmup_sec)),
        })
        rows.append(row)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


class CountGTLabeler(tk.Toplevel):
    def __init__(self, parent, video_path: str, csv_path: str, camera_count: int,
                 interval_sec: float, cut_boundaries=None, warmup_sec: float = 10.0):
        super().__init__(parent)
        self.title(tr("Count Ground Truth Labeler"))
        self.geometry("1280x860")
        self.minsize(980, 700)
        self.video_path = str(video_path)
        self.csv_path = Path(csv_path)
        self.camera_count = int(camera_count)
        self.interval_sec = float(interval_sec)
        self.cuts = list(cut_boundaries or [])
        self.warmup_sec = float(warmup_sec)
        if not self.csv_path.is_file():
            build_gt_template(
                self.video_path, self.csv_path, self.camera_count,
                self.interval_sec, self.cuts, self.warmup_sec
            )
        self.df = pd.read_csv(self.csv_path, encoding="utf-8-sig", dtype=str).fillna("")
        self.index = 0
        self.photo = None
        self.cap = open_video(self.video_path)

        self.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._load_row()

    def _build(self):
        top = ttk.Frame(self); top.pack(fill="x", padx=10, pady=8)
        self.pos_var = tk.StringVar()
        ttk.Label(top, textvariable=self.pos_var, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(top, text=tr("  A/D or Prev/Next | Enter=save+next | U=exclude | C=cut")).pack(side="left")

        self.image_label = ttk.Label(self)
        self.image_label.pack(fill="both", expand=True, padx=10, pady=6)

        edit = ttk.LabelFrame(self, text=tr("Ground truth"))
        edit.pack(fill="x", padx=10, pady=6)
        self.cam_vars = []
        for i in range(1, self.camera_count + 1):
            v = tk.StringVar(); self.cam_vars.append(v)
            ttk.Label(edit, text=tr(f"CCTV{i}")).grid(row=0, column=(i-1)*2, padx=(8,2), pady=6)
            ttk.Entry(edit, textvariable=v, width=7).grid(row=0, column=(i-1)*2+1, padx=(2,8), pady=6)
        row2 = ttk.Frame(edit); row2.grid(row=1, column=0, columnspan=max(2, self.camera_count*2), sticky="ew", padx=8, pady=6)
        self.total_var = tk.StringVar()
        self.unique_var = tk.StringVar()
        self.tag_var = tk.StringVar()
        self.note_var = tk.StringVar()
        ttk.Label(row2, text=tr("Occupied")).pack(side="left")
        ttk.Entry(row2, textvariable=self.total_var, width=8).pack(side="left", padx=4)
        ttk.Label(row2, text=tr("Unique")).pack(side="left", padx=(12,0))
        ttk.Entry(row2, textvariable=self.unique_var, width=8).pack(side="left", padx=4)
        ttk.Label(row2, text=tr("Tag")).pack(side="left", padx=(12,0))
        ttk.Entry(row2, textvariable=self.tag_var, width=12).pack(side="left", padx=4)
        ttk.Label(row2, text=tr("Notes")).pack(side="left", padx=(12,0))
        ttk.Entry(row2, textvariable=self.note_var).pack(side="left", padx=4, fill="x", expand=True)

        actions = ttk.Frame(self); actions.pack(fill="x", padx=10, pady=(2,10))
        ttk.Button(actions, text=tr("Prev"), command=self.prev).pack(side="left", padx=4)
        ttk.Button(actions, text=tr("Save"), command=self.save_current).pack(side="left", padx=4)
        ttk.Button(actions, text=tr("Save + Next"), command=self.next).pack(side="left", padx=4)
        ttk.Button(actions, text=tr("Mark U / Exclude"), command=self.mark_unknown).pack(side="left", padx=4)
        ttk.Button(actions, text=tr("Mark CUT"), command=self.mark_cut).pack(side="left", padx=4)
        ttk.Button(actions, text=tr("Close"), command=self._close).pack(side="right", padx=4)

        self.bind("<Left>", lambda e: self.prev())
        self.bind("<Right>", lambda e: self.next())
        self.bind("<Return>", lambda e: self.next())
        self.bind("a", lambda e: self.prev())
        self.bind("d", lambda e: self.next())
        self.bind("u", lambda e: self.mark_unknown())
        self.bind("c", lambda e: self.mark_cut())

    def _row_time(self) -> float:
        s = str(self.df.iloc[self.index]["timestamp"])
        parts = [float(x) for x in s.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]

    def _load_row(self):
        r = self.df.iloc[self.index]
        t = self._row_time()
        frame = read_frame_at(self.cap, t)
        h, w = frame.shape[:2]
        max_w, max_h = 1180, 580
        scale = min(max_w/max(1,w), max_h/max(1,h), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.image_label.configure(image=self.photo)
        self.pos_var.set(
            tr(f"{self.index+1}/{len(self.df)}  |  {r['timestamp']}  | "
            f"Episode {r.get('episode_id','')}  | eval_valid={r.get('eval_valid','1')}")
        )
        for i, v in enumerate(self.cam_vars, 1):
            v.set(str(r.get(f"cctv{i}_count", "")))
        self.total_var.set(str(r.get("ground_truth_occupied_space_count", "")))
        self.unique_var.set(str(r.get("ground_truth_unique_vehicle_count", "")))
        self.tag_var.set(str(r.get("tags", "")))
        self.note_var.set(str(r.get("notes", "")))

    def save_current(self):
        idx = self.df.index[self.index]
        for i, v in enumerate(self.cam_vars, 1):
            self.df.at[idx, f"cctv{i}_count"] = v.get().strip()
        self.df.at[idx, "ground_truth_occupied_space_count"] = self.total_var.get().strip()
        self.df.at[idx, "ground_truth_unique_vehicle_count"] = self.unique_var.get().strip()
        self.df.at[idx, "tags"] = self.tag_var.get().strip()
        self.df.at[idx, "notes"] = self.note_var.get().strip()
        self.df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")

    def next(self):
        self.save_current()
        if self.index < len(self.df)-1:
            self.index += 1
            self._load_row()

    def prev(self):
        self.save_current()
        if self.index > 0:
            self.index -= 1
            self._load_row()

    def mark_unknown(self):
        idx = self.df.index[self.index]
        self.df.at[idx, "eval_valid"] = "0"
        tag = set(x for x in str(self.tag_var.get()).replace(",", ";").split(";") if x)
        tag.add("U")
        self.tag_var.set(";".join(sorted(tag)))
        self.save_current()
        self.next()

    def mark_cut(self):
        idx = self.df.index[self.index]
        self.df.at[idx, "eval_valid"] = "0"
        tag = set(x for x in str(self.tag_var.get()).replace(",", ";").split(";") if x)
        tag.add("CUT")
        self.tag_var.set(";".join(sorted(tag)))
        self.save_current()

    def _close(self):
        try:
            self.save_current()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        self.destroy()


def _metric_row(name: str, pred: pd.Series, gt: pd.Series, valid: pd.Series, episode="ALL"):
    d = pd.DataFrame({"p": pd.to_numeric(pred, errors="coerce"), "g": pd.to_numeric(gt, errors="coerce"), "v": valid.astype(bool)})
    d = d[d["v"] & d["p"].notna() & d["g"].notna()]
    if d.empty:
        return {"algorithm": name, "episode": episode, "N": 0}
    err = d["p"] - d["g"]
    ae = err.abs()
    return {
        "algorithm": name, "episode": episode, "N": int(len(d)),
        "exact_rate": float((ae == 0).mean()),
        "MAE": float(ae.mean()),
        "max_abs_error": float(ae.max()),
        "under_rate": float((err < 0).mean()),
        "over_rate": float((err > 0).mean()),
    }


def write_episode_metrics(run_dir: str | Path, settings: Dict) -> Optional[Path]:
    run_dir = Path(run_dir)
    cfg = settings.get("validation", {}) or {}
    cuts = [float(x) for x in cfg.get("cut_boundaries_sec", [])]
    warmup = float(cfg.get("cut_warmup_sec", settings.get("evaluation_warmup_sec", 10.0)))
    base_path = run_dir / "baseline_count_timeseries.csv"
    if not base_path.is_file():
        return None
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    if "time_sec" not in base or "ground_truth_occupied_space_count" not in base:
        return None
    base["episode_id"] = base["time_sec"].astype(float).apply(lambda t: _episode_id(t, cuts))
    base["episode_eval_valid"] = base["time_sec"].astype(float).apply(lambda t: int(_eval_valid(t, cuts, warmup)))
    base[["time_sec","timestamp","episode_id","episode_eval_valid"]].to_csv(
        run_dir/"episode_evaluation_mask.csv", index=False, encoding="utf-8-sig"
    )
    rows = []
    valid = base["episode_eval_valid"].astype(int) == 1
    rows.append(_metric_row("v16.2_SAFE_BASELINE", base["occupied_pred"], base["ground_truth_occupied_space_count"], valid))
    for ep, d in base.groupby("episode_id"):
        rows.append(_metric_row("v16.2_SAFE_BASELINE", d["occupied_pred"], d["ground_truth_occupied_space_count"], d["episode_eval_valid"].astype(int)==1, f"EP{ep}"))

    cand_path = run_dir/"candidate_v164_count_timeseries.csv"
    if cand_path.is_file():
        cand = pd.read_csv(cand_path, encoding="utf-8-sig")
        merged = base[["time_sec","episode_id","episode_eval_valid","ground_truth_occupied_space_count"]].merge(
            cand[["time_sec","occupied_pred_candidate"]], on="time_sec", how="left"
        )
        rows.append(_metric_row("v16.4_CANDIDATE", merged["occupied_pred_candidate"], merged["ground_truth_occupied_space_count"], merged["episode_eval_valid"].astype(int)==1))
        for ep, d in merged.groupby("episode_id"):
            rows.append(_metric_row("v16.4_CANDIDATE", d["occupied_pred_candidate"], d["ground_truth_occupied_space_count"], d["episode_eval_valid"].astype(int)==1, f"EP{ep}"))
    out = run_dir/"episode_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def _repeat_rows(df: pd.DataFrame, pred_col: str, name: str, period: float, repeat_count: int):
    d = df.copy()
    d["repeat_id"] = (d["time_sec"].astype(float) // period).astype(int) + 1
    d["relative_sec"] = (d["time_sec"].astype(float) % period).round(3)
    if repeat_count > 0:
        d = d[d["repeat_id"] <= repeat_count]
    rows = []
    first = d[d["repeat_id"] == 1][["relative_sec", pred_col]].rename(columns={pred_col:"first_pred"})
    for rid, g in d.groupby("repeat_id"):
        m = g.merge(first, on="relative_sec", how="inner")
        if m.empty:
            continue
        diff = pd.to_numeric(m[pred_col], errors="coerce") - pd.to_numeric(m["first_pred"], errors="coerce")
        row = {
            "algorithm": name, "repeat_id": int(rid), "N_vs_first": int(diff.notna().sum()),
            "agreement_vs_first": float((diff.fillna(999) == 0).mean()),
            "MAE_vs_first": float(diff.abs().mean()),
            "max_drift_vs_first": float(diff.abs().max()),
        }
        if "ground_truth_occupied_space_count" in g.columns:
            gt = pd.to_numeric(g["ground_truth_occupied_space_count"], errors="coerce")
            pred = pd.to_numeric(g[pred_col], errors="coerce")
            ok = gt.notna() & pred.notna()
            if ok.any():
                ae = (pred[ok]-gt[ok]).abs()
                row["GT_exact_rate"] = float((ae == 0).mean())
                row["GT_MAE"] = float(ae.mean())
        rows.append(row)
    return rows


def write_repeat_stability(run_dir: str | Path, settings: Dict) -> Optional[Path]:
    cfg = settings.get("validation", {}) or {}
    period = float(cfg.get("repeat_period_sec", 0.0) or 0.0)
    if period <= 0:
        return None
    repeat_count = int(cfg.get("repeat_count", 0) or 0)
    run_dir = Path(run_dir)
    base_path = run_dir/"baseline_count_timeseries.csv"
    if not base_path.is_file():
        return None
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    rows = _repeat_rows(base, "occupied_pred", "v16.2_SAFE_BASELINE", period, repeat_count)
    cand_path = run_dir/"candidate_v164_count_timeseries.csv"
    if cand_path.is_file():
        cand = pd.read_csv(cand_path, encoding="utf-8-sig")
        merged = base[["time_sec","ground_truth_occupied_space_count"]].merge(
            cand[["time_sec","occupied_pred_candidate"]], on="time_sec", how="left"
        )
        rows += _repeat_rows(merged, "occupied_pred_candidate", "v16.4_CANDIDATE", period, repeat_count)
    if not rows:
        return None
    out = run_dir/"repeat_stability.csv"
    rdf = pd.DataFrame(rows)
    rdf.to_csv(out, index=False, encoding="utf-8-sig")
    report = [
        "Parking Research Agent v16.5 repeated-segment stability test",
        "===========================================================",
        "",
        f"repeat_period_sec={period}",
        f"requested_repeat_count={repeat_count or 'auto'}",
        "",
        "Purpose: system/state accumulation reproducibility only.",
        "A repeated source clip is NOT an independent validation video.",
        "",
    ]
    for r in rows:
        report.append(
            f"{r['algorithm']} repeat {r['repeat_id']}: agreement={r['agreement_vs_first']:.4f}, "
            f"MAE_vs_first={r['MAE_vs_first']:.4f}, max_drift={r['max_drift_vs_first']:.1f}"
        )
    (run_dir/"REPEAT_STABILITY_REPORT.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    return out


def _profiles() -> Dict:
    _ensure()
    return load_json(PROFILE_PATH, {}) or {}


def _save_profiles(data: Dict):
    _ensure()
    save_json(PROFILE_PATH, data)


def open_validation_center(app):
    _ensure()
    win = tk.Toplevel(app)
    win.title(tr("v16.5 Validation Center"))
    win.geometry("980x820")
    win.minsize(850, 700)

    settings = load_json(APP_DIR/"settings.json", getattr(app, "settings", {})) or {}
    vcfg = dict(settings.get("validation", {}) or {})
    ecfg = dict(settings.get("external_validation", {}) or {})

    local = ttk.LabelFrame(win, text=tr("A. Local / New Video Dataset Profile"))
    local.pack(fill="x", padx=10, pady=8)
    local.columnconfigure(1, weight=1)

    profile_var = tk.StringVar()
    profile_combo = ttk.Combobox(local, textvariable=profile_var, state="normal")
    profile_combo.grid(row=0,column=1,sticky="ew",padx=6,pady=5)
    ttk.Label(local,text=tr("Profile")).grid(row=0,column=0,sticky="w",padx=6,pady=5)

    cam_var = tk.IntVar(value=int(vcfg.get("camera_count", 3)))
    interval_var = tk.DoubleVar(value=float(vcfg.get("gt_interval_sec", 10.0)))
    cuts_var = tk.StringVar(value=format_time_list(vcfg.get("cut_boundaries_sec", [])))
    warm_var = tk.DoubleVar(value=float(vcfg.get("cut_warmup_sec", 10.0)))
    repeat_var = tk.DoubleVar(value=float(vcfg.get("repeat_period_sec", 0.0)))
    repeat_count_var = tk.IntVar(value=int(vcfg.get("repeat_count", 0)))

    ttk.Label(local,text=tr("CCTV count")).grid(row=1,column=0,sticky="w",padx=6,pady=5)
    ttk.Spinbox(local,from_=1,to=12,textvariable=cam_var,width=8).grid(row=1,column=1,sticky="w",padx=6,pady=5)
    ttk.Label(local,text=tr("GT interval (sec)")).grid(row=2,column=0,sticky="w",padx=6,pady=5)
    ttk.Entry(local,textvariable=interval_var,width=10).grid(row=2,column=1,sticky="w",padx=6,pady=5)
    ttk.Label(local,text=tr("Cut boundaries")).grid(row=3,column=0,sticky="w",padx=6,pady=5)
    ttk.Entry(local,textvariable=cuts_var).grid(row=3,column=1,sticky="ew",padx=6,pady=5)
    ttk.Label(local,text=tr("Cut warm-up exclude (sec)")).grid(row=4,column=0,sticky="w",padx=6,pady=5)
    ttk.Entry(local,textvariable=warm_var,width=10).grid(row=4,column=1,sticky="w",padx=6,pady=5)
    ttk.Label(local,text=tr("Repeated source period (sec, 0=off)")).grid(row=5,column=0,sticky="w",padx=6,pady=5)
    ttk.Entry(local,textvariable=repeat_var,width=10).grid(row=5,column=1,sticky="w",padx=6,pady=5)
    ttk.Label(local,text=tr("Repeat count (0=auto)")).grid(row=6,column=0,sticky="w",padx=6,pady=5)
    ttk.Entry(local,textvariable=repeat_count_var,width=10).grid(row=6,column=1,sticky="w",padx=6,pady=5)

    hint = (
        "Cut example: 4:00, 8:00, 12:00, 16:00. The cut itself and warm-up window are excluded from "
        "episode metrics. Repeated-source mode is a stability test, not independent validation."
    )
    ttk.Label(local,text=tr(hint),wraplength=900).grid(row=7,column=0,columnspan=3,sticky="w",padx=6,pady=5)

    def collect_validation():
        return {
            "camera_count": max(1, int(cam_var.get())),
            "gt_interval_sec": max(1.0, float(interval_var.get())),
            "cut_boundaries_sec": parse_time_list(cuts_var.get()),
            "cut_warmup_sec": max(0.0, float(warm_var.get())),
            "repeat_period_sec": max(0.0, float(repeat_var.get())),
            "repeat_count": max(0, int(repeat_count_var.get())),
        }

    def save_settings_only(show=True):
        nonlocal settings
        settings = load_json(APP_DIR/"settings.json", settings) or {}
        settings["validation"] = collect_validation()
        save_json(APP_DIR/"settings.json", settings)
        app.settings = settings
        if show:
            messagebox.showinfo(tr("Saved"), tr("v16.5 validation settings saved."), parent=win)
        return settings

    def refresh_profiles():
        names = sorted(_profiles().keys())
        profile_combo["values"] = names

    def save_profile():
        name = profile_var.get().strip()
        if not name:
            messagebox.showwarning(tr("Profile name"), tr("Enter a profile name."), parent=win); return
        cfg = collect_validation()
        data = _profiles()
        data[name] = {
            "video_path": app.video_var.get().strip(),
            "gt_path": app.gt_var.get().strip(),
            "slot_gt_path": app.slot_gt_var.get().strip(),
            "validation": cfg,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_profiles(data)
        save_settings_only(show=False)
        refresh_profiles()
        messagebox.showinfo(tr("Profile saved"), tr(name), parent=win)

    def load_profile():
        name = profile_var.get().strip()
        p = _profiles().get(name)
        if not p:
            messagebox.showwarning(tr("Profile"), tr("Profile not found."), parent=win); return
        app.video_var.set(str(p.get("video_path","")))
        app.gt_var.set(str(p.get("gt_path","")))
        app.slot_gt_var.set(str(p.get("slot_gt_path","")))
        cfg = p.get("validation", {}) or {}
        cam_var.set(int(cfg.get("camera_count",3)))
        interval_var.set(float(cfg.get("gt_interval_sec",10)))
        cuts_var.set(format_time_list(cfg.get("cut_boundaries_sec",[])))
        warm_var.set(float(cfg.get("cut_warmup_sec",10)))
        repeat_var.set(float(cfg.get("repeat_period_sec",0)))
        repeat_count_var.set(int(cfg.get("repeat_count",0)))
        save_settings_only(show=False)
        try: app._save_ui_state()
        except Exception: pass
        messagebox.showinfo(tr("Profile loaded"), tr(name), parent=win)

    row_actions = ttk.Frame(local)
    row_actions.grid(row=8,column=0,columnspan=3,sticky="ew",padx=6,pady=6)
    ttk.Button(row_actions,text=tr("Save Validation Settings"),command=save_settings_only).pack(side="left",padx=3)
    ttk.Button(row_actions,text=tr("Save Profile"),command=save_profile).pack(side="left",padx=3)
    ttk.Button(row_actions,text=tr("Load Profile"),command=load_profile).pack(side="left",padx=3)

    def create_template():
        video = app.video_var.get().strip()
        if not video or not Path(video).is_file():
            messagebox.showerror(tr("Video"), tr("Select a local video in the main window first."), parent=win); return
        cfg = collect_validation()
        out = filedialog.asksaveasfilename(
            parent=win, defaultextension=".csv", initialfile="ground_truth_v165.csv",
            filetypes=[("CSV","*.csv")]
        )
        if not out: return
        build_gt_template(video,out,cfg["camera_count"],cfg["gt_interval_sec"],cfg["cut_boundaries_sec"],cfg["cut_warmup_sec"])
        app.gt_var.set(out)
        try: app._save_ui_state()
        except Exception: pass
        messagebox.showinfo(tr("GT template"), tr(f"Created:\n{out}"), parent=win)

    def label_gt():
        video = app.video_var.get().strip()
        gt = app.gt_var.get().strip()
        if not video or not Path(video).is_file():
            messagebox.showerror(tr("Video"), tr("Select a local video first."), parent=win); return
        if not gt:
            messagebox.showerror(tr("GT"), tr("Create/select a GT CSV first."), parent=win); return
        cfg = collect_validation()
        CountGTLabeler(win, video, gt, cfg["camera_count"], cfg["gt_interval_sec"], cfg["cut_boundaries_sec"], cfg["cut_warmup_sec"])

    row_gt = ttk.Frame(local)
    row_gt.grid(row=9,column=0,columnspan=3,sticky="ew",padx=6,pady=(0,8))
    ttk.Button(row_gt,text=tr("Create Count-GT Template"),command=create_template).pack(side="left",padx=3)
    ttk.Button(row_gt,text=tr("Open GT Labeler"),command=label_gt).pack(side="left",padx=3)

    public = ttk.LabelFrame(win, text=tr("B. Public External Environment Validation"))
    public.pack(fill="x", padx=10, pady=8)
    public.columnconfigure(1,weight=1)
    mode_var = tk.StringVar(value=str(ecfg.get("download_mode","quick")).lower())
    ttk.Label(public,text=tr("MetaPKLot / CNRPark-EXT")).grid(row=0,column=0,sticky="w",padx=6,pady=5)
    ttk.Combobox(public,textvariable=mode_var,values=["quick","standard","full"],state="readonly",width=12).grid(row=0,column=1,sticky="w",padx=6,pady=5)
    ext_status = tk.StringVar(value=tr(f"Dataset root: {PUBLIC_ROOT}"))
    ext_progress = tk.DoubleVar(value=0.0)
    ttk.Progressbar(public,variable=ext_progress,maximum=100).grid(row=1,column=0,columnspan=3,sticky="ew",padx=6,pady=5)
    ttk.Label(public,textvariable=ext_status,wraplength=900).grid(row=2,column=0,columnspan=3,sticky="w",padx=6,pady=5)

    def pupdate(r,t):
        win.after(0,lambda: ext_progress.set(float(r)*100))
        win.after(0,lambda: ext_status.set(tr(str(t))))

    def download_public():
        mode = mode_var.get().strip().lower()
        settings_now = save_settings_only(show=False)
        settings_now.setdefault("external_validation",{})["download_mode"] = mode
        save_json(APP_DIR/"settings.json",settings_now)
        def worker():
            try:
                result = prepare_metapklot_cnr(PUBLIC_ROOT,mode,pupdate)
                win.after(0,lambda: messagebox.showinfo(tr("Public dataset ready"),tr(f"Images: {result['images']}\nGT: {result['gt']}"),parent=win))
            except Exception as exc:
                win.after(0,lambda e=exc: messagebox.showerror(tr("Public dataset"),tr(f"{type(e).__name__}: {e}"),parent=win))
        threading.Thread(target=worker,daemon=True).start()

    def run_external():
        settings_now = save_settings_only(show=False)
        if not (PUBLIC_ROOT/"external_gt_spots.csv").is_file():
            messagebox.showwarning(tr("Dataset"), tr("Download & Prepare the public dataset first."), parent=win); return
        out = PUBLIC_ROOT/"validation_results"/time.strftime("%Y%m%d_%H%M%S")
        def worker():
            try:
                result = evaluate_external_cnr(PUBLIC_ROOT,settings_now,out,pupdate)
                m=result["overall"]
                msg=(f"External validation complete\n\nAccuracy {m['accuracy']*100:.2f}%\n"
                     f"F1 {m['f1']:.4f}\nOccupied Recall {m['occupied_recall']:.4f}\n"
                     f"Empty Specificity {m['empty_specificity']:.4f}\n\n{out}")
                win.after(0,lambda: messagebox.showinfo(tr("External validation"),tr(msg),parent=win))
            except Exception as exc:
                win.after(0,lambda e=exc: messagebox.showerror(tr("External validation"),tr(f"{type(e).__name__}: {e}"),parent=win))
        threading.Thread(target=worker,daemon=True).start()

    buttons = ttk.Frame(public)
    buttons.grid(row=3,column=0,columnspan=3,sticky="ew",padx=6,pady=8)
    ttk.Button(buttons,text=tr("Download & Prepare"),command=download_public).pack(side="left",padx=3)
    ttk.Button(buttons,text=tr("Run External Occupancy Validation"),command=run_external).pack(side="left",padx=3)
    ttk.Button(buttons,text=tr("Open Dataset Folder"),command=lambda: os.startfile(str(PUBLIC_ROOT)) if os.name=="nt" else None).pack(side="left",padx=3)
    ttk.Label(public,text=(
        tr("Quick/Standard download only selected official images + matching MetaPKLot spot annotations. "
        "Full is explicit opt-in and may be several GB. Public evaluation tests spatial generalization; "
        "it does not claim continuous transition validation.")
    ),wraplength=900).grid(row=4,column=0,columnspan=3,sticky="w",padx=6,pady=(0,8))

    refresh_profiles()
