# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

APP_DIR = Path(__file__).resolve().parent
CANONICAL_HISTORY = APP_DIR / "research_history.json"
LOCAL_ROOT = Path(os.environ.get("LOCALAPPDATA", str(APP_DIR / "work"))) / "ParkingResearchAgent"
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_RUNS = LOCAL_ROOT / "research_runs.json"


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_history():
    data = _read_json(CANONICAL_HISTORY, {"entries": []})
    entries = list(data.get("entries", []))
    local = _read_json(LOCAL_RUNS, {"entries": []})
    for item in local.get("entries", []):
        row = dict(item)
        row.setdefault("source", "local_run")
        entries.append(row)
    return data, entries


def append_run(version: str, run_dir: str, summary: dict | None = None) -> None:
    payload = _read_json(LOCAL_RUNS, {"schema_version": 1, "entries": []})
    entries = list(payload.get("entries", []))
    item = {
        "version": str(version),
        "period": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "LOCAL_RUN",
        "confidence": "generated",
        "title": "Automated research run",
        "goal": "Record an Agent-run experiment without overwriting the canonical research timeline.",
        "changes": [],
        "results": [],
        "issues": [],
        "decision": "Pending review",
        "run_dir": str(run_dir),
    }
    if summary:
        item["summary"] = summary
    entries.append(item)
    payload["entries"] = entries[-500:]
    _write_json(LOCAL_RUNS, payload)


def _details_text(item: dict) -> str:
    lines = [
        f"Version: {item.get('version', '')}",
        f"Period: {item.get('period', '')}",
        f"Status: {item.get('status', '')}",
        f"Confidence: {item.get('confidence', '')}",
        "",
        str(item.get("title", "")),
        "",
        "Goal",
        str(item.get("goal", "")),
    ]
    for heading, key in (("Changes", "changes"), ("Results", "results"), ("Issues", "issues")):
        lines += ["", heading]
        vals = item.get(key, []) or []
        if vals:
            lines.extend([f"- {x}" for x in vals])
        else:
            lines.append("- (none recorded)")
    lines += ["", "Decision", str(item.get("decision", ""))]
    if item.get("run_dir"):
        lines += ["", "Run directory", str(item.get("run_dir"))]
    return "\n".join(lines)


def open_history_window(parent=None):
    meta, entries = load_history()
    win = tk.Toplevel(parent) if parent is not None else tk.Tk()
    win.title("Research History | Connect Hyundai Parking")
    win.geometry("1120x720")
    win.minsize(860, 560)

    header = ttk.Frame(win)
    header.pack(fill="x", padx=10, pady=(10, 4))
    ttk.Label(header, text="Research History", font=("Segoe UI", 16, "bold")).pack(side="left")
    ttk.Label(header, text="paper/research lineage only | field-operation v20+ kept separate").pack(side="left", padx=12)

    body = ttk.Panedwindow(win, orient="horizontal")
    body.pack(fill="both", expand=True, padx=10, pady=10)
    left = ttk.Frame(body)
    right = ttk.Frame(body)
    body.add(left, weight=2)
    body.add(right, weight=5)

    cols = ("version", "period", "status")
    tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
    for c, w in (("version", 110), ("period", 130), ("status", 170)):
        tree.heading(c, text=c.title())
        tree.column(c, width=w, anchor="w")
    y = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=y.set)
    tree.pack(side="left", fill="both", expand=True)
    y.pack(side="right", fill="y")

    detail = tk.Text(right, wrap="word", font=("Consolas", 10), padx=12, pady=12)
    dy = ttk.Scrollbar(right, orient="vertical", command=detail.yview)
    detail.configure(yscrollcommand=dy.set)
    detail.pack(side="left", fill="both", expand=True)
    dy.pack(side="right", fill="y")

    for idx, item in enumerate(entries):
        tree.insert("", "end", iid=str(idx), values=(item.get("version", ""), item.get("period", ""), item.get("status", "")))

    def show_selected(_event=None):
        sel = tree.selection()
        if not sel:
            return
        item = entries[int(sel[0])]
        detail.configure(state="normal")
        detail.delete("1.0", "end")
        detail.insert("1.0", _details_text(item))
        detail.configure(state="disabled")

    tree.bind("<<TreeviewSelect>>", show_selected)
    if entries:
        tree.selection_set(str(len(entries) - 1))
        tree.see(str(len(entries) - 1))
        show_selected()
    return win
