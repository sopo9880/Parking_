# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

PRESERVE_FILES = {
    "settings.json",
    "slots.json",
    "rois.json",
    "ui_state.json",
    "sample_ground_truth.csv",
}
PRESERVE_DIRS = {"work", "output", ".venv", "user_data"}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # tasklist works without extra dependencies.
        p = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return str(pid) in p.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _safe_extract(zf: zipfile.ZipFile, dst: Path):
    root = dst.resolve()
    for m in zf.infolist():
        target = (dst / m.filename).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Unsafe path in update ZIP: {m.filename}")
    zf.extractall(dst)


def _payload_root(extract_dir: Path) -> Path:
    children = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir() and (children[0] / "VERSION.txt").is_file():
        return children[0]
    return extract_dir


def _copy_payload(src: Path, dst: Path):
    for item in src.iterdir():
        name = item.name
        if name in PRESERVE_FILES or name in PRESERVE_DIRS or name == ".git":
            continue
        target = dst / name
        if item.is_dir():
            if target.exists() and target.is_file():
                target.unlink()
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _merge_settings_defaults(defaults, current):
    """Return defaults with the user's existing values recursively overlaid."""
    if not isinstance(defaults, dict) or not isinstance(current, dict):
        return current
    merged = dict(defaults)
    for key, value in current.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_settings_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_preserved_settings(app: Path):
    defaults_path = app / "settings.defaults.json"
    settings_path = app / "settings.json"
    if not defaults_path.is_file() or not settings_path.is_file():
        return False
    try:
        defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
        current = json.loads(settings_path.read_text(encoding="utf-8"))
        merged = _merge_settings_defaults(defaults, current)
        tmp = settings_path.with_suffix(".json.update_tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(settings_path)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--app-dir", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    app = Path(args.app_dir).resolve()
    zpath = Path(args.zip).resolve()
    log_dir = app / "user_data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "update.log"

    def write(msg):
        with log.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")

    write(f"Waiting for PID {args.pid} to exit")
    for _ in range(180):
        if not _pid_alive(args.pid):
            break
        time.sleep(1)
    else:
        write("Timed out waiting for application exit")
        return 2

    extract_dir = Path(str(zpath) + "_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        _safe_extract(zf, extract_dir)
    src = _payload_root(extract_dir)
    if not (src / "VERSION.txt").is_file():
        raise RuntimeError("Update payload does not contain VERSION.txt")

    backup = app / "user_data" / "update_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup.mkdir(parents=True, exist_ok=True)
    for name in PRESERVE_FILES:
        p = app / name
        if p.is_file():
            shutil.copy2(p, backup / name)

    write(f"Applying update from {zpath.name}")
    _copy_payload(src, app)
    if _merge_preserved_settings(app):
        write("Preserved settings.json merged with new settings.defaults.json keys")
    else:
        write("Settings defaults merge skipped or unavailable")
    write(f"Update installed: {(app / 'VERSION.txt').read_text(encoding='utf-8').strip()}")

    if args.restart:
        if os.name == "nt" and (app / "RUN.cmd").is_file():
            subprocess.Popen(["cmd", "/c", "start", "", str(app / "RUN.cmd")], cwd=str(app))
        else:
            subprocess.Popen([sys.executable, str(app / "app.py")], cwd=str(app))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
