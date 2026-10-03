# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
VERSION_PATH = APP_DIR / "VERSION.txt"
REPO = "sopo9880/Parking_"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"

DEFAULT_UPDATE_CONFIG = {
    "enabled": True,
    "check_on_startup": True,
    "ask_before_install": True,
    "repo": REPO,
    "asset_prefix": "ParkingResearchAgent-",
}


def _headers():
    return {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ParkingResearchAgent-Updater",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dst: Path, timeout: int = 120):
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)


def normalize_version(v: str):
    nums = [int(x) for x in re.findall(r"\d+", str(v))[:3]]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def current_version() -> str:
    try:
        return VERSION_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return "v0.0.0"


def load_update_config(settings: dict | None = None):
    cfg = dict(DEFAULT_UPDATE_CONFIG)
    if isinstance(settings, dict):
        cfg.update(settings.get("auto_update", {}) or {})
    return cfg


def check_latest(settings: dict | None = None):
    cfg = load_update_config(settings)
    repo = str(cfg.get("repo") or REPO)
    data = _get_json(f"https://api.github.com/repos/{repo}/releases/latest")
    latest = str(data.get("tag_name") or data.get("name") or "").strip()
    cur = current_version()
    assets = data.get("assets", []) or []
    prefix = str(cfg.get("asset_prefix") or "ParkingResearchAgent-")
    zip_asset = None
    sha_asset = None
    for a in assets:
        name = str(a.get("name", ""))
        if name.startswith(prefix) and name.lower().endswith(".zip"):
            zip_asset = a
        if name.startswith(prefix) and name.lower().endswith(".sha256"):
            sha_asset = a
    return {
        "current": cur,
        "latest": latest,
        "available": bool(latest) and normalize_version(latest) > normalize_version(cur),
        "html_url": data.get("html_url", f"https://github.com/{repo}/releases/latest"),
        "body": data.get("body", "") or "",
        "zip_asset": zip_asset,
        "sha_asset": sha_asset,
        "raw": data,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_update(info: dict) -> Path:
    za = info.get("zip_asset")
    sa = info.get("sha_asset")
    if not za or not sa:
        raise RuntimeError("Release is missing the ZIP or SHA-256 asset.")
    tmp = Path(tempfile.mkdtemp(prefix="parking_agent_update_"))
    zpath = tmp / str(za["name"])
    spath = tmp / str(sa["name"])
    _download(str(za["browser_download_url"]), zpath)
    _download(str(sa["browser_download_url"]), spath)
    expected = spath.read_text(encoding="utf-8", errors="replace").strip().split()[0].lower()
    actual = _sha256(zpath).lower()
    if expected != actual:
        raise RuntimeError(f"SHA-256 mismatch. expected={expected} actual={actual}")
    return zpath


def launch_apply(zip_path: Path, restart: bool = True):
    helper = APP_DIR / "update_helper.py"
    if not helper.is_file():
        raise RuntimeError("update_helper.py is missing.")
    python = sys.executable
    args = [python, str(helper), "--zip", str(zip_path), "--app-dir", str(APP_DIR), "--pid", str(os.getpid())]
    if restart:
        args.append("--restart")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(args, cwd=str(APP_DIR), creationflags=creationflags, close_fds=(os.name != "nt"))
