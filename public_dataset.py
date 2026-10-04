# -*- coding: utf-8 -*-
"""Public-dataset download/preparation and external-environment validation.

v16.5 uses MetaPKLot/CNRPark-EXT as an *external spatial generalization* test.
It is intentionally separate from the temporal Hyundai CCTV validation because
the public dataset is composed of timestamped still images rather than our
continuous maneuvering sequence.

Source:
  https://github.com/DSBD-Research/MetaPKLot-Dataset
CNRPark-EXT license:
  ODbL v1.0 (see upstream README)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from detector import VehicleDetector
from external_evidence import deduplicate_spots, freeze_baseline, binary_metrics, write_results, file_hash

REPO = "DSBD-Research/MetaPKLot-Dataset"
REPO_URL = f"https://github.com/{REPO}.git"
API_ROOT = f"https://api.github.com/repos/{REPO}/contents"
RAW_ROOT = f"https://raw.githubusercontent.com/{REPO}/main"
SOURCE_PAGE = f"https://github.com/{REPO}"
USER_AGENT = "ParkingResearchAgent-v16.5"

MODE_PLANS = {
    "quick": {
        "cameras": [1, 4],
        "weather": ["SUNNY", "RAINY"],
        "dates_per_weather": 1,
        "images_per_date": 6,
    },
    "standard": {
        "cameras": [1, 4, 8],
        "weather": ["SUNNY", "OVERCAST", "RAINY"],
        "dates_per_weather": 2,
        "images_per_date": 10,
    },
}


def _progress(cb, ratio: float, text: str):
    if cb:
        cb(max(0.0, min(1.0, float(ratio))), str(text))


def _request(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60):
    h = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    if headers:
        h.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def _api_json(path: str):
    url = path if path.startswith("http") else f"{API_ROOT}/{path.lstrip('/')}?ref=main"
    last = None
    for attempt in range(3):
        try:
            with _request(url) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"MetaPKLot GitHub API failed: {url}: {last}")


def _download(url: str, dst: Path, expected_size: int = 0,
              progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Resumable HTTP download using Range when the server supports it."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if expected_size and dst.is_file() and dst.stat().st_size == expected_size:
        if progress:
            progress(expected_size, expected_size)
        return dst

    have = dst.stat().st_size if dst.is_file() else 0
    headers = {}
    mode = "wb"
    if have > 0:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
    try:
        resp = _request(url, headers=headers, timeout=90)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and expected_size and have == expected_size:
            return dst
        if have:
            have = 0
            mode = "wb"
            resp = _request(url, timeout=90)
        else:
            raise

    status = getattr(resp, "status", 200)
    if have and status != 206:
        have = 0
        mode = "wb"
    total = expected_size
    if not total:
        try:
            clen = int(resp.headers.get("Content-Length", "0") or 0)
            total = have + clen if status == 206 else clen
        except Exception:
            total = 0
    done = have
    with resp, dst.open(mode) as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if expected_size and dst.stat().st_size != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {dst.name}: "
            f"{dst.stat().st_size} != {expected_size}"
        )
    return dst


def _even_sample(items: Sequence[dict], count: int) -> List[dict]:
    files = [x for x in items if x.get("type") == "file" and str(x.get("name", "")).lower().endswith((".jpg", ".jpeg", ".png"))]
    if count <= 0 or len(files) <= count:
        return files
    idx = np.linspace(0, len(files) - 1, count).round().astype(int)
    return [files[int(i)] for i in idx]


def _select_remote_files(mode: str, progress=None) -> List[dict]:
    plan = MODE_PLANS[mode]
    selected: List[dict] = []
    tasks = [(cam, weather) for cam in plan["cameras"] for weather in plan["weather"]]
    for ti, (cam, weather) in enumerate(tasks):
        _progress(progress, ti / max(1, len(tasks)), f"Listing camera{cam}/{weather}")
        dates = _api_json(f"CNRPark-EXT/camera{cam}/{weather}")
        date_dirs = sorted([x for x in dates if x.get("type") == "dir"], key=lambda x: x.get("name", ""))
        for d in date_dirs[: int(plan["dates_per_weather"])]:
            entries = _api_json(d["path"])
            for item in _even_sample(entries, int(plan["images_per_date"])):
                item = dict(item)
                item["camera"] = f"camera{cam}"
                item["weather"] = weather
                item["date"] = d.get("name", "")
                selected.append(item)
    return selected


def _annotation_archive_name(camera_number: int) -> str:
    return f"cnr-camera-{camera_number}_spots.tar.xz"


def _extract_annotations(root: Path, cameras: Iterable[int]) -> Path:
    out = root / "annotations_extracted"
    out.mkdir(parents=True, exist_ok=True)
    for cam in sorted(set(int(x) for x in cameras)):
        archive = root / "downloads" / "annotations" / _annotation_archive_name(cam)
        if not archive.is_file():
            continue
        marker = out / f".camera{cam}.done"
        if marker.is_file() and marker.stat().st_mtime_ns >= archive.stat().st_mtime_ns:
            continue
        with tarfile.open(archive, mode="r:xz") as tf:
            tf.extractall(out / f"camera{cam}")
        marker.write_text("ok\n", encoding="utf-8")
    return out


def _normalize_rel(path_text: str) -> str:
    return str(path_text or "").replace("\\", "/").lstrip("./")


def _find_local_image(root: Path, file_name: str, downloaded: Dict[str, Path]) -> Optional[Path]:
    n = _normalize_rel(file_name)
    candidates = [n]
    if n.startswith("CNRPark-EXT/"):
        candidates.append(n[len("CNRPark-EXT/"):])
    else:
        candidates.append("CNRPark-EXT/" + n)
    for c in candidates:
        if c in downloaded:
            return downloaded[c]
    # Upstream annotations occasionally vary in prefix/case. Suffix match is deterministic
    # because each CNRPark image filename includes timestamp and lives under camera/weather/date.
    nlow = n.lower()
    for k, p in downloaded.items():
        klow = k.lower()
        if klow.endswith(nlow) or nlow.endswith(klow):
            return p
    return None


def _json_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def _polygon_from_segmentation(seg) -> List[List[float]]:
    if not seg:
        return []
    vals = seg[0] if isinstance(seg, list) and seg and isinstance(seg[0], list) else seg
    if not isinstance(vals, list) or len(vals) < 6:
        return []
    pts = []
    for i in range(0, len(vals) - 1, 2):
        try:
            pts.append([float(vals[i]), float(vals[i + 1])])
        except Exception:
            return []
    return pts


def convert_spot_annotations(root: str | Path, progress=None) -> Path:
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloaded = {
        _normalize_rel(x["repo_path"]): root / x["local_path"]
        for x in manifest.get("images", [])
        if (root / x["local_path"]).is_file()
    }
    ann_root = root / "annotations_extracted"
    rows: List[dict] = []
    jsons = _json_files(ann_root)
    for ji, jp in enumerate(jsons):
        _progress(progress, ji / max(1, len(jsons)), f"Converting annotations {jp.name}")
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(jp.read_text(encoding="latin-1"))
        except Exception:
            continue
        images = {int(x.get("id")): x for x in data.get("images", []) if "id" in x}
        grouped = defaultdict(list)
        for a in data.get("annotations", []):
            try:
                grouped[int(a.get("image_id"))].append(a)
            except Exception:
                continue
        for image_id, anns in grouped.items():
            meta = images.get(image_id)
            if not meta:
                continue
            local = _find_local_image(root, str(meta.get("file_name", "")), downloaded)
            if local is None:
                continue
            repo_rel = None
            for k, p in downloaded.items():
                if p.resolve() == local.resolve():
                    repo_rel = k
                    break
            parts = _normalize_rel(repo_rel or meta.get("file_name", "")).split("/")
            camera = next((p for p in parts if p.lower().startswith("camera")), "")
            weather = next((p for p in parts if p.upper() in {"SUNNY", "OVERCAST", "RAINY"}), "")
            date_arr = meta.get("date", [])
            time_arr = meta.get("time", [])
            timestamp = ""
            if isinstance(date_arr, list) and isinstance(time_arr, list) and len(date_arr) >= 3 and len(time_arr) >= 3:
                timestamp = f"{int(date_arr[0]):04d}-{int(date_arr[1]):02d}-{int(date_arr[2]):02d} {int(time_arr[0]):02d}:{int(time_arr[1]):02d}:{int(time_arr[2]):02d}"
            for a in anns:
                poly = _polygon_from_segmentation(a.get("segmentation"))
                if len(poly) < 3:
                    continue
                try:
                    category_id = int(a.get("category_id", 0))
                except Exception:
                    category_id = 0
                rows.append({
                    "image_path": str(local.relative_to(root)).replace("\\", "/"),
                    "repo_path": repo_rel or "",
                    "camera": camera,
                    "weather": weather,
                    "timestamp": timestamp,
                    "image_id": image_id,
                    "spot_id": int(a.get("id", len(rows))),
                    "occupied_gt": 1 if category_id == 1 else 0,
                    "car_id": int(a.get("car_id", -1)) if str(a.get("car_id", "")).lstrip("-").isdigit() else -1,
                    "polygon_json": json.dumps(poly, separators=(",", ":")),
                    "bbox_json": json.dumps(a.get("bbox", []), separators=(",", ":")),
                    "source_json": str(jp.relative_to(root)).replace("\\", "/"),
                })
    if not rows:
        raise RuntimeError(
            "No MetaPKLot spot annotations matched the downloaded images. "
            "The upstream archive layout may have changed."
        )
    out = root / "external_gt_spots.csv"
    unique, audit = deduplicate_spots(pd.DataFrame(rows), 'annotation_conversion')
    unique.to_csv(out, index=False, encoding="utf-8-sig")
    (root/'annotation_deduplication.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    _progress(progress, 1.0, f"Annotation deduplication: {audit['duplicates_removed']} removed; {audit['unique_pairs']} unique pairs")
    return out


def prepare_metapklot_cnr(root: str | Path, mode: str = "quick", progress=None) -> Dict:
    """Download and prepare CNRPark-EXT subset/full dataset.

    quick/standard: targeted GitHub API + raw downloads.
    full: shallow clone of the complete upstream repository (explicit opt-in).
    """
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mode = str(mode).strip().lower()
    if mode not in {"quick", "standard", "full"}:
        raise ValueError("mode must be quick, standard or full")

    if mode == "full":
        repo_dir = root / "upstream_full"
        if not repo_dir.exists():
            git = shutil.which("git")
            if not git:
                raise RuntimeError(
                    "Full mode requires Git. Use Quick/Standard or install Git for Windows."
                )
            _progress(progress, 0.02, "Cloning full MetaPKLot repository")
            subprocess.run(
                [git, "clone", "--depth", "1", "https://github.com/DSBD-Research/MetaPKLot-Dataset.git", str(repo_dir)],
                check=True,
            )
        # Copy/link CNR images into a stable layout used by the evaluator.
        image_root = repo_dir / "CNRPark-EXT"
        if not image_root.is_dir():
            raise RuntimeError("Full clone does not contain CNRPark-EXT")
        images = []
        all_imgs = sorted(image_root.rglob("*.jpg"))
        for p in all_imgs:
            rel = p.relative_to(repo_dir)
            images.append({
                "repo_path": str(rel).replace("\\", "/"),
                "local_path": str(p.relative_to(root)).replace("\\", "/"),
                "size": int(p.stat().st_size),
            })
        ann_out = root / "annotations_extracted"
        ann_out.mkdir(parents=True, exist_ok=True)
        for archive in sorted((repo_dir / "annotations" / "original" / "spots" / "CNRPark-EXT").glob("*.tar.xz")):
            cam = archive.stem.split("-")[2] if "-" in archive.stem else archive.stem
            dst = ann_out / str(cam)
            dst.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:xz") as tf:
                tf.extractall(dst)
        manifest = {
            "dataset": "MetaPKLot/CNRPark-EXT",
            "mode": mode,
            "source": SOURCE_PAGE,
            "license": "ODbL-1.0 for CNRPark-EXT; see upstream README",
            "images": images,
            "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        gt = convert_spot_annotations(root, progress=lambda r, t: _progress(progress, 0.85 + 0.15*r, t))
        return {"root": str(root), "manifest": str(root/"manifest.json"), "gt": str(gt), "images": len(images)}

    selected = _select_remote_files(mode, progress=lambda r, t: _progress(progress, r*0.12, t))
    cameras = sorted({int(str(x["camera"]).replace("camera", "")) for x in selected})
    total_bytes = sum(int(x.get("size", 0) or 0) for x in selected)
    done_bytes = 0
    manifest_images = []

    for i, item in enumerate(selected):
        repo_path = _normalize_rel(item["path"])
        local = root / "downloads" / repo_path
        size = int(item.get("size", 0) or 0)
        before = local.stat().st_size if local.is_file() else 0
        def one_progress(done, total, i=i, name=item.get("name", "")):
            nonlocal done_bytes
            current_extra = max(0, done - min(before, size if size else before))
            ratio_bytes = (done_bytes + current_extra) / max(1, total_bytes)
            _progress(progress, 0.12 + 0.68 * ratio_bytes, f"Downloading {name}")
        _download(str(item["download_url"]), local, size, one_progress)
        done_bytes += size if size else int(local.stat().st_size)
        manifest_images.append({
            "repo_path": repo_path,
            "local_path": str(local.relative_to(root)).replace("\\", "/"),
            "size": int(local.stat().st_size),
            "camera": item["camera"],
            "weather": item["weather"],
            "date": item["date"],
        })

    for ai, cam in enumerate(cameras):
        name = _annotation_archive_name(cam)
        path = f"annotations/original/spots/CNRPark-EXT/{name}"
        meta = _api_json(path)
        dst = root / "downloads" / "annotations" / name
        _download(meta["download_url"], dst, int(meta.get("size", 0) or 0))
        _progress(progress, 0.80 + 0.05*(ai+1)/max(1, len(cameras)), f"Annotation camera{cam}")

    manifest = {
        "dataset": "MetaPKLot/CNRPark-EXT",
        "mode": mode,
        "source": SOURCE_PAGE,
        "license": "ODbL-1.0 for CNRPark-EXT; see upstream README",
        "images": manifest_images,
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _extract_annotations(root, cameras)
    gt = convert_spot_annotations(root, progress=lambda r, t: _progress(progress, 0.86 + 0.14*r, t))
    (root / "SOURCE_AND_LICENSE.txt").write_text(
        "MetaPKLot / CNRPark-EXT\n"
        f"Source: {SOURCE_PAGE}\n"
        "CNRPark-EXT license: Open Data Commons Open Database License (ODbL) v1.0.\n"
        "MetaPKLot combines multiple upstream datasets; consult the upstream README before redistribution.\n",
        encoding="utf-8",
    )
    _progress(progress, 1.0, f"Prepared {len(manifest_images)} images")
    return {"root": str(root), "manifest": str(root/"manifest.json"), "gt": str(gt), "images": len(manifest_images)}


def _poly_array(poly_json: str) -> np.ndarray:
    pts = np.asarray(json.loads(poly_json), dtype=np.float32).reshape(-1, 2)
    return cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)


def _det_intersects_spot(det, poly: np.ndarray, overlap_thr: float = 0.12) -> bool:
    cx = (float(det.x1) + float(det.x2)) / 2.0
    cy = (float(det.y1) + float(det.y2)) / 2.0
    bx = cx
    by = float(det.y2) - max(1.0, (float(det.y2) - float(det.y1)) * 0.08)
    contour = poly.reshape(-1, 1, 2)
    if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
        return True
    if cv2.pointPolygonTest(contour, (bx, by), False) >= 0:
        return True
    rect = np.asarray([
        [det.x1, det.y1], [det.x2, det.y1],
        [det.x2, det.y2], [det.x1, det.y2],
    ], dtype=np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(poly.astype(np.float32), rect)
        area = abs(float(cv2.contourArea(poly.reshape(-1, 1, 2))))
        return area > 1.0 and float(inter) / area >= float(overlap_thr)
    except Exception:
        return False


def _binary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return binary_metrics(df)


def evaluate_external_cnr(root: str | Path, settings: Dict, output_dir: str | Path,
                          progress=None) -> Dict:
    root = Path(root).resolve()
    gt_path = root / 'external_gt_spots.csv'
    if not gt_path.is_file():
        raise FileNotFoundError('Prepare the public dataset first: external_gt_spots.csv missing')
    raw_gt = pd.read_csv(gt_path, encoding='utf-8-sig')
    gt, input_audit = deduplicate_spots(raw_gt, 'evaluation_input_gt')
    if gt.empty:
        raise RuntimeError('External GT is empty')
    _progress(progress, 0, f"Annotation deduplication: {input_audit['duplicates_removed']} removed; {input_audit['unique_pairs']} unique pairs")
    cfg = dict(settings.get('detector', {}) or {})
    external = settings.get('external_validation', {}) or {}
    cfg.update(external.get('detector', {}) or {})
    baseline, ignored = freeze_baseline(root, cfg, external.get('spot_overlap_threshold', 0.12))
    if ignored:
        _progress(progress, 0, 'External Baseline is frozen; changed detector settings are ignored')
    cfg = dict(baseline['parameters']['detector'])
    overlap_thr = baseline['parameters']['spot_overlap_threshold']
    weights = Path(str(cfg.get('model', 'yolov8m.pt')))
    if baseline.get('model_sha256') and (not weights.is_file() or file_hash(weights)!=baseline['model_sha256']):
        raise ValueError('External Baseline model weights differ from the frozen fingerprint')
    detectors = {}; rows = []; detections = {}; skipped = []
    groups = list(gt.groupby('image_path', sort=True))
    for gi, (image_rel, spots) in enumerate(groups):
        image_path = (root/image_rel).resolve()
        if not image_path.is_relative_to(root): raise ValueError('Image escapes dataset root')
        image = cv2.imread(str(image_path))
        if image is None:
            skipped.append({'image_path':image_rel,'reason':'missing_or_unreadable_image'})
        else:
            camera = str(spots.iloc[0].get('camera', 'external'))
            if camera not in detectors: detectors[camera] = VehicleDetector(cfg)
            dets = detectors[camera].detect(image)
            detections[image_rel] = [d.as_dict() for d in dets]
            for row in spots.to_dict('records'):
                poly = _poly_array(str(row['polygon_json']))
                if len(poly)<3 or not np.isfinite(poly).all(): raise ValueError('Invalid annotation polygon')
                pred = int(any(_det_intersects_spot(d, poly, overlap_thr) for d in dets))
                rows.append({**row,'occupied_pred':pred,'correct':int(pred==int(row['occupied_gt'])),'detections_in_image':len(dets)})
        _progress(progress, .9*(gi+1)/max(1,len(groups)), f'External validation {gi+1}/{len(groups)}')
    if not rows: raise RuntimeError('No external images could be evaluated')
    if weights.is_file() and not baseline.get('model_sha256'):
        baseline['model_sha256'] = file_hash(weights)
        (root/'external_baseline.json').write_text(json.dumps(baseline,indent=2)+'\n',encoding='utf-8')
    preparation = root/'annotation_deduplication.json'
    audits = {'evaluation_input':input_audit,
              'annotation_conversion':json.loads(preparation.read_text(encoding='utf-8')) if preparation.is_file() else None}
    # GT hash is captured before evaluation; unique pair counts are computed, never hardcoded.
    baseline = {**baseline,'input_gt_sha256':file_hash(gt_path),'changed_settings_ignored':ignored}
    _progress(progress,.92,'Saving external metrics and screenshot evidence')
    result = write_results(pd.DataFrame(rows), output_dir, root, SOURCE_PAGE, baseline=baseline,
                           detections=detections, upstream_audit=audits, skipped=skipped,
                           screenshot_limit=external.get('screenshot_examples_per_class',3))
    _progress(progress,1.,'External validation complete')
    return result
