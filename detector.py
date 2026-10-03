# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import numpy as np

from utils import box_iou


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def box(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_dict(self) -> Dict[str, float | int]:
        return {
            'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2,
            'conf': self.conf, 'cls': self.cls,
        }


class VehicleDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = str(config.get('model', 'yolov8m.pt'))
        self.imgsz = int(config.get('imgsz', 1280))
        self.conf = float(config.get('conf', 0.15))
        self.tiling = bool(config.get('tiling', False))
        self.tile_overlap = float(config.get('tile_overlap', 0.20))
        self.nms_iou = float(config.get('tile_nms_iou', 0.50))
        self.physical_nms_iou = float(config.get('physical_vehicle_nms_iou', 0.42))
        self.physical_overlap_min = float(config.get('physical_vehicle_overlap_min', 0.68))
        self.vehicle_classes = set(int(x) for x in config.get('vehicle_class_ids', [2, 3, 5, 7]))
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)

    def _infer_single(self, image: np.ndarray, conf: float | None = None) -> List[Detection]:
        self._load()
        threshold = self.conf if conf is None else float(conf)
        results = self._model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=threshold,
            classes=sorted(self.vehicle_classes),
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        for b in boxes:
            xyxy = b.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            c = float(b.conf[0].detach().cpu().item())
            cls = int(b.cls[0].detach().cpu().item())
            if cls not in self.vehicle_classes:
                continue
            out.append(Detection(*xyxy, c, cls))
        return out

    def detect_raw(self, image: np.ndarray) -> List[Detection]:
        if not self.tiling:
            return self._infer_single(image)
        return self._detect_tiled(image)

    def detect(self, image: np.ndarray) -> List[Detection]:
        # v9: a physical vehicle must appear only once before slot assignment.
        # COCO can occasionally emit overlapping car/truck/bus labels for one vehicle;
        # class-aware NMS alone does not remove those cross-class duplicates.
        raw = self.detect_raw(image)
        return physical_vehicle_nms(raw, self.physical_nms_iou, self.physical_overlap_min)

    def detect_batch(self, images: Sequence[np.ndarray]) -> List[List[Detection]]:
        """Detect a batch of images with one model call when tiling is disabled.

        Slot-crop inference can involve dozens of small images per CCTV frame.
        Batching them is substantially cheaper than invoking YOLO once per slot.
        """
        imgs = list(images)
        if not imgs:
            return []
        if self.tiling:
            return [self.detect(img) for img in imgs]
        self._load()
        results = self._model.predict(
            source=imgs,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=sorted(self.vehicle_classes),
            verbose=False,
        )
        out: List[List[Detection]] = []
        for res in results:
            dets: List[Detection] = []
            boxes = getattr(res, 'boxes', None)
            if boxes is not None:
                for b in boxes:
                    xyxy = b.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                    c = float(b.conf[0].detach().cpu().item())
                    cls = int(b.cls[0].detach().cpu().item())
                    if cls in self.vehicle_classes:
                        dets.append(Detection(*xyxy, c, cls))
            out.append(physical_vehicle_nms(dets, self.physical_nms_iou, self.physical_overlap_min))
        # Ultralytics normally returns one result per source image. Keep a stable
        # contract even if an unusual backend returns fewer items.
        while len(out) < len(imgs):
            out.append([])
        return out[:len(imgs)]

    def _detect_tiled(self, image: np.ndarray) -> List[Detection]:
        h, w = image.shape[:2]
        overlap = max(0.0, min(0.45, self.tile_overlap))
        tw = int(round(w / (2.0 - overlap)))
        th = int(round(h / (2.0 - overlap)))
        x_positions = [0, max(0, w - tw)]
        y_positions = [0, max(0, h - th)]
        dets: List[Detection] = []
        for y in y_positions:
            for x in x_positions:
                tile = image[y:y+th, x:x+tw]
                if tile.size == 0:
                    continue
                for d in self._infer_single(tile):
                    dets.append(Detection(
                        d.x1 + x, d.y1 + y, d.x2 + x, d.y2 + y, d.conf, d.cls
                    ))
        # First clean duplicate tile detections within class. Cross-class physical
        # dedup is applied once more by detect().
        return class_aware_nms(dets, self.nms_iou)


def _intersection_over_smaller(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = min(aa, ba)
    return inter / denom if denom > 0 else 0.0


def physical_vehicle_nms(
    dets: Sequence[Detection], iou_thr: float = 0.42, overlap_min_thr: float = 0.68
) -> List[Detection]:
    """Class-agnostic deduplication for physical vehicles.

    A single car can be emitted as overlapping COCO classes (e.g. car/truck),
    especially with low-resolution re-recorded CCTV. Keep only the strongest box
    when boxes have high IoU OR one box substantially covers the other.
    """
    items = sorted(dets, key=lambda z: (z.conf, z.area), reverse=True)
    kept: List[Detection] = []
    while items:
        best = items.pop(0)
        kept.append(best)
        new_items = []
        for x in items:
            same_physical = (
                box_iou(best.box, x.box) >= iou_thr
                or _intersection_over_smaller(best.box, x.box) >= overlap_min_thr
            )
            if not same_physical:
                new_items.append(x)
        items = new_items
    return kept


def class_aware_nms(dets: Sequence[Detection], iou_thr: float = 0.5) -> List[Detection]:
    by_cls: Dict[int, List[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls, []).append(d)
    kept: List[Detection] = []
    for _, items in by_cls.items():
        items = sorted(items, key=lambda z: z.conf, reverse=True)
        while items:
            best = items.pop(0)
            kept.append(best)
            items = [x for x in items if box_iou(best.box, x.box) < iou_thr]
    return kept


def draw_detections(image: np.ndarray, dets: Sequence[Detection]) -> np.ndarray:
    out = image.copy()
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, (0, 255, 255), 2)
        cx, cy = map(int, d.center)
        cv2.circle(out, (cx, cy), 3, (255, 0, 255), -1)
        cv2.putText(out, f'{d.conf:.2f}', (p1[0], max(15, p1[1]-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return out
