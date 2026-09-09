"""
riskam.feature_cache

Disk cache for the parameter-independent outputs of the inference path.

The expensive part of a per-frame featextr pass — YOLO/pose detection,
ByteTrack matching, and depth extraction at each bounding box — is
deterministic given the (model, frame) pair. Re-running it for every cell
of a sweep that only varies scoring parameters (weights, gaze sigmas,
algorithm choice) is wasted compute.

This module records those primitives once per (model, dataset, run, frame)
and re-uses them for subsequent passes. Cells that change weights/sigmas/
algorithm hit cache; cells that change the model file get a fresh SHA and
miss the cache by design.

Important assumption — sequential processing
--------------------------------------------
ByteTrack track IDs are coupled to the temporal sequence of preceding
detections. The cache stores per-frame track IDs computed during a
sequential pass over a run's frames in sorted order. Mixing cache hits and
misses *within the same run* desynchronises ByteTrack's internal state from
the cached IDs. The cache is therefore most useful when a full uncached
pass over a run populates the cache first; subsequent sweep cells then hit
on every frame and stay consistent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CachedFrameFeatures:
    """Per-frame primitives that are deterministic given (model, frame).

    Subscores are *not* cached — they depend on tunable parameters (weights,
    sigmas, gaze algorithm) and recompute from these primitives in
    microseconds.
    """

    human_bboxes: list                   # list of [x1, y1, x2, y2]
    keypoints_np: np.ndarray | None      # (N, 17, 2) float, or None
    track_ids: list                      # list of int | None
    bbox_depths_m: list                  # depth in metres per bbox
    depth_viz: np.ndarray                # (H, W) uint8 visualisation
    # Per-person eye-region texture (facegate.measure_face_texture); NaN =
    # no face triple. None = cached before the field existed (the gaze gate
    # then degrades to geometry-only; refresh via
    # scripts/add_face_texture_to_cache.py).
    face_texture_np: np.ndarray | None = None

    def to_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # ``np.savez_compressed`` requires arrays — use sentinels for None.
        kpts_present = self.keypoints_np is not None
        kpts = (
            self.keypoints_np
            if kpts_present
            else np.zeros((0,), dtype=np.float32)
        )
        track_arr = np.array(
            [(-1 if t is None else int(t)) for t in self.track_ids],
            dtype=np.int64,
        )
        texture_present = self.face_texture_np is not None
        texture = (
            self.face_texture_np
            if texture_present
            else np.zeros((0,), dtype=np.float32)
        )
        np.savez_compressed(
            path,
            bboxes=np.asarray(self.human_bboxes, dtype=np.float32).reshape(-1, 4),
            keypoints=kpts,
            keypoints_present=np.array([kpts_present]),
            track_ids=track_arr,
            bbox_depths_m=np.asarray(self.bbox_depths_m, dtype=np.float32),
            depth_viz=self.depth_viz,
            face_texture=texture,
            face_texture_present=np.array([texture_present]),
        )

    @classmethod
    def from_npz(cls, path: Path) -> "CachedFrameFeatures":
        data = np.load(path)
        bboxes = data["bboxes"].tolist()
        kpts_present = bool(data["keypoints_present"][0])
        keypoints = (
            data["keypoints"]
            if kpts_present and data["keypoints"].size > 0
            else None
        )
        track_ids = [
            (None if int(t) == -1 else int(t)) for t in data["track_ids"]
        ]
        bbox_depths_m = data["bbox_depths_m"].tolist()
        depth_viz = data["depth_viz"]
        # Backward-compatible: caches written before the face-texture field
        # simply lack the keys — load as None (geometry-only gating).
        face_texture = None
        if "face_texture_present" in data.files and bool(
            data["face_texture_present"][0]
        ):
            face_texture = data["face_texture"]
        return cls(
            human_bboxes=bboxes,
            keypoints_np=keypoints,
            track_ids=track_ids,
            bbox_depths_m=bbox_depths_m,
            depth_viz=depth_viz,
            face_texture_np=face_texture,
        )


def model_sha(model_path: Path) -> str:
    """Short SHA-256 prefix of the model file. Used as part of the cache key
    so a model swap invalidates everything below it on disk."""
    h = hashlib.sha256()
    with model_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class FeatureCache:
    """Read-through, write-back per-frame cache.

    Layout::
        <root>/<dataset>/<run>/<model_sha>/<rgb_stem>.npz

    The model SHA sits at the bottom of the path so swapping models lives
    in a sibling directory rather than overwriting the previous cache.
    """

    def __init__(self, root: Path, model_sha_str: str, dataset: str):
        self.root = Path(root)
        self.model_sha = model_sha_str
        self.dataset = dataset
        self.hits = 0
        self.misses = 0

    def _path(self, run: str, rgb_stem: str) -> Path:
        return (
            self.root
            / self.dataset
            / run
            / self.model_sha
            / f"{rgb_stem}.npz"
        )

    def get(self, run: str, rgb_stem: str) -> CachedFrameFeatures | None:
        p = self._path(run, rgb_stem)
        if not p.is_file():
            self.misses += 1
            return None
        self.hits += 1
        return CachedFrameFeatures.from_npz(p)

    def put(
        self, run: str, rgb_stem: str, features: CachedFrameFeatures
    ) -> None:
        features.to_npz(self._path(run, rgb_stem))

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
