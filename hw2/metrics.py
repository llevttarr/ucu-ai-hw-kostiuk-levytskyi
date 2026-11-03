import contextlib
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import CLASSES
from .dataset import iterate_frames


class Evaluator:
    """Coco map on pool frames, ignored regions as iscrowd boxes of every class like the competition json."""

    def __init__(self, pool, images):
        from pycocotools.coco import COCO
        self.images = images.reset_index(drop=True)
        ids = set(self.images["image_id"])
        b = pool.boxes[pool.boxes["image_id"].isin(ids)]
        g = pool.ignore[pool.ignore["image_id"].isin(ids)]
        anns = [{"image_id": int(i), "category_id": int(c), "bbox": [float(x), float(y), float(w), float(h)],
                 "area": float(w * h), "iscrowd": 0}
                for i, c, x, y, w, h in zip(b["image_id"], b["cls"], b["x"], b["y"], b["w"], b["h"])]
        anns += [{"image_id": int(i), "category_id": c, "bbox": [float(x), float(y), float(w), float(h)],
                  "area": float(w * h), "iscrowd": 1}
                 for i, x, y, w, h in zip(g["image_id"], g["x"], g["y"], g["w"], g["h"])
                 for c in range(len(CLASSES))]
        for k, a in enumerate(anns):
            a["id"] = k + 1
        self.gt = COCO()
        im = self.images
        self.gt.dataset = {
            "images": [{"id": int(i), "width": int(w), "height": int(h)}
                       for i, w, h in zip(im["image_id"], im["width"], im["height"])],
            "annotations": anns, "categories": [{"id": k, "name": n} for k, n in enumerate(CLASSES)]}
        with contextlib.redirect_stdout(io.StringIO()):
            self.gt.createIndex()

    def score(self, preds):
        """Dict of mAP, AP50, AP75, AP_small/medium/large, AR500 and per-class AP from {image_id: (N, 6)}."""
        from pycocotools.cocoeval import COCOeval
        rows = [np.column_stack([np.full(len(p), i), p[:, :5], p[:, 5]]) for i, p in preds.items() if len(p)]
        if not rows:
            return {"mAP": 0.0}
        with contextlib.redirect_stdout(io.StringIO()):
            dt = self.gt.loadRes(np.concatenate(rows))
            ev = COCOeval(self.gt, dt, "bbox")
            ev.params.imgIds = [int(i) for i in self.images["image_id"]]
            ev.params.maxDets = [1, 100, 500]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = ev.stats
        res = {"mAP": s[0], "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5],
               "AR500": s[8]}
        pr = ev.eval["precision"][:, :, :, 0, -1]
        for k, n in enumerate(CLASSES):
            p = pr[:, :, k]
            res[f"AP_{n}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
        return {k: round(float(v), 4) for k, v in res.items()}


def predict_frames(det, images, root, batch=4, workers=4, downscale=1.0):
    """Predictions {image_id: (N, 6)} and model-only fps, on frames optionally shrunk by downscale."""
    preds, spent = {}, 0.0
    for idx, frames in iterate_frames(images, root, batch, workers):
        if downscale != 1:
            frames = [cv2.resize(f, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
                      for f in frames]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time()
        out = det.predict(list(frames))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        spent += time.time() - t
        for i, p in zip(idx, out):
            if downscale != 1:
                p = p.copy()
                p[:, :4] /= downscale
            preds[images.at[i, "image_id"]] = p
    return preds, len(images) / max(spent, 1e-9)


def by_window(images, preds, fn, col="window"):
    """Applies fn(list of (N, 6) in frame order) -> list to each clip or window of the predictions."""
    out = dict(preds)
    for _, g in images.sort_values([col, "frame"]).groupby(col):
        ids = g["image_id"].tolist()
        for i, p in zip(ids, fn([preds[i] for i in ids])):
            out[i] = p
    return out


def clip_boxes(p, W, H):
    """(N, 6) boxes clipped to the frame, degenerate ones dropped."""
    x1, y1 = np.clip(p[:, 0], 0, W), np.clip(p[:, 1], 0, H)
    x2, y2 = np.clip(p[:, 0] + p[:, 2], 0, W), np.clip(p[:, 1] + p[:, 3], 0, H)
    q = np.column_stack([x1, y1, x2 - x1, y2 - y1, p[:, 4:]])
    return q[(q[:, 2] > 1) & (q[:, 3] > 1)]


def write_submission(path, test, preds, top=500):
    """Competition csv: image_id and space-separated "class score x y w h" groups, best top per image."""
    rows = []
    for r in test.itertuples():
        p = clip_boxes(preds.get(r.image_id, np.zeros((0, 6))), r.width, r.height)
        p = p[np.argsort(-p[:, 4])][:top]
        rows.append(" ".join(f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}" for x, y, w, h, s, c in p))
    sub = pd.DataFrame({"image_id": test["image_id"], "PredictionString": rows})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"wrote {path}: {len(sub)} images, {sum(len(s.split()) // 6 for s in rows)} boxes")
    return sub


def log_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def evaluate(det, pool, images, temporal=None, batch=4, workers=4, downscale=1.0):
    """Scores, model frames per second and predictions of a detector on pool frames, optionally rescored."""
    preds, fps = predict_frames(det, images, pool.root, batch, workers, downscale)
    if temporal is not None:
        preds = by_window(images, preds, temporal)
    return Evaluator(pool, images).score(preds), fps, preds
