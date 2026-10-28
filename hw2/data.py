import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get(
    "HW2_DATA", ROOT / "data/kagglehub/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026"))
CACHE = ROOT / "data/cache"
HOLDOUT = ROOT / "splits/holdout.json"
CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
VISDRONE = ["ignored", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
            "awning-tricycle", "bus", "motor", "others"]

class Split:
    def __init__(self, name, root=DATA):
        self.name = name
        self.root = Path(root)
        cache = CACHE / f"{name}.pkl"
        if cache.exists():
            self.images, self.boxes, self.ignore = pd.read_pickle(cache)
        else:
            self.images, self.boxes, self.ignore = self._parse()
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle((self.images, self.boxes, self.ignore), cache)

    def _parse(self):
        with open(self.root / self.name / "annotations_coco.json") as f:
            d = json.load(f)
        img = pd.DataFrame(d["images"]).rename(
            columns={"id": "image_id", "video": "clip", "frame_id": "frame"})
        img["path"] = self.name + "/" + img["file_name"]
        img = img[["image_id", "clip", "frame", "width", "height", "path"]]

        anns = d.pop("annotations")
        del d
        ann = pd.DataFrame({
            "image_id": np.array([a["image_id"] for a in anns], dtype=np.int32),
            "cls": np.array([a["category_id"] - 1 for a in anns], dtype=np.int8),
            "crowd": np.array([a["iscrowd"] for a in anns], dtype=bool),
            "orig": np.array([a["orig_category_id"] for a in anns], dtype=np.int8),
            "occlusion": np.array([a.get("occlusion", -1) for a in anns], dtype=np.int8),
            "truncation": np.array([a.get("truncation", -1) for a in anns], dtype=np.int8),
            "track_id": np.array([a.get("track_id", -1) for a in anns], dtype=np.int32),
        })
        ann[["x", "y", "w", "h"]] = np.array([a["bbox"] for a in anns], dtype=np.float32)
        del anns
        ann = ann.merge(img[["image_id", "clip", "frame", "width", "height"]], on="image_id")

        ign = ann[ann["crowd"]].drop_duplicates(["image_id", "x", "y", "w", "h"])
        ign = ign[["image_id", "clip", "frame", "x", "y", "w", "h", "width", "height", "orig"]]
        box = ann[~ann["crowd"]].drop(columns="crowd")
        box["size"] = np.sqrt(box["w"] * box["h"])
        box["cx"] = (box["x"] + box["w"] / 2) / box["width"]
        box["cy"] = (box["y"] + box["h"] / 2) / box["height"]

        img["n_obj"] = img["image_id"].map(box["image_id"].value_counts()).fillna(0).astype(int)
        img["n_ignore"] = img["image_id"].map(ign["image_id"].value_counts()).fillna(0).astype(int)
        return img, box.reset_index(drop=True), ign.reset_index(drop=True)

    @property
    def clips(self):
        return sorted(self.images["clip"].unique())

    def frame(self, image_id):
        return load_image(self.images.set_index("image_id").at[image_id, "path"], self.root)
    def frame_boxes(self, image_id):
        b, ign = self.boxes, self.ignore
        return b[b["image_id"] == image_id], ign[ign["image_id"] == image_id]


class Pool:
    def __init__(self, names=("train", "val"), root=DATA):
        self.root = Path(root)
        ims, bxs, igs, off = [], [], [], 0
        for n in names:
            s = Split(n, root)
            ids = dict(zip(s.images["image_id"], range(off, off + len(s.images))))
            ims.append(s.images.assign(image_id=s.images["image_id"].map(ids), split=n))
            bxs.append(s.boxes.assign(image_id=s.boxes["image_id"].map(ids)))
            igs.append(s.ignore.assign(image_id=s.ignore["image_id"].map(ids)))
            off += len(s.images)
        self.images = pd.concat(ims, ignore_index=True)
        self.images["key"] = self.images["clip"] + "_" + self.images["frame"].map("{:07d}".format)
        self.boxes = pd.concat(bxs, ignore_index=True)
        self.ignore = pd.concat(igs, ignore_index=True)
        self._xywh = self.boxes[["x", "y", "w", "h"]].to_numpy(np.float32)
        self._cls = self.boxes["cls"].to_numpy(np.int64)
        self._ign = self.ignore[["x", "y", "w", "h"]].to_numpy(np.float32)
        self._bi = self.boxes.groupby("image_id").indices
        self._ii = self.ignore.groupby("image_id").indices

    @property
    def clips(self):
        return sorted(self.images["clip"].unique())

    def targets(self, image_id):
        b, i = self._bi.get(image_id, []), self._ii.get(image_id, [])
        return self._xywh[b], self._cls[b], self._ign[i]

    def frames(self, clips, step=1):
        im = self.images[self.images["clip"].isin(clips)]
        return im[(im["frame"] - 1) % step == 0].reset_index(drop=True)

    def windows(self, clips, length=80, every=4):
        im = self.images[self.images["clip"].isin(clips)].sort_values(["clip", "frame"])
        run = im.groupby("clip").cumcount() // length
        im = im.assign(window=im["clip"] + "/" + run.astype(str))
        return im[run % every == 0].reset_index(drop=True)


def make_holdout(pool, n_val=10, draws=20000, frac=0.15, seed=0):
    b = pool.boxes
    cls = pd.crosstab(b["clip"], b["cls"]).reindex(columns=range(len(CLASSES)), fill_value=0)
    size = pd.crosstab(b["clip"], pd.cut(b["size"], [0, 16, 32, 96, np.inf]))
    frames = pool.images.groupby("clip").size()
    clips = np.array(cls.index)
    C, S, F = cls.to_numpy(float), size.loc[clips].to_numpy(float), frames.loc[clips].to_numpy(float)
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(draws):
        v = np.zeros(len(clips), bool)
        v[rng.choice(len(clips), n_val, replace=False)] = True
        if (C[v].sum(0) < 100).any():
            continue
        share = lambda a, m: a[m].sum(0) / a[m].sum()
        score = (np.abs(share(C, v) - share(C, ~v)).sum() + np.abs(share(S, v) - share(S, ~v)).sum()
                 + 2 * abs(F[v].sum() / F.sum() - frac))
        if score < best[0]:
            best = (score, v)
    v = best[1]
    return {"val": sorted(clips[v]), "train": sorted(clips[~v]), "score": round(float(best[0]), 4),
            "seed": seed}


def load_holdout(path=HOLDOUT):
    with open(path) as f:
        d = json.load(f)
    assert not set(d["train"]) & set(d["val"]), "val clips leak into train"
    return d["train"], d["val"]


def load_image(path, root=DATA):
    return np.ascontiguousarray(cv2.imread(str(Path(root) / path))[..., ::-1])

def read_yolo(path, width, height):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5))
    a = np.loadtxt(path, ndmin=2).reshape(-1, 5)
    a[:, 1:] *= [width, height, width, height]
    a[:, 1:3] -= a[:, 3:5] / 2
    return a
def test_images(root=DATA):
    df = pd.read_csv(Path(root) / "test_images.csv")
    return df.rename(columns={"file_name": "path"})
