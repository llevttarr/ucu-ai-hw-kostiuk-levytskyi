import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/kagglehub/competitions/multiple-object-video-detection-hw-2-ucu-ai-2026"
CACHE = ROOT / "data/cache"
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


def load_image(path, root=DATA):
    return cv2.imread(str(Path(root) / path))[..., ::-1]

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
