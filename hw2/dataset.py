from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .data import CLASSES, load_image

FILL = 114
class Boxes:
    @staticmethod
    def centre_inside(xywh, regions):
        if not len(regions) or not len(xywh):
            return np.zeros(len(xywh), bool)
        c = xywh[:, :2] + xywh[:, 2:] / 2
        r = regions[None]
        inside = ((c[:, None, 0] >= r[..., 0]) & (c[:, None, 0] <= r[..., 0] + r[..., 2]) & (c[:, None, 1] >= r[..., 1]) & (c[:, None, 1] <= r[..., 1] + r[..., 3]))
        return inside.any(1)

    @staticmethod
    def mask(img, regions, fill=FILL):
        for x, y, w, h in np.round(regions).astype(int):
            img[max(y, 0):y + h, max(x, 0):x + w] = fill
        return img

    @staticmethod
    def crop(xywh, cls, x0, y0, w, h, keep=0.4):
        x1 = np.clip(xywh[:, 0] - x0, 0, w)
        y1 = np.clip(xywh[:, 1] - y0, 0, h)
        x2 = np.clip(xywh[:, 0] + xywh[:, 2] - x0, 0, w)
        y2 = np.clip(xywh[:, 1] + xywh[:, 3] - y0, 0, h)
        area = (x2 - x1) * (y2 - y1)
        ok = (area >= keep * xywh[:, 2] * xywh[:, 3]) & (x2 - x1 >= 2) & (y2 - y1 >= 2)
        return np.stack([x1, y1, x2 - x1, y2 - y1], 1)[ok], cls[ok]

def resize_long(img, side):
    s = side / max(img.shape[:2])
    if s == 1:
        return img, s
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR), s


def prepare(pool, row, mask=True):
    img = load_image(row.path, pool.root)
    xywh, cls, ign = pool.targets(row.image_id)
    if mask and len(ign):
        Boxes.mask(img, ign)
        keep = ~Boxes.centre_inside(xywh, ign)
        xywh, cls = xywh[keep], cls[keep]
    return img, xywh, cls

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, pool, images, size=1536, crop=None, train=True, jitter=(0.75, 1.25), degrade=0.2,
                 flip=0.5, focus=0.7, color=0.2, seed=0):
        self.pool, self.images = pool, images.reset_index(drop=True)
        self.size, self.crop, self.train = size, crop, train
        self.jitter, self.degrade, self.flip, self.focus, self.color = jitter, degrade, flip, focus, color
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        row = self.images.iloc[i]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        img, xywh, cls = prepare(self.pool, row, mask=self.train)
        side = self.size * (rng.uniform(*self.jitter) if self.train else 1)
        img, s = resize_long(img, round(side))
        xywh = xywh * s
        if self.train:
            img, xywh, cls = self.augment(img, xywh, cls, rng)
        xyxy = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1).reshape(-1, 4)
        t = {"boxes": torch.from_numpy(xyxy.astype(np.float32)), "labels": torch.from_numpy(cls + 1),
             "image_id": int(row.image_id)}
        return torch.from_numpy(img).permute(2, 0, 1).float().div(255), t

    def augment(self, img, xywh, cls, rng):
        H, W = img.shape[:2]
        if rng.random() < self.degrade:
            k = rng.uniform(0.4, 0.7)
            small = cv2.resize(img, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.crop:
            cw, ch = min(self.crop, W), min(self.crop, H)
            if len(xywh) and rng.random() < self.focus:
                b = xywh[rng.integers(len(xywh))]
                cx = b[0] + b[2] / 2 + rng.uniform(-0.4, 0.4) * cw
                cy = b[1] + b[3] / 2 + rng.uniform(-0.4, 0.4) * ch
            else:
                cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            x0, y0 = int(np.clip(cx - cw / 2, 0, W - cw)), int(np.clip(cy - ch / 2, 0, H - ch))
            img = img[y0:y0 + ch, x0:x0 + cw]
            xywh, cls = Boxes.crop(xywh, cls, x0, y0, cw, ch)
        if rng.random() < self.flip:
            img = img[:, ::-1]
            xywh = xywh.copy()
            xywh[:, 0] = img.shape[1] - xywh[:, 0] - xywh[:, 2]
        if self.color:
            a, b = 1 + rng.uniform(-self.color, self.color), 255 * rng.uniform(-self.color, self.color) / 2
            img = cv2.convertScaleAbs(img, alpha=a, beta=b)
        return np.ascontiguousarray(img), xywh, cls


class FrameSampler(torch.utils.data.Sampler):
    def __init__(self, images, step=4, rare=0.0, seed=0):
        self.images, self.step, self.seed = images.reset_index(drop=True), step, seed
        self.repeat = np.ones(len(self.images))
        self.set_epoch(0)

    def set_repeat(self, pool, thr):
        b = pool.boxes[pool.boxes["image_id"].isin(self.images["image_id"])]
        has = b.groupby(["image_id", "cls"]).size().unstack(fill_value=0) > 0
        r = np.maximum(1, np.sqrt(thr / has.mean(0)))
        self.repeat = self.images["image_id"].map((has * r).max(1)).fillna(1).to_numpy()
        self.set_epoch(self.epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch
        rng = np.random.default_rng([self.seed, epoch])
        off = {c: rng.integers(self.step) for c in self.images["clip"].unique()}
        pick = ((self.images["frame"] + self.images["clip"].map(off)) % self.step == 0).to_numpy()
        n = np.floor(self.repeat + rng.random(len(self.repeat))).astype(int) * pick
        self.order = rng.permutation(np.repeat(np.arange(len(self.images)), n))

    def __iter__(self):
        return iter(self.order.tolist())

    def __len__(self):
        return len(self.order)


def collate(batch):
    return tuple(zip(*batch))


def loader(ds, batch, workers, sampler=None):
    return torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers,
                                       collate_fn=collate, pin_memory=True,
                                       multiprocessing_context="fork" if workers else None)

def iterate_frames(images, root, batch=4, workers=4):
    paths = images["path"].tolist()
    chunks = [list(range(s, min(s + batch, len(paths)))) for s in range(0, len(paths), batch)]
    with ThreadPoolExecutor(max(workers, 1)) as ex:
        submit = lambda idx: [ex.submit(load_image, paths[i], root) for i in idx]
        pending = submit(chunks[0]) if chunks else []
        for k, idx in enumerate(chunks):
            frames = [f.result() for f in pending]
            if k + 1 < len(chunks):
                pending = submit(chunks[k + 1])
            yield idx, frames

def tiles(W, H, tile, overlap):
    def starts(n):
        if n <= tile:
            return [0]
        k = int(np.ceil((n - tile) / (tile * (1 - overlap)))) + 1
        return np.linspace(0, n - tile, k).round().astype(int).tolist()
    return [(x, y) for y in starts(H) for x in starts(W)]

def export_yolo(pool, parts, out, size=1536, tile=0, overlap=0.2, workers=6):
    out = Path(out)

    def write(name, row):
        img, xywh, cls = prepare(pool, row)
        img, s = resize_long(img, size)
        xywh = xywh * s
        H, W = img.shape[:2]
        crops = tiles(W, H, tile, overlap) if tile else [(0, 0)]
        for x0, y0 in crops:
            cw, ch = (min(tile, W), min(tile, H)) if tile else (W, H)
            b, c = Boxes.crop(xywh, cls, x0, y0, cw, ch)
            key = row.key if not tile else f"{row.key}_{y0}_{x0}"
            cv2.imwrite(str(out / "images" / name / f"{key}.jpg"), img[y0:y0 + ch, x0:x0 + cw, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = [f"{k} {(x + w / 2) / cw:.6f} {(y + h / 2) / ch:.6f} {w / cw:.6f} {h / ch:.6f}"
                     for k, (x, y, w, h) in zip(c, b)]
            (out / "labels" / name / f"{key}.txt").write_text("\n".join(lines))
        return len(crops)

    n = {}
    for name, images in parts.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(workers) as ex:
            n[name] = sum(ex.map(lambda r: write(name, r), images.itertuples()))
        print(f"{name}: {len(images)} frames, {n[name]} images")
    cfg = {"path": str(out.resolve()), **{k: f"images/{k}" for k in parts}, "names": dict(enumerate(CLASSES))}
    cfg.setdefault("val", "images/train")
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out / "data.yaml"
