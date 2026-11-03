from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import export_yolo, resize_long

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = Path(__file__).resolve().parents[1] / "runs/weights"


class Anchors:
    @staticmethod
    def default(kind):
        """Torchvision default sizes per level and h/w ratios."""
        if kind == "frcnn":
            return tuple((s,) for s in (32, 64, 128, 256, 512)), (0.5, 1.0, 2.0)
        sizes = tuple(tuple(s * 2 ** (k / 3) for k in range(3)) for s in (32, 64, 128, 256, 512))
        return sizes, (0.5, 1.0, 2.0)

    @staticmethod
    def fit(pool, images, size, levels=5, per_level=1, q=0.05):
        """Data anchors at training scale: q quantile of sqrt(wh) doubling per level, h/w at 15/50/85%."""
        b = pool.boxes[pool.boxes["image_id"].isin(images["image_id"])]
        s = size / np.maximum(b["width"], b["height"]).to_numpy()
        base = np.quantile(b["size"].to_numpy() * s, q)
        sizes = tuple(tuple(round(float(base) * 2 ** (lv + k / per_level), 1) for k in range(per_level))
                      for lv in range(levels))
        ratios = tuple(np.quantile((b["h"] / b["w"]).to_numpy(), [0.15, 0.5, 0.85]).round(2).tolist())
        return sizes, ratios


class Detector:
    """Base detector: predict(list of rgb frames) -> list of (N, 6) [x, y, w, h, score, cls] in frame pixels."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.size = cfg["size"]

    def fit(self, pool, train_images, val_images, out):
        raise NotImplementedError

    def predict(self, frames):
        raise NotImplementedError

    def save(self, out):
        raise NotImplementedError

    def load(self, run):
        raise NotImplementedError


def no_resize(image, target=None):
    return image, target


class TorchDetector(Detector):
    """Torchvision detector, frames resized to the long side by us, the model's own resize turned off."""

    def __init__(self, cfg, pool=None, images=None):
        super().__init__(cfg)
        self.anchors = None
        if cfg.get("anchors", "default") == "fit" and pool is not None:
            self.anchors = Anchors.fit(pool, images, self.size, **self.anchor_args())
            cfg["anchor_sizes"] = [list(a) for a in self.anchors[0]]
            cfg["anchor_ratios"] = list(self.anchors[1])
        elif "anchor_sizes" in cfg:
            self.anchors = tuple(tuple(a) for a in cfg["anchor_sizes"]), tuple(cfg["anchor_ratios"])
        self.model = self.build()
        self.model.transform.resize = no_resize
        self.model.to(DEVICE)

    def anchor_args(self):
        return {}

    def build(self):
        raise NotImplementedError

    def fit(self, pool, train_images, val_images, out):
        from .train import Trainer
        return Trainer(self, pool, train_images, val_images, out, self.cfg).run()

    @torch.no_grad()
    def predict(self, frames):
        self.model.eval()
        ims, scales = zip(*(resize_long(f, self.size) for f in frames))
        x = [torch.from_numpy(im).to(DEVICE).permute(2, 0, 1).float().div(255) for im in ims]
        with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
            out = self.model(x)
        res = []
        for o, s in zip(out, scales):
            o = {k: v[o["labels"] > 0] for k, v in o.items()}
            b = o["boxes"].float().cpu().numpy() / s
            sc, k = o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy() - 1
            res.append(np.column_stack([b[:, :2], b[:, 2:] - b[:, :2], sc, k]))
        return res

    def save(self, out):
        torch.save(self.model.state_dict(), Path(out) / "best.pt")

    def load(self, run):
        self.model.load_state_dict(torch.load(Path(run) / "best.pt", map_location=DEVICE))
        return self


class FasterRCNN(TorchDetector):
    def build(self):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.rpn import RPNHead

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if self.cfg.get("pretrained", True) else None
        m = fasterrcnn_resnet50_fpn_v2(weights=weights, **self.cfg.get("params", {}))
        m.roi_heads.box_predictor = FastRCNNPredictor(1024, len(self.cfg["classes"]) + 1)
        if self.anchors:
            sizes, ratios = self.anchors
            m.rpn.anchor_generator = AnchorGenerator(sizes, (ratios,) * len(sizes))
            n = m.rpn.anchor_generator.num_anchors_per_location()[0]
            if n != m.rpn.head.cls_logits.out_channels:
                head = RPNHead(256, n, conv_depth=2)
                head.conv.load_state_dict(m.rpn.head.conv.state_dict())
                m.rpn.head = head
        return m


class RetinaNet(TorchDetector):
    def anchor_args(self):
        return {"per_level": 3}

    def build(self):
        from torchvision.models.detection import (RetinaNet_ResNet50_FPN_V2_Weights, retinanet,
                                                  retinanet_resnet50_fpn_v2)
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.ops import sigmoid_focal_loss

        retinanet.sigmoid_focal_loss = partial(sigmoid_focal_loss, alpha=self.cfg.get("focal_alpha", 0.25),
                                               gamma=self.cfg.get("focal_gamma", 2.0))
        weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if self.cfg.get("pretrained", True) else None
        m = retinanet_resnet50_fpn_v2(weights=weights, **self.cfg.get("params", {}))
        if self.anchors:
            sizes, ratios = self.anchors
            m.anchor_generator = AnchorGenerator(sizes, (ratios,) * len(sizes))
        n, k = m.anchor_generator.num_anchors_per_location()[0], len(self.cfg["classes"]) + 1
        norm = partial(torch.nn.GroupNorm, 32)
        cls = retinanet.RetinaNetClassificationHead(256, n, k, norm_layer=norm)
        cls.conv.load_state_dict(m.head.classification_head.conv.state_dict())
        m.head.classification_head = cls
        if n != m.head.regression_head.bbox_reg.out_channels // 4:
            reg = retinanet.RetinaNetRegressionHead(256, n, norm_layer=norm)
            reg.conv.load_state_dict(m.head.regression_head.conv.state_dict())
            reg._loss_type = "giou"
            m.head.regression_head = reg
        return m


class UltraDetector(Detector):
    """Ultralytics yolo or rt-detr, trained on an exported dataset with ignored regions greyed out."""

    def __init__(self, cfg, pool=None, images=None):
        super().__init__(cfg)
        self.model = self.make()

    def make(self, path=None):
        raise NotImplementedError

    def weights(self, name):
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        return str(WEIGHTS / name)

    def dataset(self, pool, train_images, val_images):
        c = self.cfg
        name = f"s{c['export_size']}_t{c.get('tile', 0)}_k{c['step']}" + ("_full" if val_images is None else "")
        out = Path(c.get("dataset_root", "data/yolo")) / name
        if not (out / "data.yaml").exists():
            parts = {"train": train_images[(train_images["frame"] - 1) % c["step"] == 0]}
            if val_images is not None:
                parts["val"] = val_images
            export_yolo(pool, parts, out, size=c["export_size"], tile=c.get("tile", 0))
        return out / "data.yaml"

    def fit(self, pool, train_images, val_images, out):
        c, out = self.cfg, Path(out)
        data = self.dataset(pool, train_images, val_images)
        self.model.train(data=str(data), epochs=c["epochs"], imgsz=c["train_size"], batch=c["batch"],
                         project=str(out.parent.resolve()), name=out.name, exist_ok=True,
                         workers=c.get("workers", 4), seed=c.get("seed", 0), deterministic=False,
                         val=val_images is not None, max_det=500, **c.get("train", {}))
        self.save(out)

    @torch.no_grad()
    def predict(self, frames):
        c = self.cfg
        rs = self.model.predict([f[..., ::-1] for f in frames], imgsz=self.size, conf=c.get("conf", 0.001),
                                iou=c.get("nms", 0.7), max_det=500, half=DEVICE == "cuda",
                                augment=c.get("tta", False), verbose=False)
        res = []
        for r in rs:
            b = r.boxes.xyxy.cpu().numpy()
            res.append(np.concatenate([b[:, :2], b[:, 2:] - b[:, :2], r.boxes.conf.cpu().numpy()[:, None],
                                       r.boxes.cls.cpu().numpy()[:, None]], 1))
        return res

    def save(self, out):
        out = Path(out)
        best = out / "weights/best.pt"
        if not best.exists():
            best = out / "weights/last.pt"
        (out / "best.pt").write_bytes(best.read_bytes())

    def load(self, run):
        self.model = self.make(Path(run) / "best.pt")
        return self


class YOLO(UltraDetector):
    def make(self, path=None):
        from ultralytics import YOLO as Y
        if path:
            return Y(str(path))
        m = Y(self.cfg["arch"]) if self.cfg.get("arch") else Y(self.weights(self.cfg["weights"]))
        return m.load(self.weights(self.cfg["weights"])) if self.cfg.get("arch") else m


class RTDETR(UltraDetector):
    def make(self, path=None):
        from ultralytics import RTDETR as R
        return R(str(path) if path else self.weights(self.cfg["weights"]))


class Ensemble(Detector):
    """Several trained runs at several sizes, with optional flip, fused by weighted box fusion."""

    def __init__(self, cfg, pool=None, images=None):
        super().__init__(cfg)
        self.members = [(load_run(m["run"]), m.get("sizes"), m.get("flip", False), m.get("weight", 1.0))
                        for m in cfg["members"]]

    def fit(self, pool, train_images, val_images, out):
        pass

    def predict(self, frames):
        from .post import wbf
        outs, ws = [], []
        for det, sizes, flip, w in self.members:
            for size in sizes or [det.size]:
                det.size = size
                outs.append(det.predict(frames))
                ws.append(w)
                if flip:
                    p = det.predict([np.ascontiguousarray(f[:, ::-1]) for f in frames])
                    for d, f in zip(p, frames):
                        d[:, 0] = f.shape[1] - d[:, 0] - d[:, 2]
                    outs.append(p)
                    ws.append(w)
        fused = [wbf([o[i] for o in outs], ws, iou=self.cfg.get("fuse_iou", 0.6)) for i in range(len(frames))]
        return [f[np.argsort(-f[:, 4])[:500]] for f in fused]

    def save(self, out):
        pass

    def load(self, run):
        return self


MODELS = {"frcnn": FasterRCNN, "retinanet": RetinaNet, "yolo": YOLO, "rtdetr": RTDETR, "ensemble": Ensemble}


def build(cfg, pool=None, images=None):
    return MODELS[cfg["model"]](cfg, pool, images)


def load_run(run, **override):
    """Trained detector from a run folder with its config.yaml and best.pt."""
    cfg = yaml.safe_load((Path(run) / "config.yaml").read_text())
    cfg.update(override)
    if cfg["model"] in ("frcnn", "retinanet"):
        cfg["pretrained"] = False
    return build(cfg).load(run)


def parse_set(items):
    """KEY=VALUE pairs to a nested dict, dotted keys nest and values are parsed as yaml."""
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        d = out
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(v)
    return out


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, sets=()):
    return merge(yaml.safe_load(Path(path).read_text()), parse_set(sets))


def detector(path, sets=()):
    """A trained run folder with overrides, or a yaml config for a model that needs no training."""
    path = Path(path)
    if path.is_dir():
        return load_run(path, **parse_set(sets))
    return build(load_config(path, sets))
