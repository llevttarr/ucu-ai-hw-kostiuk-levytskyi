import math
import time
from pathlib import Path

import torch

from .dataset import FrameDataset, FrameSampler, loader
from .metrics import Evaluator, log_row, predict_frames


class Trainer:
    """Torchvision detector training: sgd or adamw, linear warmup then cosine, amp, held-out map every epoch."""

    def __init__(self, det, pool, train_images, val_images, out, cfg):
        self.det, self.pool, self.cfg, self.out = det, pool, cfg, Path(out)
        c = cfg
        self.ds = FrameDataset(pool, train_images, size=c["size"], crop=c.get("crop"), seed=c.get("seed", 0),
                               **c.get("augment", {}))
        self.sampler = FrameSampler(train_images, step=c["step"], seed=c.get("seed", 0))
        if c.get("repeat", 0):
            self.sampler.set_repeat(pool, c["repeat"])
        self.val = val_images
        self.evaluator = Evaluator(pool, val_images) if val_images is not None else None
        model = det.model
        params = [p for p in model.parameters() if p.requires_grad]
        if c.get("optimizer", "sgd") == "sgd":
            self.opt = torch.optim.SGD(params, lr=c["lr"], momentum=0.9,
                                       weight_decay=c.get("weight_decay", 1e-4))
        else:
            self.opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=c.get("weight_decay", 0.05))
        self.iters = c["epochs"] * len(self.sampler) // c["batch"]
        warm = c.get("warmup", 500)
        lr_min = c.get("lr_min", 0.05)
        sched = lambda i: (i + 1) / warm if i < warm else lr_min + (1 - lr_min) * 0.5 * (
            1 + math.cos(math.pi * (i - warm) / max(1, self.iters - warm)))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, sched)
        self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
        self.epoch, self.best = 0, -1.0

    def state(self):
        return {"model": self.det.model.state_dict(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(), "epoch": self.epoch,
                "best": self.best}

    def resume(self):
        last = self.out / "last.pt"
        if last.exists():
            s = torch.load(last, map_location="cpu", weights_only=False)
            self.det.model.load_state_dict(s["model"])
            self.opt.load_state_dict(s["opt"])
            self.sched.load_state_dict(s["sched"])
            self.scaler.load_state_dict(s["scaler"])
            self.epoch, self.best = s["epoch"], s["best"]
            print(f"resumed {last} at epoch {self.epoch}")

    def train_mode(self):
        m = self.det.model
        m.train()
        if self.cfg.get("freeze_bn", True):
            for mod in m.backbone.body.modules():
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.eval()

    def run(self):
        c, model = self.cfg, self.det.model
        dev = next(model.parameters()).device
        self.resume()
        while self.epoch < c["epochs"]:
            self.ds.epoch = self.epoch
            self.sampler.set_epoch(self.epoch)
            self.train_mode()
            t, sums, n = time.time(), {}, 0
            for it, (imgs, tgts) in enumerate(loader(self.ds, c["batch"], c.get("workers", 4), self.sampler)):
                imgs = [x.to(dev, non_blocking=True) for x in imgs]
                tgts = [{k: v.to(dev) if torch.is_tensor(v) else v for k, v in d.items()} for d in tgts]
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    losses = model(imgs, tgts)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"loss is {loss.item()} at epoch {self.epoch} iter {it}")
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), c.get("clip", 10.0))
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()
                for k, v in losses.items():
                    sums[k] = sums.get(k, 0.0) + v.item()
                n += 1
                if it % c.get("print_every", 50) == 0:
                    print(f"epoch {self.epoch} iter {it}/{len(self.sampler) // c['batch']} "
                          f"loss {loss.item():.3f} "
                          f"lr {self.sched.get_last_lr()[0]:.5f} {time.time() - t:.0f}s", flush=True)
                if c.get("max_iters") and it + 1 >= c["max_iters"]:
                    break
            row = {"epoch": self.epoch, "lr": self.sched.get_last_lr()[0], "seconds": round(time.time() - t, 1),
                   **{k: round(v / max(n, 1), 4) for k, v in sums.items()}}
            if self.evaluator is not None:
                preds, fps = predict_frames(self.det, self.val, self.pool.root, c.get("eval_batch", 4),
                                            c.get("workers", 4))
                row.update({k: v for k, v in self.evaluator.score(preds).items() if not k.startswith("AP_")
                            or k == "AP_small"}, fps=round(fps, 2))
                if row["mAP"] > self.best:
                    self.best = row["mAP"]
                    self.det.save(self.out)
            else:
                self.det.save(self.out)
            print(" ".join(f"{k} {v}" for k, v in row.items()), flush=True)
            log_row(self.out / "log.csv", row)
            self.epoch += 1
            torch.save(self.state(), self.out / "last.pt")
        return self.best
