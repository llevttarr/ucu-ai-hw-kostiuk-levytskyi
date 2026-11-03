import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from hw2 import CLASSES, Pool, load_holdout
from hw2.metrics import evaluate, log_row
from hw2.models import build, load_config, load_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--name")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. lr=0.01")
    p.add_argument("--full", action="store_true", help="train on all clips, no held-out evaluation")
    p.add_argument("--note", default="")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    cfg["classes"] = CLASSES
    name = args.name or args.config.stem + ("_full" if args.full else "")
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)

    pool = Pool()
    tr, va = load_holdout()
    if args.full:
        train_images, val_images = pool.frames(tr + va), None
    else:
        train_images, val_images = pool.frames(tr), pool.windows(va, every=cfg.get("train_eval_every", 8))
        if cfg.get("eval_frames"):
            val_images = val_images.groupby("window").head(cfg["eval_frames"]).reset_index(drop=True)
    det = build(cfg, pool, train_images)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    t = time.time()
    det.fit(pool, train_images, val_images, out)
    seconds = time.time() - t
    if args.full:
        return
    del det
    gc.collect()
    torch.cuda.empty_cache()

    images = pool.windows(va, every=cfg.get("eval_every", 4))
    if cfg.get("eval_frames"):
        images = images.groupby("window").head(cfg["eval_frames"]).reset_index(drop=True)
    scores, fps, _ = evaluate(load_run(out), pool, images, batch=cfg.get("eval_batch", 4),
                              workers=cfg.get("workers", 4))
    (out / "eval.json").write_text(json.dumps({**scores, "fps": fps}, indent=1))
    log_row(args.out / "experiments.csv", {"name": name, "model": cfg["model"], **scores, "fps": round(fps, 2),
                                           "frames": len(images), "train_min": round(seconds / 60, 1),
                                           "note": args.note})
    print(name, " ".join(f"{k} {v}" for k, v in scores.items()), f"fps {fps:.1f}")


if __name__ == "__main__":
    main()
