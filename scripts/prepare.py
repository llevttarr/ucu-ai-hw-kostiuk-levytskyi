import argparse
from pathlib import Path

from hw2 import CLASSES, Pool
from hw2.data import run_images
from hw2.models import YOLO_DATA, build, load_config, yolo_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="*", type=Path, help="configs whose yolo export and weights to prepare")
    p.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()

    pool = Pool()
    print(f"pool: {len(pool.images)} frames, {len(pool.boxes)} boxes, yolo data in {YOLO_DATA}")
    for path in args.configs:
        cfg = {**load_config(path, args.set), "classes": CLASSES}
        if cfg["model"] in ("yolo", "rtdetr"):
            print(path, yolo_dataset(cfg, pool, *run_images(pool, cfg, args.full)))
            build(cfg)
        else:
            build({**cfg, "anchors": "default"})
        print(path, "ready")


if __name__ == "__main__":
    main()
