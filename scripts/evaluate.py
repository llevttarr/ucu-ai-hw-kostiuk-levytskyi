import argparse
import json
import pickle
from pathlib import Path

from hw2 import Pool, load_holdout
from hw2.metrics import evaluate, log_row
from hw2.models import detector, parse_set
from hw2.post import Temporal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run folder or ensemble config")
    p.add_argument("--name")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides, e.g. size=1280")
    p.add_argument("--every", type=int, default=4, help="keep every n-th 80-frame window of the val clips")
    p.add_argument("--temporal", nargs="*", metavar="KEY=VALUE", help="track rescoring, with optional params")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--save", action="store_true", help="pickle the predictions next to the run")
    p.add_argument("--note", default="")
    args = p.parse_args()
    det = detector(args.run, args.set)
    name = args.name or Path(args.run).stem + ("_temporal" if args.temporal is not None else "")

    pool = Pool()
    _, va = load_holdout()
    images = pool.windows(va, every=args.every)
    temporal = Temporal(**parse_set(args.temporal)) if args.temporal is not None else None
    scores, fps, preds = evaluate(det, pool, images, temporal, det.cfg.get("eval_batch", 4),
                                  det.cfg.get("workers", 4))
    log_row(args.out / "experiments.csv", {"name": name, "model": det.cfg["model"], **scores,
                                           "fps": round(fps, 2), "frames": len(images), "note": args.note})
    print(json.dumps({**scores, "fps": round(fps, 2)}, indent=1))
    if args.save:
        dst = (Path(args.run) if Path(args.run).is_dir() else args.out / name) / f"preds_{name}.pkl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            pickle.dump({"images": images, "preds": preds}, f)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
