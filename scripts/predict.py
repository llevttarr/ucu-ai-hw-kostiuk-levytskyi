import argparse
from pathlib import Path

from hw2 import DATA, test_images
from hw2.metrics import by_window, predict_frames, write_submission
from hw2.models import detector, parse_set
from hw2.post import Temporal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run folder or ensemble config")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--temporal", nargs="*", metavar="KEY=VALUE")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    det = detector(args.run, args.set)

    test = test_images()
    preds, fps = predict_frames(det, test, DATA, det.cfg.get("eval_batch", 4), det.cfg.get("workers", 4))
    if args.temporal is not None:
        preds = by_window(test, preds, Temporal(**parse_set(args.temporal)), col="clip")
    name = Path(args.run).stem + ("_temporal" if args.temporal is not None else "")
    out = args.out or Path("runs/submissions") / f"{name}.csv"
    write_submission(out, test, preds)
    print(f"{fps:.1f} frames/s")


if __name__ == "__main__":
    main()
