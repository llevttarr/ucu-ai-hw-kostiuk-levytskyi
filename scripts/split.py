import argparse
import json
from pathlib import Path

import pandas as pd

from hw2 import CLASSES, Pool
from hw2.data import HOLDOUT, make_holdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-val", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=HOLDOUT)
    args = p.parse_args()

    pool = Pool()
    d = make_holdout(pool, args.n_val, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(d, indent=1))

    part = pool.boxes["clip"].isin(d["val"]).map({True: "val", False: "train"})
    share = pd.crosstab(pool.boxes["cls"].map(lambda c: CLASSES[c]), part, normalize="columns").reindex(CLASSES)
    frames = pool.images["clip"].isin(d["val"]).mean()
    print(share.round(3))
    print(f"val: {len(d['val'])} clips, {frames:.1%} of frames, score {d['score']}, wrote {args.out}")


if __name__ == "__main__":
    main()
