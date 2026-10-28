import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

from .data import CLASSES, VISDRONE

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPLIT_COLORS = {"train": COLORS[0], "val": COLORS[1], "test": COLORS[2]}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f7f6f3", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
MUTED = "#8a8980"
SIZE_BINS = [0, 8, 16, 32, 96, np.inf]
SIZE_LABELS = ["<8", "8-16", "16-32", "32-96", ">96"]

def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)

def shape_iou(wh, anchors):
    inter = np.minimum(wh[:, None, 0], anchors[None, :, 0]) * np.minimum(wh[:, None, 1], anchors[None, :, 1])
    return inter / (wh.prod(1)[:, None] + anchors.prod(1)[None, :] - inter)

class Stats:
    @staticmethod
    def overview(splits, test):
        rows = []
        for s in splits:
            im = s.images
            rows.append({"split": s.name, "clips": im["clip"].nunique(), "frames": len(im),
                         "boxes": len(s.boxes), "obj/frame mean": im["n_obj"].mean().round(1),
                         "obj/frame max": im["n_obj"].max(),
                         "frames with ignore": (im["n_ignore"] > 0).mean().round(3),
                         "resolutions": len(im.groupby(["width", "height"]))})
        rows.append({"split": "test", "clips": test["clip"].nunique(), "frames": len(test),
                     "resolutions": len(test.groupby(["width", "height"]))})
        df = pd.DataFrame(rows).set_index("split")
        return df.astype({c: "Int64" for c in ("boxes", "obj/frame max")})

    @staticmethod
    def clips(split):
        im, b = split.images, split.boxes
        g = im.groupby("clip")
        df = pd.DataFrame({
            "frames": g.size(),
            "res": g.apply(lambda d: f"{d['width'].iat[0]}x{d['height'].iat[0]}"),
            "obj/frame": g["n_obj"].mean().round(1),
            "obj max": g["n_obj"].max(),
            "ignore frames": g["n_ignore"].apply(lambda n: (n > 0).mean()).round(2),
        })
        bg = b.groupby("clip")
        df["median px"] = bg["size"].median().round(1)
        df["<32 px"] = bg["size"].apply(lambda s: (s < 32).mean()).round(2)
        top = b.groupby("clip")["cls"].value_counts(normalize=True).groupby(level=0).head(1)
        df["top class"] = pd.Series({c: f"{CLASSES[k]} {v:.2f}" for (c, k), v in top.items()})
        return df

    @staticmethod
    def resolutions(splits, test):
        parts = [s.images.drop_duplicates("clip").assign(split=s.name) for s in splits]
        parts.append(test.drop_duplicates("clip").assign(split="test"))
        df = pd.concat(parts)
        df["res"] = df["width"].astype(str) + "x" + df["height"].astype(str)
        return pd.crosstab(df["res"], df["split"])

    @staticmethod
    def clip_lengths(splits, test):
        parts = [s.images.groupby("clip").size().describe().rename(s.name) for s in splits]
        parts.append(test.groupby("clip").size().describe().rename("test"))
        return pd.concat(parts, axis=1).round(1)

    @staticmethod
    def class_counts(splits):
        df = pd.concat({s.name: s.boxes["cls"].value_counts().sort_index() for s in splits}, axis=1)
        df.index = CLASSES
        for s in splits:
            df[f"{s.name} share"] = (df[s.name] / df[s.name].sum()).round(4)
        return df

    @staticmethod
    def class_by_clip(splits):
        b = pd.concat([s.boxes.assign(clip=s.name + "/" + s.boxes["clip"]) for s in splits])
        df = pd.crosstab(b["clip"], b["cls"], normalize="index")
        return df.reindex(columns=range(len(CLASSES)), fill_value=0).set_axis(CLASSES, axis=1)

    @staticmethod
    def merges(split):
        b = pd.concat([split.boxes[["orig", "cls"]], split.ignore[["orig"]].assign(cls=-1)])
        names = b["cls"].map(lambda c: "ignore" if c < 0 else CLASSES[c])
        return pd.crosstab(b["orig"].map(lambda o: VISDRONE[o]), names)

    @staticmethod
    def size_bins(boxes, col="size"):
        bins = pd.cut(boxes[col], SIZE_BINS, labels=SIZE_LABELS, right=False)
        names = boxes["cls"].map(lambda c: CLASSES[c])
        df = pd.crosstab(names, bins, normalize="index").reindex(columns=SIZE_LABELS, fill_value=0)
        df.loc["all"] = bins.value_counts(normalize=True).reindex(SIZE_LABELS)
        df = df.round(3)
        df["count"] = names.value_counts().reindex(df.index).fillna(len(boxes)).astype(int)
        return df.reindex([c for c in CLASSES if c in df.index] + ["all"])

    @staticmethod
    def scaled(boxes, long_side):
        if long_side is None:
            return boxes["size"]
        return boxes["size"] * long_side / np.maximum(boxes["width"], boxes["height"])

    @staticmethod
    def under(boxes, sides=(640, 1024, 1280, 1536, None), px=(8, 16, 32)):
        rows = {}
        for side in sides:
            s = Stats.scaled(boxes, side)
            rows["native" if side is None else side] = {f"<{p} px": (s < p).mean() for p in px}
        return pd.DataFrame(rows).T.round(3)

    @staticmethod
    def aspect(boxes):
        b = boxes.assign(ar=boxes["w"] / boxes["h"], name=boxes["cls"].map(lambda c: CLASSES[c]))
        g = b.groupby("name")
        return pd.DataFrame({"ar p10": g["ar"].quantile(0.1), "ar median": g["ar"].median(),
                             "ar p90": g["ar"].quantile(0.9), "median w": g["w"].median(),
                             "median h": g["h"].median()}).reindex(CLASSES).round(2)

    @staticmethod
    def kmeans(wh, k, iters=200, seed=0, n=200_000):
        rng = np.random.default_rng(seed)
        wh = wh[rng.choice(len(wh), min(n, len(wh)), replace=False)]
        c = wh[rng.choice(len(wh), k, replace=False)]
        for _ in range(iters):
            a = shape_iou(wh, c).argmax(1)
            new = np.array([np.median(wh[a == j], 0) if (a == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        return c[np.argsort(c.prod(1))]

    @staticmethod
    def default_anchors(kind):
        if kind == "retinanet":
            sizes = [x * 2 ** (i / 3) for x in (32, 64, 128, 256, 512) for i in range(3)]
        else:
            sizes = [32, 64, 128, 256, 512]
        hr = np.sqrt(np.array([0.5, 1.0, 2.0]))
        return np.array([(s / r, s * r) for s in sizes for r in hr])

    @staticmethod
    def anchor_fit(wh, anchors, thr=(0.3, 0.4, 0.5)):
        best = np.concatenate([shape_iou(c, anchors).max(1) for c in np.array_split(wh, 50)])
        return {"mean best iou": best.mean().round(3), **{f">={t}": (best >= t).mean().round(3) for t in thr}}

    @staticmethod
    def crowding(split):
        xywh = split.boxes[["x", "y", "w", "h"]].to_numpy()
        out = np.zeros(len(xywh), dtype=np.float32)
        for idx in split.boxes.groupby("image_id").indices.values():
            if len(idx) < 2:
                continue
            m = iou_matrix(xywh[idx], xywh[idx])
            np.fill_diagonal(m, 0)
            out[idx] = m.max(1)
        return pd.Series(out, index=split.boxes.index)

    @staticmethod
    def ignore_cover(split, down=8):
        cover = pd.Series(0.0, index=split.images["image_id"])
        inside = np.zeros(len(split.boxes), dtype=bool)
        groups = split.boxes.groupby("image_id").indices
        cx = (split.boxes["x"] + split.boxes["w"] / 2).to_numpy()
        cy = (split.boxes["y"] + split.boxes["h"] / 2).to_numpy()
        for iid, reg in split.ignore.groupby("image_id"):
            W, H = reg["width"].iat[0], reg["height"].iat[0]
            mask = np.zeros((H // down + 1, W // down + 1), dtype=bool)
            r = reg[["x", "y", "w", "h"]].to_numpy()
            for x, y, w, h in (r / down).round().astype(int):
                mask[y:y + h + 1, x:x + w + 1] = True
            cover[iid] = mask.mean()
            idx = groups.get(iid)
            if idx is not None:
                hit = ((cx[idx, None] >= r[:, 0]) & (cx[idx, None] <= r[:, 0] + r[:, 2])
                       & (cy[idx, None] >= r[:, 1]) & (cy[idx, None] <= r[:, 1] + r[:, 3]))
                inside[idx] = hit.any(1)
        return cover, pd.Series(inside, index=split.boxes.index)

    @staticmethod
    def track_shift(boxes, steps=(1, 2, 4, 8, 16), by=None):
        b = boxes[["clip", "track_id", "frame", "x", "y", "w", "h"]]
        rows = []
        for s in steps:
            m = b.merge(b.assign(frame=b["frame"] - s), on=["clip", "track_id", "frame"], suffixes=("", "_n"))
            a, n = m[["x", "y", "w", "h"]].to_numpy(), m[["x_n", "y_n", "w_n", "h_n"]].to_numpy()
            x1, y1 = np.maximum(a[:, 0], n[:, 0]), np.maximum(a[:, 1], n[:, 1])
            x2 = np.minimum(a[:, 0] + a[:, 2], n[:, 0] + n[:, 2])
            y2 = np.minimum(a[:, 1] + a[:, 3], n[:, 1] + n[:, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            m["iou"] = inter / (a[:, 2] * a[:, 3] + n[:, 2] * n[:, 3] - inter)
            m["shift"] = np.hypot(a[:, 0] + a[:, 2] / 2 - n[:, 0] - n[:, 2] / 2,
                                  a[:, 1] + a[:, 3] / 2 - n[:, 1] - n[:, 3] / 2)
            m["rel"] = m["shift"] / np.sqrt(a[:, 2] * a[:, 3])
            m["step"] = s
            g = m.groupby(["step"] + ([by] if by else []))
            rows.append(pd.DataFrame({"pairs": g.size(), "iou median": g["iou"].median(),"iou<0.5": g["iou"].apply(lambda v: (v < 0.5).mean()), "shift px": g["shift"].median(), "shift/size": g["rel"].median()}))
        return pd.concat(rows).round(3)

    @staticmethod
    def camera_motion(split, pairs=4, step=1, seed=0, long_side=960):
        rng = np.random.default_rng(seed)
        orb = cv2.ORB_create(3000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        rows = []
        for clip, im in split.images.groupby("clip"):
            im = im.set_index("frame")
            starts = [f for f in im.index if f + step in im.index]
            for t in rng.choice(starts, min(pairs, len(starts)), replace=False):
                a, b = (cv2.imread(str(split.root / im.at[f, "path"]), cv2.IMREAD_GRAYSCALE)
                        for f in (t, t + step))
                k = long_side / max(a.shape)
                a, b = (cv2.resize(x, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) for x in (a, b))
                ka, da = orb.detectAndCompute(a, None)
                kb, db = orb.detectAndCompute(b, None)
                if da is None or db is None:
                    continue
                mt = bf.match(da, db)
                if len(mt) < 20:
                    continue
                pa = np.float32([ka[m.queryIdx].pt for m in mt])
                pb = np.float32([kb[m.trainIdx].pt for m in mt])
                H, ok = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
                if H is None:
                    continue
                h, w = a.shape
                c = np.float32([[[w / 2, h / 2]]])
                shift = np.linalg.norm(cv2.perspectiveTransform(c, H) - c) / k
                rows.append({"clip": clip, "shift px": shift, "inliers": ok.mean(),
                             "shift % diag": 100 * shift * k / np.hypot(w, h)})
        df = pd.DataFrame(rows)
        return df.groupby("clip").median().round(2)

    @staticmethod
    def occlusion(boxes):
        names = boxes["cls"].map(lambda c: CLASSES[c])
        occ = boxes["occlusion"].map({0: "none", 1: "partial", 2: "heavy"})
        occ = pd.crosstab(names, occ, normalize="index")
        trunc = pd.crosstab(names, boxes["truncation"].map({0: "none", 1: "truncated"}), normalize="index")
        return pd.concat({"occlusion": occ[["none", "partial", "heavy"]], "truncation": trunc[["truncated"]]},
                         axis=1).reindex(CLASSES).round(3)


def densest_window(boxes, w, h):
    c = np.stack([(boxes["x"] + boxes["w"] / 2).to_numpy(), (boxes["y"] + boxes["h"] / 2).to_numpy()], 1)
    d = np.abs(c[:, None] - c[None, :])
    n = ((d[..., 0] < w / 2) & (d[..., 1] < h / 2)).sum(1)
    cx, cy = c[n.argmax()]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


class Plot:
    @staticmethod
    def style():
        plt.rcParams.update({
            "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
            "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": "#e4e3dd", "grid.linewidth": 0.6,
            "axes.axisbelow": True, "xtick.color": "#55544e", "ytick.color": "#55544e", "font.size": 9,
            "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
        })

    @staticmethod
    def class_bars(counts, splits, ax):
        """Grouped bars of class share per split."""
        x = np.arange(len(CLASSES))
        w = 0.8 / len(splits)
        for i, s in enumerate(splits):
            ax.bar(x + (i - (len(splits) - 1) / 2) * w, counts[f"{s} share"], w, color=SPLIT_COLORS[s],
                   edgecolor="white", linewidth=1, label=s)
        ax.set_xticks(x, CLASSES)
        ax.set_ylabel("share of boxes")
        ax.set_yscale("log")
        ax.legend()

    @staticmethod
    def size_ecdf(boxes, ax, col="size", title=None):
        for c, name in enumerate(CLASSES):
            s = np.sort(boxes.loc[boxes["cls"] == c, col].to_numpy())
            if len(s):
                ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=COLORS[c], label=name)
        for t in (8, 16, 32, 96):
            ax.axvline(t, color=MUTED, linewidth=0.8, linestyle="--")
            ax.text(t, 1.01, f"{t}", color=MUTED, ha="center", va="bottom", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("sqrt(box area), px")
        ax.set_ylabel("share of boxes below")
        ax.set_title(title or "", pad=14)
        ax.legend(ncol=2, fontsize=8, loc="lower right")

    @staticmethod
    def heatmap(df, ax, annotate=None, fmt="{:.2f}", vmax=None):
        im = ax.imshow(df.to_numpy(dtype=float), cmap=BLUES, aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha="right")
        ax.set_yticks(range(df.shape[0]), df.index, fontsize=7 if df.shape[0] > 30 else 9)
        ax.grid(False)
        if annotate if annotate is not None else df.size <= 120:
            v = df.to_numpy(dtype=float)
            hi = np.nanmax(v) if vmax is None else vmax
            for i, j in np.ndindex(v.shape):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if v[i, j] > 0.6 * hi else "#2b2b28")
        plt.colorbar(im, ax=ax, fraction=0.03)

    @staticmethod
    def per_clip(images, col, ax, ylabel):
        order = images.groupby("clip")[col].median().sort_values().index
        data = [images.loc[images["clip"] == c, col] for c in order]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        split = images.drop_duplicates("clip").set_index("clip")["split"]
        for patch, c in zip(bp["boxes"], order):
            patch.set(facecolor=SPLIT_COLORS[split[c]], edgecolor="white", alpha=0.9)
        for k in ("medians", "whiskers", "caps"):
            plt.setp(bp[k], color="#55544e", linewidth=1)
        ax.set_xticks([])
        ax.set_xlabel("clips, sorted by median")
        ax.set_ylabel(ylabel)
        ax.legend(handles=[Patch(color=SPLIT_COLORS[s], label=s) for s in split.unique()])

    @staticmethod
    def hist(values, ax, bins=50, color=COLORS[0], label=None, log=False, density=False):
        ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.5, label=label, alpha=0.75,
                density=density)
        if log:
            ax.set_yscale("log")

    @staticmethod
    def centers(boxes, ax, bins=60, title=None):
        """2d histogram of box centres in normalised image coordinates."""
        h, _, _ = np.histogram2d(boxes["cy"], boxes["cx"], bins=bins, range=[[0, 1], [0, 1]])
        h = h / h.sum()
        im = ax.imshow(h, cmap=BLUES, extent=[0, 1, 1, 0], vmax=np.quantile(h[h > 0], 0.99))
        ax.set_title(title or "")
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.046)

    @staticmethod
    def frame(split, image_id, ax, crop=None, title=None):
        img = split.frame(image_id)
        b, ign = split.frame_boxes(image_id)
        ax.imshow(img)
        for r in ign.itertuples():
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, facecolor=(0.5, 0.5, 0.5, 0.35), edgecolor="white",
                                   hatch="///", linewidth=0.8))
        for r in b.itertuples():
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False, edgecolor=COLORS[r.cls], linewidth=1))
        if crop is not None:
            x0, y0, x1, y1 = crop
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
        ax.set_title(title or f"{b['clip'].iat[0] if len(b) else ''} #{image_id}  {len(b)} boxes", fontsize=9)
        ax.axis("off")

    @staticmethod
    def class_legend(fig, ignore=True):
        handles = [Patch(edgecolor=COLORS[c], facecolor="none", linewidth=2, label=n)
                   for c, n in enumerate(CLASSES)]
        if ignore:
            handles.append(Patch(facecolor=(0.5, 0.5, 0.5, 0.35), hatch="///", edgecolor="white",
                                 label="ignored"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
