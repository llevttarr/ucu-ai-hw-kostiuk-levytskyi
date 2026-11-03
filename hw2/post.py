import numpy as np


def iou_xywh(a, b):
    """(N, M) iou between (N, 4) and (M, 4) xywh boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum(a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter, 1e-9)


def wbf(preds, weights=None, iou=0.6, skip=0.001):
    """Weighted box fusion of several (N, 6) predictions of one frame into one (K, 6)."""
    weights = weights or [1.0] * len(preds)
    allp = [np.column_stack([p, np.full(len(p), w)]) for p, w in zip(preds, weights) if len(p)]
    if not allp:
        return np.zeros((0, 6))
    allp = np.concatenate(allp)
    allp = allp[allp[:, 4] >= skip]
    out = []
    for c in np.unique(allp[:, 5]):
        p = allp[allp[:, 5] == c]
        p = p[np.argsort(-p[:, 4] * p[:, 6])]
        box, wsum, top = np.zeros((len(p), 4)), np.zeros(len(p)), np.zeros(len(p))
        acc, n = np.zeros((len(p), 4)), 0
        for row in p:
            sw = row[4] * row[6]
            o = iou_xywh(row[None, :4], box[:n])[0] if n else np.zeros(1)
            j = int(o.argmax())
            if not n or o[j] <= iou:
                j, n = n, n + 1
                top[j] = row[4]
            acc[j] += sw * row[:4]
            wsum[j] += sw
            box[j] = acc[j] / wsum[j]
        score = np.minimum(wsum[:n] / sum(weights), top[:n])
        out.append(np.column_stack([box[:n], score, np.full(n, c)]))
    return np.concatenate(out)


class Temporal:
    """Rescoring of one clip's (N, 6) frames along iou tracks: long ones boosted, short damped, gaps filled."""

    def __init__(self, iou=0.5, link=0.05, alpha=0.6, min_len=3, short=0.8, gap=2, fill=0.7):
        self.iou, self.link, self.alpha, self.min_len = iou, link, alpha, min_len
        self.short, self.gap, self.fill = short, gap, fill

    def tracks(self, preds):
        """Lists of (frame index, row) per track, only rows with score >= link take part."""
        tracks, active = [], []
        for t, p in enumerate(preds):
            cand = np.flatnonzero(p[:, 4] >= self.link)
            active = [k for k in active if t - tracks[k][-1][0] <= self.gap + 1]
            took, used = set(), set()
            if active and len(cand):
                last = np.array([preds[tracks[k][-1][0]][tracks[k][-1][1]] for k in active])
                o = iou_xywh(last[:, :4], p[cand, :4])
                o[last[:, 5][:, None] != p[cand, 5][None]] = 0
                for a, b in zip(*np.unravel_index(np.argsort(-o, axis=None), o.shape)):
                    if o[a, b] < self.iou:
                        break
                    if active[a] in took or cand[b] in used:
                        continue
                    tracks[active[a]].append((t, cand[b]))
                    took.add(active[a])
                    used.add(cand[b])
            for b in cand:
                if b not in used:
                    tracks.append([(t, b)])
                    active.append(len(tracks) - 1)
        return tracks

    def __call__(self, preds):
        preds = [p.copy() for p in preds]
        tr = self.tracks(preds)
        new = [p[:, 4].copy() for p in preds]
        extra = [[] for _ in preds]
        for track in tr:
            s = np.array([preds[t][i, 4] for t, i in track])
            if len(track) < self.min_len:
                for t, i in track:
                    new[t][i] = preds[t][i, 4] * self.short
                continue
            m = s.mean()
            for t, i in track:
                new[t][i] = max(preds[t][i, 4], self.alpha * preds[t][i, 4] + (1 - self.alpha) * m)
            for (t0, i0), (t1, i1) in zip(track, track[1:]):
                a, b = preds[t0][i0], preds[t1][i1]
                for t in range(t0 + 1, t1):
                    k = (t - t0) / (t1 - t0)
                    row = a.copy()
                    row[:4] = (1 - k) * a[:4] + k * b[:4]
                    row[4] = self.fill * min(a[4], b[4])
                    extra[t].append(row)
        out = []
        for p, s, e in zip(preds, new, extra):
            p[:, 4] = s
            out.append(np.concatenate([p, np.array(e).reshape(-1, 6)]) if e else p)
        return out
