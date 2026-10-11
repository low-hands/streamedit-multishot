"""Mask propagation check: SAM3 only on keyframes, cheap propagation in between.

Question: the per-frame SAM3 masks are accurate but SAM3 on every frame is too slow. How close do we get
if SAM3 runs only on keyframes and the mask is carried to the other frames by an off-the-shelf cheap method?
The per-frame SAM3 masks (sam3_masks_v2) are the reference; a keyframe takes the reference mask as is
(stands in for "run SAM3 on this frame").

Methods (all causal: frame t only uses frames <= t)
  hold   copy the last mask (no motion at all; the floor)
  mv     warp with codec motion vectors. The source is re-encoded P-frames only, 1 reference frame
         (x264 zerolatency, like a live stream), so every vector points to the previous frame
  dis    warp with OpenCV DIS optical flow (CPU, computed at --flow_scale resolution)
Schedules (keyframes are always placed at shot starts; cuts come from the manifest here)
  N      a keyframe every N frames (e.g. 12, 24, 48)
  shot   only at shot starts
  aN     adaptive: a keyframe when the propagated area leaves [1/r, r] x the keyframe area, at most N apart
  In every schedule an empty mask (object not found / left) is re-checked every --search frames.

Comparison is done after the exact v1 conversion used by inference_edit_streamedit.py (causal glitch
filter -> majority over the 4 frames of a latent -> 8 px latent cell >= 50% -> 16 px token >= 50%),
so the numbers are what the editor would actually see:
  tok_iou      token masks (attention side, before dilation)    tok_iou_d1   both dilated by 1 token
  lat_iou      latent masks (SOG side, before dilation)         px_iou       frame masks, before conversion
  IoU is averaged over latents where either mask is non-empty; drift = tok_iou vs frames since keyframe.
  sam_per_100f = keyframes per 100 frames, prop_ms = propagation compute per frame (mv: plus decoding with
  vector export and building the maps; plain decoding excluded). A latent where the object appears while the
  mask is still empty (waiting for the next --search keyframe) counts as IoU 0.

Usage (server):  python mask_prop_eval.py --out /root/autodl-tmp/e0_runs/maskprop
Single input:    python mask_prop_eval.py --video v.mp4 --masks m.npz --shot_frames 117,45 --out dir
Writes <out>/<name>.json per mask, <out>/summary.json, and <out>/summary.png when matplotlib is present.
Needs only numpy, opencv, av (PyAV) and the ffmpeg binary from imageio-ffmpeg; no GPU.
"""
import argparse
import glob
import json
import os
import subprocess
import tempfile
import time

import av
import cv2
import numpy as np

H, W = 480, 832  # inference resolution; masks and frames are brought to it
DRIFT_BINS = [(0, 3), (4, 11), (12, 23), (24, 47), (48, 10 ** 9)]


# ---------------------------------------------------------------- v1 mask conversion (inference_edit_streamedit.py)
def glitch_filter(mb, shot_frames):
    """Causal glitch filter of --oracle_version v1, restarted at every cut. mb: bool [T, H, W], modified copy."""
    mb = mb.copy()
    s = 0
    for n in shot_frames:
        prev, raw_prev, rej_prev = None, None, False
        for t in range(s, s + n):
            raw = mb[t].copy()
            cur, rej = raw, False
            if prev is not None and prev.sum() > 0:
                iou = (raw & prev).sum() / max((raw | prev).sum(), 1)
                if raw.sum() > 2 * prev.sum() and iou < 0.5:
                    iou_raw = ((raw & raw_prev).sum() / max((raw | raw_prev).sum(), 1)
                               if raw_prev is not None else 0.0)
                    if not (rej_prev and iou_raw >= 0.5):  # unconfirmed jump -> glitch
                        cur, rej = prev, True
                        mb[t] = prev
            prev, raw_prev, rej_prev = cur, raw, rej
        s += n
    return mb


def pool_ge(x, k):
    """avg_pool2d(x, k) >= 0.5 over the last two dims."""
    L, h, w = x.shape
    return x.reshape(L, h // k, k, w // k, k).mean((2, 4)) >= 0.5


def to_latent_token(mb, shot_frames):
    """bool [T, H, W] -> (latent [L, 60, 104], token [L, 30, 52], first frame index of every latent)."""
    m = glitch_filter(mb, shot_frames).astype(np.float32)
    lats, first, s = [], [], 0
    for n in shot_frames:
        groups = [[0]] + [list(range(4 * k - 3, 4 * k + 1)) for k in range(1, (n - 1) // 4 + 1)]
        px = np.stack([(m[s:s + n][g].mean(0) >= 0.5) for g in groups]).astype(np.float32)
        lats.append(pool_ge(px, 8))
        first += [s + g[0] for g in groups]
        s += n
    lat = np.concatenate(lats)
    return lat, pool_ge(lat.astype(np.float32), 2), np.array(first)


def dilate(x, r=1):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return np.stack([cv2.dilate(f.astype(np.uint8), k) > 0 for f in x])


def iou_per(a, b):
    """per-item IoU over the last two dims; nan where both are empty."""
    inter = (a & b).sum((-1, -2))
    union = (a | b).sum((-1, -2))
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


# ---------------------------------------------------------------- inputs
def load_masks(path, T):
    m = np.load(path)["masks"][:T].astype(bool)
    if m.shape[1:] != (H, W):
        m = np.stack([cv2.resize(f.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for f in m])
    return m


def load_gray(video, T):
    out = []
    with av.open(video) as c:
        for fr in c.decode(video=0):
            out.append(cv2.resize(fr.to_ndarray(format="gray"), (W, H), interpolation=cv2.INTER_AREA))
            if len(out) == T:
                break
    return np.stack(out)


def load_mv_maps(video, T, ffmpeg):
    """Backward maps from codec motion vectors: frame t pixel p comes from frame t-1 at p + d_t(p).
    Returns float32 [T, H/4, W/4, 2] (dx, dy) on the 4 px cell grid, zero where no vector (I-frames, intra blocks), and the ms/frame of
    decoding with vector export + building the maps (the re-encode is excluded: a live stream arrives encoded)."""
    with tempfile.TemporaryDirectory() as d:
        enc = os.path.join(d, "p.mp4")
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-an", "-frames:v", str(T),
                        "-vf", f"scale={W}:{H}", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                        "-bf", "0", "-refs", "1", "-g", "100000", "-crf", "18", enc], check=True)
        maps = np.zeros((T, H // 4, W // 4, 2), np.float32)
        t0 = time.perf_counter()
        with av.open(enc) as c:
            st = c.streams.video[0]
            st.codec_context.options = {"flags2": "+export_mvs"}
            for t, fr in enumerate(c.decode(st)):
                if t == T:
                    break
                sd = fr.side_data.get("MOTION_VECTORS")
                if sd is None:
                    continue
                a = sd.to_ndarray()
                a = a[a["source"] < 0]  # past reference = previous frame (P-only, refs=1)
                sc = a["motion_scale"].astype(np.float32)
                d = np.stack([a["motion_x"] / sc, a["motion_y"] / sc], -1)  # src = dst + motion / scale
                # paint every block on a 4 px cell grid (H.264 partitions are multiples of 4), one size at a time
                bw, bh = a["w"].astype(int), a["h"].astype(int)
                cx, cy = (a["dst_x"].astype(int) - bw // 2) // 4, (a["dst_y"].astype(int) - bh // 2) // 4
                for w, h in set(zip(bw.tolist(), bh.tolist())):
                    i = np.nonzero((bw == w) & (bh == h))[0]
                    oy, ox = np.meshgrid(np.arange(h // 4), np.arange(w // 4), indexing="ij")
                    yy, xx = cy[i, None, None] + oy, cx[i, None, None] + ox
                    ok = (yy >= 0) & (yy < H // 4) & (xx >= 0) & (xx < W // 4)
                    maps[t, yy[ok], xx[ok]] = np.broadcast_to(d[i, None, None], yy.shape + (2,))[ok]
        assert t >= T - 1, f"re-encode lost frames: {t + 1} < {T}"
    return maps, (time.perf_counter() - t0) * 1e3 / T


# ---------------------------------------------------------------- propagation
GRID = np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)), -1)


def warp(soft, d):
    """soft mask of frame t-1 -> frame t, with backward displacement d [H, W, 2]."""
    m = GRID + d
    return cv2.remap(soft, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class DIS:
    def __init__(self, scale, preset):
        self.s = scale
        self.dis = cv2.DISOpticalFlow_create(preset)

    def __call__(self, prev, cur):
        """backward flow: cur(x) ~ prev(x + f(x)) at full resolution."""
        h, w = int(H * self.s), int(W * self.s)
        a, b = cv2.resize(cur, (w, h), interpolation=cv2.INTER_AREA), cv2.resize(prev, (w, h), interpolation=cv2.INTER_AREA)
        f = self.dis.calc(a, b, None)
        return cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) / self.s


def parse_schedule(s):
    if s == "shot":
        return 10 ** 9, None
    if s.startswith("a"):
        return int(s[1:]), True
    return int(s), None


def run(method, schedule, ref, shot_frames, gray, mv_maps, flow, args):
    """Propagate; returns (pred bool [T, H, W], keyframe flags [T], ms/frame of propagation)."""
    max_gap, adaptive = parse_schedule(schedule)
    T = len(ref)
    pred = np.zeros_like(ref)
    key = np.zeros(T, bool)
    starts = set(np.concatenate([[0], np.cumsum(shot_frames)[:-1]]).astype(int).tolist())
    soft, last_key, key_area, ms = None, 0, 0, 0.0
    for t in range(T):
        gap = t - last_key
        empty = soft is None or (soft > 0.5).sum() == 0
        area = 0 if soft is None else (soft > 0.5).sum()
        if t in starts:
            do_key = True
        elif empty:
            do_key = gap >= args.search
        else:
            do_key = gap >= max_gap or (adaptive and key_area > 0 and
                                        not (1 / args.area_ratio <= area / key_area <= args.area_ratio))
        if do_key:
            soft, last_key, key[t] = ref[t].astype(np.float32), t, True
            key_area = ref[t].sum()
        elif t > 0:  # propagate frame t-1 -> t (shot starts are always keyframes, so never across a cut)
            t0 = time.perf_counter()
            if method == "mv":
                soft = warp(soft, cv2.resize(mv_maps[t], (W, H), interpolation=cv2.INTER_NEAREST))
            elif method == "dis":
                soft = warp(soft, flow(gray[t - 1], gray[t]))
            ms += (time.perf_counter() - t0) * 1e3
        pred[t] = soft > 0.5
    n_prop = max(T - key.sum(), 1)
    return pred, key, ms / n_prop


def evaluate(name, video, masks, shot_frames, args, ffmpeg):
    T = sum(shot_frames)
    ref = load_masks(masks, T)
    assert len(ref) == T, (len(ref), T)
    gray = load_gray(video, T)
    assert len(gray) == T, (len(gray), T)
    methods = args.methods.split(",")
    mv_maps, mv_ms = load_mv_maps(video, T, ffmpeg) if "mv" in methods else (None, 0.0)
    flow = DIS(args.flow_scale, {"ultrafast": 0, "fast": 1, "medium": 2}[args.dis_preset])
    lat_ref, tok_ref, first = to_latent_token(ref, shot_frames)
    tok_ref_d = dilate(tok_ref)
    res = {"name": name, "video": video, "masks": masks, "shot_frames": shot_frames, "frames": T,
           "ref_area": float(ref.mean()), "ref_empty_frames": int((ref.sum((1, 2)) == 0).sum()), "runs": []}
    for method in methods:
        for schedule in args.schedules.split(","):
            pred, key, ms = run(method, schedule, ref, shot_frames, gray, mv_maps, flow, args)
            if method == "mv":
                ms += mv_ms
            lat, tok, _ = to_latent_token(pred, shot_frames)
            ti = iou_per(tok, tok_ref)
            # frames between the latent's first frame and the keyframe it was carried from
            kidx = np.maximum.accumulate(np.where(key, np.arange(T), 0))
            since = first - kidx[first]
            drift = {}
            for lo, hi in DRIFT_BINS:
                sel = (since >= lo) & (since <= hi)
                drift[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = float(np.nanmean(ti[sel])) if np.isfinite(ti[sel]).any() else None
            r = {"method": method, "schedule": schedule,
                 "tok_iou": float(np.nanmean(ti)),
                 "tok_iou_d1": float(np.nanmean(iou_per(dilate(tok), tok_ref_d))),
                 "lat_iou": float(np.nanmean(iou_per(lat, lat_ref))),
                 "px_iou": float(np.nanmean(iou_per(pred, ref))),
                 "tok_false_empty": int(((tok.sum((1, 2)) == 0) & (tok_ref.sum((1, 2)) > 0)).sum()),
                 "tok_false_object": int(((tok.sum((1, 2)) > 0) & (tok_ref.sum((1, 2)) == 0)).sum()),
                 "latents": int(len(tok)),
                 "sam_calls": int(key.sum()), "sam_per_100f": float(key.sum() / T * 100),
                 "prop_ms": ms, "drift_tok_iou": drift}
            res["runs"].append(r)
            print(f"  {name[:40]:40s} {method:5s} {schedule:5s} tok {r['tok_iou']:.3f} tok_d1 {r['tok_iou_d1']:.3f} "
                  f"lat {r['lat_iou']:.3f} px {r['px_iou']:.3f}  sam/100f {r['sam_per_100f']:5.1f}  "
                  f"{ms:6.2f} ms/f", flush=True)
    return res


def summarize(results, out):
    rows = {}
    for res in results:
        for r in res["runs"]:
            rows.setdefault((r["method"], r["schedule"]), []).append(r)
    summ = []
    print(f"\n{'method':6s} {'sched':6s} {'n':>3s} {'tok_iou':>8s} {'worst':>6s} {'tok_d1':>7s} {'lat_iou':>8s} "
          f"{'px_iou':>7s} {'sam/100f':>9s} {'ms/f':>6s}   drift tok_iou by frames since keyframe")
    for (m, s), rs in rows.items():
        g = lambda k: float(np.mean([r[k] for r in rs]))
        drift = {b: (float(np.mean(v)) if (v := [r["drift_tok_iou"][b] for r in rs if r["drift_tok_iou"][b] is not None]) else None)
                 for b in rs[0]["drift_tok_iou"]}
        row = {"method": m, "schedule": s, "n": len(rs), "tok_iou": g("tok_iou"),
               "tok_iou_worst": float(min(r["tok_iou"] for r in rs)), "tok_iou_d1": g("tok_iou_d1"),
               "lat_iou": g("lat_iou"), "px_iou": g("px_iou"), "sam_per_100f": g("sam_per_100f"),
               "prop_ms": g("prop_ms"), "drift_tok_iou": drift}
        summ.append(row)
        ds = " ".join(f"{b}:{'  -  ' if v is None else f'{v:.3f}'}" for b, v in drift.items())
        print(f"{m:6s} {s:6s} {len(rs):3d} {row['tok_iou']:8.3f} {row['tok_iou_worst']:6.3f} {row['tok_iou_d1']:7.3f} "
              f"{row['lat_iou']:8.3f} {row['px_iou']:7.3f} {row['sam_per_100f']:9.1f} {row['prop_ms']:6.2f}   {ds}")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in dict.fromkeys(r["method"] for r in summ):
        rs = sorted([r for r in summ if r["method"] == m], key=lambda r: r["sam_per_100f"])
        ax[0].plot([r["sam_per_100f"] for r in rs], [r["tok_iou"] for r in rs], "o-", label=m)
        for r in rs:
            ax[0].annotate(r["schedule"], (r["sam_per_100f"], r["tok_iou"]), fontsize=7)
        r = next((r for r in rs if r["schedule"] == "shot"), rs[0])
        bins = list(r["drift_tok_iou"])
        ax[1].plot(range(len(bins)), [r["drift_tok_iou"][b] if r["drift_tok_iou"][b] is not None else np.nan for b in bins],
                   "o-", label=f"{m} ({r['schedule']})")
        ax[1].set_xticks(range(len(bins)), bins)
    ax[0].set(xlabel="SAM3 calls per 100 frames", ylabel="token IoU vs per-frame SAM3", title="accuracy vs SAM calls")
    ax[1].set(xlabel="frames since keyframe", ylabel="token IoU", title="drift")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--pool", default="/root/autodl-tmp/pool_v3")
    p.add_argument("--clips", default="", help="comma-separated clip names (default: every clip in edits.json)")
    p.add_argument("--video", default=None, help="single-input mode: video path")
    p.add_argument("--masks", default=None, help="single-input mode: npz with `masks` bool [T, H, W]")
    p.add_argument("--shot_frames", default=None, help="single-input mode: frames per shot, e.g. 117,45")
    p.add_argument("--methods", default="hold,mv,dis")
    p.add_argument("--schedules", default="12,24,48,shot,a48")
    p.add_argument("--search", type=int, default=12, help="re-check interval while the mask is empty")
    p.add_argument("--area_ratio", type=float, default=1.5, help="adaptive schedule: re-key outside [1/r, r]")
    p.add_argument("--flow_scale", type=float, default=0.5)
    p.add_argument("--dis_preset", choices=["ultrafast", "fast", "medium"], default="fast")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    if args.video:
        jobs = [(os.path.splitext(os.path.basename(args.masks))[0], args.video, args.masks,
                 [int(x) for x in args.shot_frames.split(",")])]
    else:
        meta = {m["name"]: m for m in json.load(open(f"{args.pool}/manifest_aligned_v2.json"))}
        edits = json.load(open(f"{args.pool}/edits.json"))
        keep = set(args.clips.split(",")) if args.clips else None
        jobs, seen = [], set()
        for c in edits:
            if keep and c["clip"] not in keep:
                continue
            for e in c["edits"].values():  # one job per (clip, SAM3 prompt): several edits can share a mask
                npz = f"{args.pool}/sam3_masks_v2/{c['clip']}__{e['sam3_prompt'].replace(' ', '_')}.npz"
                if npz in seen or not os.path.exists(npz):
                    continue
                seen.add(npz)
                m = meta[c["clip"]]
                jobs.append((os.path.basename(npz)[:-4], m["video"], npz, m["shot_frames"]))
    print(f"{len(jobs)} masks", flush=True)
    results = []
    for name, video, masks, sf in jobs:
        js = os.path.join(args.out, name + ".json")
        if os.path.exists(js):  # resumable
            results.append(json.load(open(js)))
            continue
        res = evaluate(name, video, masks, sf, args, ffmpeg)
        json.dump(res, open(js, "w"), indent=1)
        results.append(res)
    summarize(results, args.out)


if __name__ == "__main__":
    main()
