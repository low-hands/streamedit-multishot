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
         (r = --area_ratio, or per schedule: a24r1.25)
  In every schedule an empty mask (object not found / left) is re-checked every --search frames.

Comparison is done after the exact conversion used by inference_edit_streamedit.py, so the numbers are
what the editor would actually see. --conv v1 (default): causal glitch filter -> majority over the 4 frames
of a latent -> 8 px latent cell >= 50% -> 16 px token >= 50%. --conv v0: union over the 4 frames -> 16 px
token covered > 10%; the SOG side then uses the token mask, so lat_* is the token mask at latent size.
  tok_iou      token masks (attention side, before dilation)    tok_iou_d1   both dilated by 1 token
  lat_iou      latent masks (SOG side, before dilation)         px_iou       frame masks, before conversion
  IoU is averaged over latents where either mask is non-empty; drift = tok_iou vs frames since keyframe.
  sam_per_100f = keyframes per 100 frames, prop_ms = propagation compute per frame (mv: plus decoding with
  vector export and building the maps; plain decoding excluded). A latent where the object appears while the
  mask is still empty (waiting for the next --search keyframe) counts as IoU 0.

Per mask it also prints the object size in tokens and ref_consec = token IoU of the reference between
consecutive latents of a shot (how much the object itself moves / the reference jitters at token level).

--save_pred mv:a48,mv:12 writes the propagated masks as <out>/pred_<method>_<schedule>/<name>.npz in the
sam3_masks_v2 format (`masks` bool [T, 480, 832], plus `keyframes`), usable as --oracle_mask for editing.

--viz mv:12,dis:12,hold:12 writes <out>/viz/<name>.mp4: one panel per method:schedule, the source frame
with the per-frame SAM3 mask (green outline) and the propagated mask (red fill); the header shows the
frame, the shot, KEY on SAM3 keyframes and the frame's pixel IoU. --viz_tokens draws both at 16 px token level
(what attention sees) instead of pixel level.

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


def to_latent_token_v0(mb, shot_frames):
    """--oracle_version v0: union over the frames of a latent, token = 16 px cell covered > 10%."""
    toks, first, s = [], [], 0
    for n in shot_frames:
        groups = [[0]] + [list(range(4 * k - 3, 4 * k + 1)) for k in range(1, (n - 1) // 4 + 1)]
        px = np.stack([mb[s:s + n][g].any(0) for g in groups]).astype(np.float32)
        L, h, w = px.shape
        toks.append(px.reshape(L, h // 16, 16, w // 16, 16).mean((2, 4)) > 0.1)
        first += [s + g[0] for g in groups]
        s += n
    tok = np.concatenate(toks)
    return tok.repeat(2, 1).repeat(2, 2), tok, np.array(first)


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


def parse_schedule(s, area_ratio):
    """-> (max gap, adaptive area ratio or None)"""
    if s == "shot":
        return 10 ** 9, None
    if s.startswith("a"):
        n, _, r = s[1:].partition("r")
        return int(n), float(r) if r else area_ratio
    return int(s), None


def run(method, schedule, ref, shot_frames, gray, mv_maps, flow, args):
    """Propagate; returns (pred bool [T, H, W], keyframe flags [T], ms/frame of propagation)."""
    max_gap, ratio = parse_schedule(schedule, args.area_ratio)
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
            do_key = gap >= max_gap or (ratio is not None and key_area > 0 and
                                        not (1 / ratio <= area / key_area <= ratio))
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


def ref_consec(tok, shot_frames):
    """token IoU of the reference between consecutive latents inside a shot."""
    lens = [1 + (n - 1) // 4 for n in shot_frames]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(int)
    vals = [iou_per(tok[a + 1:a + l], tok[a:a + l - 1]) for a, l in zip(starts, lens) if l > 1]
    v = np.concatenate(vals) if vals else np.array([np.nan])
    return float(np.nanmean(v)) if np.isfinite(v).any() else None


def evaluate(name, video, masks, shot_frames, args, ffmpeg):
    T = sum(shot_frames)
    ref = load_masks(masks, T)
    assert len(ref) == T, (len(ref), T)
    gray = load_gray(video, T)
    assert len(gray) == T, (len(gray), T)
    methods = args.methods.split(",")
    mv_maps, mv_ms = load_mv_maps(video, T, ffmpeg) if "mv" in methods else (None, 0.0)
    flow = DIS(args.flow_scale, {"ultrafast": 0, "fast": 1, "medium": 2}[args.dis_preset])
    convert = to_latent_token_v0 if args.conv == "v0" else to_latent_token
    lat_ref, tok_ref, first = convert(ref, shot_frames)
    tok_ref_d = dilate(tok_ref)
    res = {"name": name, "video": video, "masks": masks, "shot_frames": shot_frames, "frames": T,
           "ref_area": float(ref.mean()), "ref_empty_frames": int((ref.sum((1, 2)) == 0).sum()),
           "ref_tokens": float(np.mean([n for n in tok_ref.sum((1, 2)) if n > 0] or [0])),
           "ref_consec": ref_consec(tok_ref, shot_frames), "runs": []}
    save = {tuple(x.split(":")) for x in args.save_pred.split(",") if x}
    viz = [tuple(x.split(":")) for x in args.viz.split(",") if x]
    viz_runs = {}
    for method in methods:
        for schedule in args.schedules.split(","):
            pred, key, ms = run(method, schedule, ref, shot_frames, gray, mv_maps, flow, args)
            if method == "mv":
                ms += mv_ms
            if (method, schedule) in viz:
                viz_runs[(method, schedule)] = (pred, key)
            if (method, schedule) in save:
                d = os.path.join(args.out, f"pred_{method}_{schedule}")
                os.makedirs(d, exist_ok=True)
                np.savez_compressed(os.path.join(d, name + ".npz"), masks=pred, keyframes=np.nonzero(key)[0])
            lat, tok, _ = convert(pred, shot_frames)
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
            print(f"  {name[:40]:40s} {method:5s} {schedule:9s} tok {r['tok_iou']:.3f} tok_d1 {r['tok_iou_d1']:.3f} "
                  f"lat {r['lat_iou']:.3f} px {r['px_iou']:.3f}  sam/100f {r['sam_per_100f']:5.1f}  "
                  f"{ms:6.2f} ms/f", flush=True)
    if viz:
        render(os.path.join(args.out, "viz", name + ".mp4"), video, ref, shot_frames,
               [(m, s) + viz_runs[(m, s)] for m, s in viz if (m, s) in viz_runs], args.viz_tokens)
    return res


def render(path, video, ref, shot_frames, runs, tokens):
    """Overlay video: one panel per (method, schedule, pred, keyframes), half resolution, 2 columns."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    T = len(ref)
    starts = np.concatenate([[0], np.cumsum(shot_frames)[:-1]]).astype(int)
    shot_of = np.searchsorted(starts, np.arange(T), side="right") - 1
    if tokens:  # 16 px token level, before dilation, per frame (no latent grouping)
        tok = lambda m: np.kron(pool_ge(m[None].astype(np.float32), 16)[0], np.ones((16, 16), bool))
    else:
        tok = lambda m: m
    h, w = H // 2, W // 2
    cols = 2
    n = len(runs)
    rows_ = (n + cols - 1) // cols
    out = av.open(path, "w")
    st = out.add_stream("libx264", rate=16)
    st.width, st.height, st.pix_fmt = cols * w, rows_ * h, "yuv420p"
    st.options = {"crf": "20"}
    with av.open(video) as c:
        for t, fr in enumerate(c.decode(video=0)):
            if t == T:
                break
            rgb = cv2.resize(fr.to_ndarray(format="rgb24"), (W, H), interpolation=cv2.INTER_AREA)
            r = tok(ref[t])
            canvas = np.zeros((rows_ * h, cols * w, 3), np.uint8)
            for i, (m, s, pred, key) in enumerate(runs):
                p = tok(pred[t])
                img = rgb.copy()
                img[p] = (0.55 * img[p] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
                cnt, _ = cv2.findContours(r.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cnt, -1, (40, 255, 40), 3)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                u = (p | r).sum()
                iou = f"{(p & r).sum() / u:.2f}" if u else "-"
                kf = key[:t + 1].nonzero()[0]
                txt = f"{m} {s}  f{t} shot{shot_of[t]}  +{t - kf[-1] if len(kf) else 0}  IoU {iou}"
                cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
                cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if key[t]:
                    cv2.putText(img, "KEY (SAM3)", (w - 105, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA)
                y, x = (i // cols) * h, (i % cols) * w
                canvas[y:y + h, x:x + w] = img
            for pkt in st.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()


def summarize(results, out):
    rows = {}
    for res in results:
        for r in res["runs"]:
            rows.setdefault((r["method"], r["schedule"]), []).append(r)
    summ = []
    print(f"\n{'method':6s} {'sched':9s} {'n':>3s} {'tok_iou':>8s} {'worst':>6s} {'tok_d1':>7s} {'lat_iou':>8s} "
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
        print(f"{m:6s} {s:9s} {len(rs):3d} {row['tok_iou']:8.3f} {row['tok_iou_worst']:6.3f} {row['tok_iou_d1']:7.3f} "
              f"{row['lat_iou']:8.3f} {row['px_iou']:7.3f} {row['sam_per_100f']:9.1f} {row['prop_ms']:6.2f}   {ds}")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)

    cols = [k for k in (("hold", "12"), ("mv", "12"), ("dis", "12"), ("mv", "a48"), ("mv", "shot")) if k in rows]
    print(f"\n{'mask':48s} {'area%':>6s} {'tokens':>7s} {'ref_consec':>10s} " + " ".join(f"{m + ' ' + s:>9s}" for m, s in cols))
    for res in results:
        by = {(r["method"], r["schedule"]): r["tok_iou"] for r in res["runs"]}
        rc = res.get("ref_consec")
        print(f"{res['name'][:48]:48s} {100 * res['ref_area']:6.2f} {res.get('ref_tokens', float('nan')):7.1f} "
              f"{'   -' if rc is None else f'{rc:10.3f}'} " + " ".join(f"{by.get(k, float('nan')):9.3f}" for k in cols))
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
    p.add_argument("--conv", choices=["v0", "v1"], default="v1", help="mask conversion used for the comparison")
    p.add_argument("--viz", default="", help="method:schedule panels for the overlay video, e.g. mv:12,dis:12,hold:12")
    p.add_argument("--viz_tokens", action="store_true", help="draw masks at 16 px token level")
    p.add_argument("--save_pred", default="", help="method:schedule list whose masks are saved, e.g. mv:a48,mv:12")
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
                if npz in seen:
                    continue
                if not os.path.exists(npz):
                    print(f"skip (no SAM3 mask): {npz}", flush=True)
                    continue
                seen.add(npz)
                m = meta[c["clip"]]
                jobs.append((os.path.basename(npz)[:-4], m["video"], npz, m["shot_frames"]))
    print(f"{len(jobs)} masks", flush=True)
    results = []
    for name, video, masks, sf in jobs:
        js = os.path.join(args.out, name + ".json")
        want = [os.path.join(args.out, f"pred_{x.replace(':', '_')}", name + ".npz") for x in args.save_pred.split(",") if x]
        if args.viz:
            want.append(os.path.join(args.out, "viz", name + ".mp4"))
        if os.path.exists(js) and all(os.path.exists(f) for f in want):  # resumable
            results.append(json.load(open(js)))
            continue
        res = evaluate(name, video, masks, sf, args, ffmpeg)
        json.dump(res, open(js, "w"), indent=1)
        results.append(res)
    summarize(results, args.out)


if __name__ == "__main__":
    main()
