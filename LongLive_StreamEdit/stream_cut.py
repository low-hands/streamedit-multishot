"""Online (causal) hard-cut detection for streaming editing, and its check against the manifest cuts.

Detector (frame t is decided when frame t + lookahead has arrived; lookahead 0 or 1):
  content(t)  mean |HSV(t) - HSV(t-1)| over pixels and channels at 160 x 90 (PySceneDetect ContentDetector)
  cut at t    content(t) >= min_content  and  content(t) >= ratio x mean(content of the previous `window` frames)
              and (lookahead 1) |HSV(t+1) - HSV(t-1)| >= min_content too: a flash comes back, a cut does not
              and at least min_shot frames since the previous cut
codec cue: x264 (zerolatency, as a live encoder) inserts an I-frame at a scene change; `codec` reports a
cut at every I-frame after the first, `hsv+codec` requires the HSV rule OR the I-frame (union).

A cut at t means frame t is the first frame of a new shot.

Check (no GPU):  python stream_cut.py [--tol 1] [--clips a,b]
  P / R / F1 against manifest_aligned_v2 cuts (frame offsets of shot starts), ms/frame of the HSV rule.
"""
import argparse
import json
import os
import subprocess
import tempfile
import time

import av
import cv2
import numpy as np

SMALL = (160, 90)


class CutDetector:
    """Feed frames one by one with push(); returns the cut frame indices decided so far (each once)."""

    def __init__(self, min_content=20.0, ratio=3.0, window=8, lookahead=1, min_shot=8):
        self.min_content, self.ratio, self.window = min_content, ratio, window
        self.lookahead, self.min_shot = lookahead, min_shot
        self.hsv, self.content, self.last_cut = [], [], 0

    def push(self, rgb):
        """rgb: HxWx3 uint8 frame. Returns a list with the newly decided cut (0 or 1 element)."""
        h = cv2.cvtColor(cv2.resize(rgb, SMALL, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV).astype(np.int16)
        self.hsv.append(h)
        t = len(self.hsv) - 1
        self.content.append(0.0 if t == 0 else float(np.abs(h - self.hsv[t - 1]).mean()))
        if len(self.hsv) > self.window + 3:  # keep only what the rule needs
            self.hsv[t - self.window - 3] = None
        c = t - self.lookahead  # frame being decided
        if c < 1 or c - self.last_cut < self.min_shot:
            return []
        v = self.content[c]
        prev = self.content[max(1, c - self.window):c]
        base = float(np.mean(prev)) if prev else 0.0
        ok = v >= self.min_content and v >= self.ratio * base
        if ok and self.lookahead:
            ok = float(np.abs(self.hsv[c + 1] - self.hsv[c - 1]).mean()) >= self.min_content
        if ok:
            self.last_cut = c
            return [c]
        return []

    def flush(self):
        """End of stream: the last `lookahead` frames are not decided (a cut there has no next frame)."""
        return []


def codec_cuts(video, ffmpeg, T):
    """I-frames (after the first) of a live-style x264 re-encode = the encoder's own scene-change detection."""
    with tempfile.TemporaryDirectory() as d:
        enc = os.path.join(d, "p.mp4")
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", video, "-an", "-frames:v", str(T),
                        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-bf", "0", "-refs", "1",
                        "-g", "100000", "-crf", "18", enc], check=True)
        with av.open(enc) as c:
            return [t for t, fr in enumerate(c.decode(video=0)) if t > 0 and fr.pict_type.name == "I"]


def score(pred, gt, tol):
    """greedy one-to-one matching within +-tol frames -> (tp, fp, fn)."""
    gt_left = list(gt)
    tp = 0
    for p in pred:
        m = [g for g in gt_left if abs(g - p) <= tol]
        if m:
            gt_left.remove(min(m, key=lambda g: abs(g - p)))
            tp += 1
    return tp, len(pred) - tp, len(gt_left)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/root/autodl-tmp/pool_v3")
    ap.add_argument("--clips", default="")
    ap.add_argument("--video", default=None, help="single video; --cuts gives its true cut frames")
    ap.add_argument("--cuts", default="")
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--min_content", type=float, default=20.0)
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--lookahead", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    if args.video:
        jobs = [(os.path.basename(args.video), args.video, [int(x) for x in args.cuts.split(",") if x], None)]
    else:
        meta = json.load(open(f"{args.pool}/manifest_aligned_v2.json"))
        keep = set(args.clips.split(",")) if args.clips else None
        jobs = [(m["name"], m["video"], np.cumsum(m["shot_frames"])[:-1].astype(int).tolist(), sum(m["shot_frames"]))
                for m in meta if not keep or m["name"] in keep]
    tot = {k: np.zeros(3, int) for k in ("hsv", "codec", "hsv+codec")}
    ms = []
    for name, video, gt, T in jobs:
        det = CutDetector(args.min_content, args.ratio, lookahead=args.lookahead)
        pred, n = [], 0
        with av.open(video) as c:
            for fr in c.decode(video=0):
                if T is not None and n == T:
                    break
                rgb = fr.to_ndarray(format="rgb24")
                t0 = time.perf_counter()
                pred += det.push(rgb)
                ms.append((time.perf_counter() - t0) * 1e3)
                n += 1
        cod = codec_cuts(video, ffmpeg, n)
        both = sorted(set(pred) | set(cod))
        line = []
        for k, p in (("hsv", pred), ("codec", cod), ("hsv+codec", both)):
            s = score(p, gt, args.tol)
            tot[k] += s
            line.append(f"{k} tp{s[0]} fp{s[1]} fn{s[2]}")
        print(f"{name[:40]:40s} frames {n:4d} cuts {gt}  hsv {pred}  codec {cod}  | " + "  ".join(line), flush=True)
    print(f"\n{len(jobs)} videos, tolerance +-{args.tol} frames, lookahead {args.lookahead}, "
          f"HSV rule {np.mean(ms):.2f} ms/frame")
    for k, (tp, fp, fn) in tot.items():
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"  {k:10s} P {p:.3f}  R {r:.3f}  F1 {2 * p * r / max(p + r, 1e-9):.3f}   (tp {tp} fp {fp} fn {fn})")


if __name__ == "__main__":
    main()
