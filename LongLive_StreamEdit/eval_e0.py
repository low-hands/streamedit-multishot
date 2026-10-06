"""E0 metrics. All regions come from the SOURCE video's SAM3 mask (v1-style glitch handling not needed:
metrics are per frame and averaged). No edit-type labels are used except `sam3_invert` (background edit,
where the edit region is the complement of the person mask).

Per run (one output .mp4) and per shot:
  strength      CLIP directional similarity on the edit-region crop:
                cos(E(out_crop) - E(src_crop), E(txt_trg) - E(txt_src)); higher = more of the requested edit
  strength_first / strength_rest   same, first chunk of the shot (9 frames) vs the rest
  bg_psnr       PSNR(out, src) outside the edit region dilated by 16 px; bg_psnr_first / _rest likewise
Per run, across shots:
  consist_clip  mean pairwise cosine between per-shot mean CLIP embeddings of the edit-region crops
                (output); consist_clip_src = same on the source (the ceiling set by real viewpoint change)
  consist_lab   mean pairwise distance between per-shot mean Lab colours inside the mask (output; _src likewise)

Usage: python eval_e0.py <root> [<root> ...]       root = e0_runs/main_<sog>/s15 or e0_runs/baseline/s15
Writes <video>_eval.json next to every video and prints one summary line per run.
"""
import glob
import json
import os
import sys

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

POOL = "/root/autodl-tmp/pool_v3"
CLIP = "openai/clip-vit-large-patch14"
FIRST = 9            # frames in the first chunk of a shot (1 + 2 x 4)
MIN_AREA = 0.003     # an object frame counts if the mask covers >= 0.3% of the frame
dev = "cuda"

meta = {m["name"]: m for m in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}
edits = {c["clip"]: c for c in json.load(open(f"{POOL}/edits.json"))}
model = CLIPModel.from_pretrained(CLIP).to(dev).eval().half()
proc = CLIPProcessor.from_pretrained(CLIP)


MEAN = torch.tensor(proc.image_processor.image_mean, device=dev).view(1, 3, 1, 1)
STD = torch.tensor(proc.image_processor.image_std, device=dev).view(1, 3, 1, 1)


def clip_pre(crop):
    """CLIP preprocessing on the GPU: shortest side -> 224 (bicubic), centre crop 224, normalise."""
    x = torch.from_numpy(np.ascontiguousarray(crop)).to(dev).permute(2, 0, 1)[None].float() / 255
    h, w = x.shape[2:]
    s = 224 / min(h, w)
    x = F.interpolate(x, size=(max(224, round(h * s)), max(224, round(w * s))), mode="bicubic", align_corners=False)
    h, w = x.shape[2:]
    y0, x0 = (h - 224) // 2, (w - 224) // 2
    return ((x[:, :, y0:y0 + 224, x0:x0 + 224].clamp(0, 1) - MEAN) / STD)[0]


@torch.no_grad()
def img_emb(crops):
    out = []
    for i in range(0, len(crops), 64):
        x = torch.stack([clip_pre(c) for c in crops[i:i + 64]]).half()
        out.append(F.normalize(model.get_image_features(pixel_values=x).float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def txt_emb(t):
    x = proc(text=[f"a photo of {t}"], return_tensors="pt", padding=True).to(dev)
    return F.normalize(model.get_text_features(**x).float(), dim=-1)[0]


def rgb2lab(x):
    x = x.astype(np.float32) / 255
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    xyz = x @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def crop_box(m, H, W, pad=0.2, min_side=64):
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = max((y1 - y0) * pad, (min_side - (y1 - y0)) / 2, 0), max((x1 - x0) * pad, (min_side - (x1 - x0)) / 2, 0)
    return int(max(y0 - py, 0)), int(min(y1 + py, H)), int(max(x0 - px, 0)), int(min(x1 + px, W))


def pairwise_mean(v, dist):
    n = len(v)
    vals = [dist(v[i], v[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else None


def mean_or_none(x):
    return float(np.mean(x)) if len(x) else None


def evaluate(video, clip, ek):
    m, c = meta[clip], edits[clip]
    e = c["edits"][ek]
    src = iio.imread(m["video"], plugin="FFMPEG")
    out = iio.imread(video, plugin="FFMPEG")
    sam = np.load(f"{POOL}/sam3_masks_v2/{clip}__{e['sam3_prompt'].replace(' ', '_')}.npz")["masks"].astype(bool)
    T = min(len(src), len(out), len(sam))
    H, W = src.shape[1:3]
    inv = bool(e.get("sam3_invert"))
    region = ~sam[:T] if inv else sam[:T]
    d_txt = txt_emb(e["trg_word"]) - txt_emb(e["src_word"])
    d_txt = d_txt / d_txt.norm()

    starts = np.concatenate([[0], np.cumsum(m["shot_frames"])[:-1]]).astype(int)
    shots, emb_out_shot, emb_src_shot, lab_out_shot, lab_src_shot = [], [], [], [], []
    for s, (a, n) in enumerate(zip(starts, m["shot_frames"])):
        fr = [t for t in range(a, min(a + n, T)) if region[t].mean() >= MIN_AREA]
        rec = {"shot": s, "frames": int(min(n, T - a)), "object_frames": len(fr)}
        # background preservation (all frames of the shot), batched on the GPU
        b = min(a + n, T)
        keep = ~(F.max_pool2d(torch.from_numpy(region[a:b]).to(dev).float()[:, None], 33, 1, 16)[:, 0] > 0)
        se = ((torch.from_numpy(out[a:b]).to(dev).float() - torch.from_numpy(src[a:b]).to(dev).float()) ** 2).mean(-1)
        kf = keep.float().sum((1, 2))
        mse = (se * keep).sum((1, 2)) / kf.clamp(min=1)
        ps = (10 * torch.log10(255 ** 2 / mse.clamp(min=1e-6))).cpu().numpy()
        ps[(kf / keep[0].numel()).cpu().numpy() < 0.05] = np.nan
        rec["bg_psnr"] = float(np.nanmean(ps)) if np.isfinite(ps).any() else None
        rec["bg_psnr_first"] = float(np.nanmean(ps[:FIRST])) if np.isfinite(ps[:FIRST]).any() else None
        rec["bg_psnr_rest"] = float(np.nanmean(ps[FIRST:])) if np.isfinite(ps[FIRST:]).any() else None
        if fr:
            boxes = [(0, H, 0, W) if inv else crop_box(region[t], H, W) for t in fr]
            eo = img_emb([out[t][y0:y1, x0:x1] for t, (y0, y1, x0, x1) in zip(fr, boxes)])
            es = img_emb([src[t][y0:y1, x0:x1] for t, (y0, y1, x0, x1) in zip(fr, boxes)])
            dirs = F.normalize(eo - es, dim=-1) @ d_txt
            first = np.array([t - a < FIRST for t in fr])
            dn = dirs.cpu().numpy()
            rec["strength"] = float(dn.mean())
            rec["strength_first"] = mean_or_none(dn[first])
            rec["strength_rest"] = mean_or_none(dn[~first])
            emb_out_shot.append(F.normalize(eo.mean(0), dim=-1)); emb_src_shot.append(F.normalize(es.mean(0), dim=-1))
            lab_out_shot.append(np.mean([rgb2lab(out[t][region[t]]).mean(0) for t in fr], 0))
            lab_src_shot.append(np.mean([rgb2lab(src[t][region[t]]).mean(0) for t in fr], 0))
        shots.append(rec)
    cos = lambda u, v: float(u @ v)
    l2 = lambda u, v: float(np.linalg.norm(u - v))
    res = {"video": video, "clip": clip, "edit": ek, "type": e["type"], "shots": shots,
           "consist_clip": pairwise_mean(emb_out_shot, cos), "consist_clip_src": pairwise_mean(emb_src_shot, cos),
           "consist_lab": pairwise_mean(lab_out_shot, l2), "consist_lab_src": pairwise_mean(lab_src_shot, l2)}
    obj = [r for r in shots if r.get("strength") is not None]
    res["strength"] = mean_or_none([r["strength"] for r in obj])
    # first chunk after a CUT (shots >= 1) vs the rest of those shots
    res["strength_first_after_cut"] = mean_or_none([r["strength_first"] for r in obj if r["shot"] > 0 and r["strength_first"] is not None])
    res["strength_rest_after_cut"] = mean_or_none([r["strength_rest"] for r in obj if r["shot"] > 0 and r["strength_rest"] is not None])
    bg = [r for r in shots if r["bg_psnr"] is not None]
    res["bg_psnr"] = mean_or_none([r["bg_psnr"] for r in bg])
    res["bg_psnr_first_after_cut"] = mean_or_none([r["bg_psnr_first"] for r in bg if r["shot"] > 0 and r["bg_psnr_first"] is not None])
    res["bg_psnr_rest_after_cut"] = mean_or_none([r["bg_psnr_rest"] for r in bg if r["shot"] > 0 and r["bg_psnr_rest"] is not None])
    return res


def fmt(x, p=3):
    return "  -  " if x is None else f"{x:.{p}f}"


for root in sys.argv[1:]:
    for video in sorted(glob.glob(f"{root}/*/*/*.mp4")):
        name = os.path.basename(video)
        if any(k in name for k in ("overlay", "cmp_", "_eval")):
            continue
        clip, ek = video.split("/")[-3], video.split("/")[-2]
        if clip not in edits:
            continue
        r = evaluate(video, clip, ek)
        json.dump(r, open(video[:-4] + "_eval.json", "w"), indent=1)
        print(f"{clip[:22]:22s} {ek:11s} {name[:-4]:22s} str {fmt(r['strength'])} "
              f"(cut1st {fmt(r['strength_first_after_cut'])} / rest {fmt(r['strength_rest_after_cut'])})  "
              f"cons {fmt(r['consist_clip'])} [src {fmt(r['consist_clip_src'])}]  "
              f"lab {fmt(r['consist_lab'], 1)} [src {fmt(r['consist_lab_src'], 1)}]  "
              f"bg {fmt(r['bg_psnr'], 1)} (cut1st {fmt(r['bg_psnr_first_after_cut'], 1)} / rest {fmt(r['bg_psnr_rest_after_cut'], 1)})",
              flush=True)
