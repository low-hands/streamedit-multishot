"""Three facts to settle before designing the edit memory (docs/memory_rethink_2026-10-12.md).

A  cross-shot inconsistency without memory (H1), from existing outputs (default name C.mp4).
   The edit region of every shot is cut into chunk segments (first 9 frames, then 12); each segment gets the
   mean CLIP embedding of the region crops and the mean Lab colour of the region. Distances between segments
   of the SAME shot (within) and of DIFFERENT shots (cross), for the output and for the source:
     excess = (cross_out - within_out) - (cross_src - within_src)
   i.e. how much more the edit differs across shots than inside a shot, beyond what the source itself does
   (viewpoint, lighting). CLIP distance = 1 - cos; Lab distance = L2. Grouped by edit type.
B  does the raw memory (memcheck M: memory tokens keep their old spatial position) leak the old layout?
   Per frame of shots >= 1, mean |M - C| (0-255) in three regions:
     cur    the current edit region (dilated 16 px)            -> what the memory changes on the object
     old    where the object was in earlier shots, not cur     -> leakage would show up here
     other  the rest (away from both, dilated 16 px)
   Leakage: old >> other. Also listed for the first chunk after each cut alone.
C  can the source branch locate a token inside the edit region? Source-only clean pass of the Wan model
   (as the first call of EditStreamSession.step), pre-RoPE self-attention keys of some layers. For two
   consecutive latent frames of a shot, every interior token of frame f is matched (cosine) against the
   interior tokens of frame f-1:
     nn      distance (tokens) from the best match to the token's own position, after removing the shift of
             the region centroid
     spread  mean distance of the top-5 matches to their centroid
   both divided by the same quantity for random matches (1.0 = no better than chance), overall and by the
   texture of the token's 16 px patch (grey std; terciles per clip: low / mid / high).

Run from LongLive_StreamEdit/ (HF_HUB_OFFLINE=1 for the CLIP weights):
  python memory_facts.py A --roots /root/autodl-tmp/e0_runs/maskprop_edit_v0/s15,/root/autodl-tmp/e0_runs/memcheck/s15
  python memory_facts.py B --roots /root/autodl-tmp/e0_runs/memcheck/s15
  python memory_facts.py C [--clips a,b] [--layers 5,15,25]       (needs the GPU and the Wan model)
Writes <out>/facts_<part>.json; prints tables.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

POOL = "/root/autodl-tmp/pool_v3"
OUT = "/root/autodl-tmp/e0_runs/memory_facts"
FIRST = 9
MIN_AREA = 0.003
dev = "cuda"


def load_meta(pool):
    meta = {m["name"]: m for m in json.load(open(f"{pool}/manifest_aligned_v2.json"))}
    edits = {c["clip"]: c for c in json.load(open(f"{pool}/edits.json"))}
    return meta, edits


def sam_region(pool, clip, e):
    sam = np.load(f"{pool}/sam3_masks_v2/{clip}__{e['sam3_prompt'].replace(' ', '_')}.npz")["masks"].astype(bool)
    return ~sam if e.get("sam3_invert") else sam


def read_video(path):
    import imageio.v3 as iio
    return iio.imread(path, plugin="FFMPEG")


def shot_starts(m):
    return np.concatenate([[0], np.cumsum(m["shot_frames"])[:-1]]).astype(int)


def dilate(m, r):
    """bool [T, H, W] -> dilated by r px (square), on the GPU."""
    x = torch.from_numpy(np.ascontiguousarray(m)).to(dev).float()[:, None]
    return (F.max_pool2d(x, 2 * r + 1, 1, r)[:, 0] > 0).cpu().numpy()


# ---------------------------------------------------------------------------------------------------- A
class Clip:
    def __init__(self, name="openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).to(dev).eval().half()
        p = CLIPProcessor.from_pretrained(name)
        self.mean = torch.tensor(p.image_processor.image_mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(p.image_processor.image_std, device=dev).view(1, 3, 1, 1)

    def pre(self, crop):
        x = torch.from_numpy(np.ascontiguousarray(crop)).to(dev).permute(2, 0, 1)[None].float() / 255
        h, w = x.shape[2:]
        s = 224 / min(h, w)
        x = F.interpolate(x, size=(max(224, round(h * s)), max(224, round(w * s))), mode="bicubic",
                          align_corners=False)
        h, w = x.shape[2:]
        y0, x0 = (h - 224) // 2, (w - 224) // 2
        return ((x[:, :, y0:y0 + 224, x0:x0 + 224].clamp(0, 1) - self.mean) / self.std)[0]

    @torch.no_grad()
    def emb(self, crops):
        out = []
        for i in range(0, len(crops), 64):
            x = torch.stack([self.pre(c) for c in crops[i:i + 64]]).half()
            out.append(F.normalize(self.model.get_image_features(pixel_values=x).float(), dim=-1))
        return torch.cat(out).cpu().numpy()


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
    py = max((y1 - y0) * pad, (min_side - (y1 - y0)) / 2, 0)
    px = max((x1 - x0) * pad, (min_side - (x1 - x0)) / 2, 0)
    return int(max(y0 - py, 0)), int(min(y1 + py, H)), int(max(x0 - px, 0)), int(min(x1 + px, W))


def segments(m, T, region):
    """[(shot, [object frames])] per chunk segment (first 9 frames of a shot, then 12); >= 3 object frames."""
    segs = []
    for s, (a, n) in enumerate(zip(shot_starts(m), m["shot_frames"])):
        b = min(a + n, T)
        edges = [a, min(a + FIRST, b)] + list(range(a + FIRST + 12, b, 12)) + [b]
        for lo, hi in zip(edges[:-1], edges[1:]):
            fr = [t for t in range(lo, hi) if region[t].mean() >= MIN_AREA]
            if len(fr) >= 3:
                segs.append((s, fr))
    return segs


def within_cross(feats, shots, dist):
    w, c = [], []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            (w if shots[i] == shots[j] else c).append(dist(feats[i], feats[j]))
    return (float(np.mean(w)) if w else None), (float(np.mean(c)) if c else None)


def part_a(args, meta, edits):
    clip_model = Clip()
    rows = []
    for root in args.roots.split(","):
        for video in sorted(glob.glob(f"{root}/*/*/{args.name}.mp4")):
            clip, ek = video.split("/")[-3], video.split("/")[-2]
            if clip not in edits or clip not in meta or ek not in edits[clip]["edits"]:
                continue
            m, e = meta[clip], edits[clip]["edits"][ek]
            src, out = read_video(m["video"]), read_video(video)
            region = sam_region(args.pool, clip, e)
            T = min(len(src), len(out), len(region))
            H, W = src.shape[1:3]
            segs = segments(m, T, region)
            if len({s for s, _ in segs}) < 2:
                continue
            res = {"video": video, "clip": clip, "edit": ek, "type": e["type"], "segments": len(segs)}
            for tag, vid in (("out", out), ("src", src)):
                embs, labs = [], []
                for _, fr in segs:
                    boxes = [(0, H, 0, W) if e.get("sam3_invert") else crop_box(region[t], H, W) for t in fr]
                    em = clip_model.emb([vid[t][y0:y1, x0:x1] for t, (y0, y1, x0, x1) in zip(fr, boxes)]).mean(0)
                    embs.append(em / np.linalg.norm(em))
                    labs.append(np.mean([rgb2lab(vid[t][region[t]]).mean(0) for t in fr], 0))
                sh = [s for s, _ in segs]
                res[f"clip_{tag}"] = within_cross(embs, sh, lambda u, v: float(1 - u @ v))
                res[f"lab_{tag}"] = within_cross(labs, sh, lambda u, v: float(np.linalg.norm(u - v)))
            for k in ("clip", "lab"):
                (wo, co), (ws, cs) = res[f"{k}_out"], res[f"{k}_src"]
                res[f"{k}_excess"] = None if None in (wo, co, ws, cs) else (co - wo) - (cs - ws)
            rows.append(res)
            f = lambda x, p=3: "  -  " if x is None else f"{x:.{p}f}"
            print(f"{clip[:24]:24s} {ek:11s} {e['type']:11s} {root.rstrip('/').split('/')[-2]:16s} "
                  f"CLIP within/cross out {f(res['clip_out'][0])}/{f(res['clip_out'][1])} src {f(res['clip_src'][0])}/{f(res['clip_src'][1])} "
                  f"excess {f(res['clip_excess'])} | Lab out {f(res['lab_out'][0], 1)}/{f(res['lab_out'][1], 1)} "
                  f"src {f(res['lab_src'][0], 1)}/{f(res['lab_src'][1], 1)} excess {f(res['lab_excess'], 1)}", flush=True)
    print("\nby edit type (mean excess; > 0: the edit differs more across shots than the source explains)")
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    for t, rs in sorted(by.items()):
        ce = [r["clip_excess"] for r in rs if r["clip_excess"] is not None]
        le = [r["lab_excess"] for r in rs if r["lab_excess"] is not None]
        print(f"  {t:12s} n={len(rs):2d}  CLIP excess {np.mean(ce):+.4f} (min {np.min(ce):+.4f} max {np.max(ce):+.4f})  "
              f"Lab excess {np.mean(le):+.2f} (min {np.min(le):+.2f} max {np.max(le):+.2f})")
    return rows


# ---------------------------------------------------------------------------------------------------- B
def part_b(args, meta, edits):
    rows = []
    for root in args.roots.split(","):
        for d in sorted(glob.glob(f"{root}/*/*/")):
            c_mp4, m_mp4 = f"{d}C.mp4", f"{d}M.mp4"
            clip, ek = d.rstrip("/").split("/")[-2:]
            if not (os.path.exists(c_mp4) and os.path.exists(m_mp4)) or clip not in edits or clip not in meta:
                continue
            m, e = meta[clip], edits[clip]["edits"][ek]
            C, M = read_video(c_mp4), read_video(m_mp4)
            region = sam_region(args.pool, clip, e)
            T = min(len(C), len(M), len(region))
            region = region[:T]
            cur = dilate(region, 16)
            starts = shot_starts(m)
            res = {"dir": d, "clip": clip, "edit": ek, "type": e["type"], "shots": []}
            for s in range(1, len(starts)):
                a, b = starts[s], min(starts[s] + m["shot_frames"][s], T)
                if a >= T:
                    break
                old = region[:a].any(0)
                if not old.any():
                    continue
                old_d = dilate(old[None], 16)[0]
                acc = {k: [] for k in ("cur", "old", "other", "cur1", "old1", "other1")}
                for t in range(a, b):
                    diff = np.abs(M[t].astype(np.float32) - C[t].astype(np.float32)).mean(-1)
                    regs = {"cur": cur[t], "old": old & ~cur[t], "other": ~old_d & ~cur[t]}
                    for k, r in regs.items():
                        if r.mean() > 0.002:
                            acc[k].append(float(diff[r].mean()))
                            if t - a < FIRST:
                                acc[k + "1"].append(float(diff[r].mean()))
                res["shots"].append({"shot": s, **{k: (float(np.mean(v)) if v else None) for k, v in acc.items()}})
            rows.append(res)
            f = lambda x: "  -  " if x is None else f"{x:5.2f}"
            for r in res["shots"]:
                ratio = (r["old"] / r["other"]) if r["old"] is not None and r["other"] else None
                print(f"{clip[:24]:24s} {ek:9s} shot {r['shot']}  |M-C| cur {f(r['cur'])}  old {f(r['old'])}  "
                      f"other {f(r['other'])}  old/other {f(ratio)}   first chunk: cur {f(r['cur1'])} "
                      f"old {f(r['old1'])} other {f(r['other1'])}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------------------- C
def token_masks(region_shot, n_lat):
    """frames of one shot [n, H, W] -> interior token masks [L, 30, 52] (latent group mean >= 0.5 per 16 px)."""
    g = [[0]] + [list(range(4 * k - 3, 4 * k + 1)) for k in range(1, n_lat)]
    x = torch.from_numpy(region_shot).float()
    px = torch.stack([x[i].mean(0) for i in g])
    return (F.avg_pool2d(px[:, None], 16)[:, 0] >= 0.5).numpy(), g


def patch_texture(gray_frame):
    """grey [H, W] uint8 -> std per 16 px patch [30, 52]."""
    x = torch.from_numpy(gray_frame.astype(np.float32))[None, None]
    m = F.avg_pool2d(x, 16)
    return (F.avg_pool2d(x ** 2, 16) - m ** 2).clamp(min=0).sqrt()[0, 0].numpy()


def match_stats(kq, kp, pq, pp, rng):
    """kq [N, D], kp [M, D] keys; pq [N, 2], pp [M, 2] token positions (already centroid-aligned)."""
    sim = F.normalize(kq.float(), dim=-1) @ F.normalize(kp.float(), dim=-1).T
    top = sim.topk(min(5, kp.shape[0]), dim=1).indices.cpu().numpy()
    ppn = pp
    nn = np.linalg.norm(ppn[top[:, 0]] - pq, axis=1)
    tops = ppn[top]
    spread = np.linalg.norm(tops - tops.mean(1, keepdims=True), axis=2).mean(1)
    rnd = rng.integers(0, len(pp), size=(len(pq), top.shape[1]))
    nn_r = np.linalg.norm(ppn[rnd[:, 0]] - pq, axis=1)
    tr = ppn[rnd]
    spread_r = np.linalg.norm(tr - tr.mean(1, keepdims=True), axis=2).mean(1)
    return nn, spread, nn_r, spread_r


def part_c(args, meta, edits):
    import cv2
    from diffusers.utils import load_video
    import inference_edit_streamedit as ies
    from pipeline.edit_stream import EditStreamSession
    from stream_edit import TRANSFORM

    layers = [int(x) for x in args.layers.split(",")]
    margs = ies.build_parser().parse_args(
        ["--data_path", "x", "--save_path", "x", "--src_prompt", "x", "--trg_prompt", "x", "--src_word", "x",
         "--trg_word", "x", "--step", "15", "--seed", "0", "--force_low_memory", "off"])
    pipe, low_memory, device, _ = ies.setup(margs)
    grab = {}
    blocks = pipe.generator.model.blocks
    hooks = [blocks[li].self_attn.norm_k.register_forward_hook(
        lambda mod, inp, out, li=li: grab.__setitem__(li, out[0].detach())) for li in layers]
    rng = np.random.default_rng(0)
    keep = set(args.clips.split(",")) if args.clips else None
    rows = []
    for clip, c in edits.items():
        if clip not in meta or (keep and clip not in keep):
            continue
        m = meta[clip]
        for ek, e in c["edits"].items():
            if e.get("sam3_invert"):
                continue  # background edits: the region is the complement of a person
            region = sam_region(args.pool, clip, e)
            frames = load_video(m["video"])
            gray = [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY) for f in frames]
            acc = {li: defaultdict(list) for li in layers}
            for s, (a, n) in enumerate(zip(shot_starts(m), m["shot_frames"])):
                if a + n > min(len(frames), len(region)):
                    break
                n_lat = 1 + (n - 1) // 4
                tm, groups = token_masks(region[a:a + n], n_lat)
                if tm.sum(axis=(1, 2)).max() < 8:
                    continue
                px = torch.stack([TRANSFORM(f) for f in frames[a:a + n]], 1)[None].to(device, torch.bfloat16)
                pipe.vae.model.clear_cache()
                z = pipe.vae.encode_to_latent(px).to(device, torch.bfloat16)
                pipe.vae.model.clear_cache()
                sess = EditStreamSession(pipe, c["src_prompt"], e["trg_prompt"], e["src_word"], e["trg_word"],
                                         device, low_memory=low_memory)
                keys = {li: [] for li in layers}
                for i in range(0, n_lat, 3):
                    x = z[:, i:i + 3]
                    ts = torch.ones([1, x.shape[1]], device=device) * pipe.args.context_noise
                    pipe._register_crossattn_mask_gatherer(sess.crossattn_cache_src, sess.tok_src,
                                                           layers=sess.mask_layers, fg_scale=sess.fg_scale)
                    with torch.no_grad():
                        pipe.generator(noisy_image_or_video=x, conditional_dict=sess.src_conditional_dict,
                                       timestep=ts, kv_cache=sess.kv_cache_src, crossattn_cache=sess.crossattn_cache_src,
                                       current_start=i * pipe.frame_seq_length)
                    for li in layers:
                        keys[li].append(grab[li][:x.shape[1] * 30 * 52].reshape(x.shape[1], 30 * 52, -1))  # drop padding
                keys = {li: torch.cat(v) for li, v in keys.items()}
                tex = np.stack([patch_texture(gray[a + g[-1]]) for g in groups])
                lo, hi = np.quantile(tex[tm], [1 / 3, 2 / 3]) if tm.any() else (0, 0)
                pos = np.stack(np.meshgrid(np.arange(30), np.arange(52), indexing="ij"), -1).reshape(-1, 2).astype(float)
                for f_ in range(1, n_lat):
                    mq, mp = tm[f_].reshape(-1), tm[f_ - 1].reshape(-1)
                    if mq.sum() < 8 or mp.sum() < 8:
                        continue
                    shift = pos[mq].mean(0) - pos[mp].mean(0)
                    pq, pp = pos[mq] - shift, pos[mp]
                    t_q = tex[f_].reshape(-1)[mq]
                    band = np.where(t_q < lo, 0, np.where(t_q < hi, 1, 2))
                    for li in layers:
                        nn, sp, nn_r, sp_r = match_stats(keys[li][f_][torch.from_numpy(mq).to(device)],
                                                         keys[li][f_ - 1][torch.from_numpy(mp).to(device)], pq, pp, rng)
                        for b_ in (0, 1, 2, None):
                            sel = slice(None) if b_ is None else band == b_
                            tag = "all" if b_ is None else ("low", "mid", "high")[b_]
                            for k, v in (("nn", nn), ("sp", sp), ("nn_r", nn_r), ("sp_r", sp_r)):
                                acc[li][f"{tag}_{k}"] += v[sel].tolist()
                del sess, keys, z, px
                torch.cuda.empty_cache()
            res = {"clip": clip, "edit": ek, "type": e["type"], "layers": {}}
            line = []
            for li in layers:
                r = {}
                for tag in ("all", "low", "mid", "high"):
                    if acc[li][f"{tag}_nn"]:
                        r[tag] = {"nn": float(np.mean(acc[li][f"{tag}_nn"]) / np.mean(acc[li][f"{tag}_nn_r"])),
                                  "spread": float(np.mean(acc[li][f"{tag}_sp"]) / np.mean(acc[li][f"{tag}_sp_r"])),
                                  "nn_tokens": float(np.mean(acc[li][f"{tag}_nn"])), "n": len(acc[li][f"{tag}_nn"])}
                res["layers"][li] = r
                if "all" in r:
                    line.append(f"L{li} nn " + " ".join(f"{t}:{r[t]['nn']:.2f}" for t in ("all", "low", "mid", "high") if t in r)
                                + f" spread {r['all']['spread']:.2f}")
            rows.append(res)
            print(f"{clip[:24]:24s} {ek:11s} {e['type']:11s} " + " | ".join(line), flush=True)
    for h in hooks:
        h.remove()
    print("\nby edit type, nn / chance (1.0 = random; low / mid / high texture tercile)")
    for li in layers:
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for tag, v in r["layers"].get(li, {}).items():
                by[r["type"]][tag].append(v["nn"])
        for t, d in sorted(by.items()):
            print(f"  L{li:2d} {t:12s} " + "  ".join(f"{tag} {np.mean(d[tag]):.2f}" for tag in ("all", "low", "mid", "high") if d[tag]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["A", "B", "C"])
    ap.add_argument("--roots", default="/root/autodl-tmp/e0_runs/maskprop_edit_v0/s15")
    ap.add_argument("--name", default="C", help="A: which output to measure (<name>.mp4)")
    ap.add_argument("--clips", default="")
    ap.add_argument("--layers", default="5,15,25")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    meta, edits = load_meta(args.pool)
    os.makedirs(args.out, exist_ok=True)
    rows = {"A": part_a, "B": part_b, "C": part_c}[args.part](args, meta, edits)
    json.dump(rows, open(f"{args.out}/facts_{args.part}.json", "w"), indent=1, default=str)
    print(f"wrote {args.out}/facts_{args.part}.json")
