"""Streaming multi-shot editing: frames come in one at a time, cuts are decided on the spot.

Per incoming frame t (only frames <= t are used):
  codec   the frame is encoded with x264 zerolatency (P-frames only, 1 reference) and decoded right away:
          an I-frame = the first frame of a new shot (the encoder's scene-change decision, no lookahead),
          the motion vectors are kept for the mask
  mask    --mask prop (default, docs/mask_propagation_2026-10-11.md): SAM3 on keyframes (shot start, area
          change > x1.15, at most 16 frames apart, re-check every 12 frames while empty) + motion-vector
          propagation. The SAM3 result of a keyframe is read from the per-frame SAM3 npz, only at keyframes.
          --mask sam uses the per-frame SAM3 mask on every frame (reference / equivalence check).
  chunk   frames gather into the current shot's chunk: 9 frames (latents 1 + 4 + 4) for its first chunk,
          12 afterwards. A full chunk is VAE-encoded (causal caches), edited by EditStreamSession.step
          (the StreamEdit loop body: mask v0, dilate 1, SOG gate, reset none by default) and VAE-decoded.
  cut     on a cut the open chunk of the old shot is completed by repeating its last frame, processed, and
          only its real frames are emitted; VAE caches restart and the new shot starts a fresh chunk grid.

Source: --clip/--edit from the pool (prompts from edits.json; --k shortens every shot by k tail frames so
cuts land anywhere in a chunk, as in run_online_cut.py), or --data_path with explicit prompts.
Writes --save_path, <save_path>_stream.json (detected cuts, keyframes, per-chunk latency, background PSNR
near / away from cuts), and with --compare <video> the per-frame difference to another output
(e.g. the batch C.mp4 of run_maskprop_edit.py: with --mask sam --k 0 the two should match).

Run from LongLive_StreamEdit/:
  python stream_edit.py --clip 226154551_5_viewpoint_4shot_12.6s --edit replacement --k 3 --save_path out.mp4
"""
import argparse
import json
import os
import time
from fractions import Fraction

import av
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms
from torchvision.io import write_video

import inference_edit_streamedit as ies
import mask_prop_eval as mpe
from pipeline.edit_stream import EditStreamSession, StreamVAE
from utils.misc import set_seed

POOL = "/root/autodl-tmp/pool_v3"
H, W = 480, 832
TRANSFORM = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


class LiveCodec:
    """x264 zerolatency encoder + h264 decoder with motion-vector export; one frame in, one frame out."""

    def __init__(self):
        e = av.CodecContext.create("libx264", "w")
        e.width, e.height, e.pix_fmt = W, H, "yuv420p"
        e.time_base, e.framerate = Fraction(1, 16), Fraction(16, 1)
        e.options = {"preset": "veryfast", "tune": "zerolatency", "bf": "0", "refs": "1", "g": "100000", "crf": "18"}
        d = av.CodecContext.create("h264", "r")
        d.options = {"flags2": "+export_mvs"}
        d.thread_count = 1  # frame threads would delay the output
        self.enc, self.dec, self.n = e, d, 0

    def push(self, rgb):
        """-> (is_keyframe, motion-vector cells [H/4, W/4, 2] or None)."""
        vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        vf.pts, self.n = self.n, self.n + 1
        out = [(pkt.is_keyframe, fr) for pkt in self.enc.encode(vf) for fr in self.dec.decode(pkt)]
        assert len(out) == 1, f"codec delay: {len(out)} frames out for frame {self.n - 1}"
        key, fr = out[0]
        sd = fr.side_data.get("MOTION_VECTORS")
        return key, (None if sd is None else mpe.mv_cells(sd.to_ndarray()))


class PropMask:
    """keyframe SAM3 + motion-vector propagation, schedule a<max_gap>r<ratio> (as mask_prop_eval.run)."""

    def __init__(self, sam_at, max_gap=16, ratio=1.15, search=12):
        self.sam_at, self.max_gap, self.ratio, self.search = sam_at, max_gap, ratio, search
        self.soft, self.last_key, self.key_area, self.keyframes = None, 0, 0, []

    def push(self, t, shot_start, cells):
        gap = t - self.last_key
        area = 0 if self.soft is None else int((self.soft > 0.5).sum())
        if shot_start:
            key = True
        elif area == 0:
            key = gap >= self.search
        else:
            key = gap >= self.max_gap or (self.key_area > 0 and not (1 / self.ratio <= area / self.key_area <= self.ratio))
        if key:
            m = self.sam_at(t)  # the only place SAM3 is consulted
            self.soft, self.last_key, self.key_area = m.astype(np.float32), t, int(m.sum())
            self.keyframes.append(t)
        elif cells is not None:
            self.soft = mpe.warp(self.soft, cv2.resize(cells, (W, H), interpolation=cv2.INTER_NEAREST))
        return self.soft > 0.5


def token_mask_v0(masks, first, invert, device):
    """--oracle_version v0 for one chunk: union over the frames of a latent, 16 px token covered > 10%,
    dilate 1 token. masks bool [T, H, W] -> [L, 30 * 52] bool."""
    T = len(masks)
    groups = ([[0]] + [list(range(4 * k - 3, 4 * k + 1)) for k in range(1, (T - 1) // 4 + 1)]) if first \
        else [list(range(4 * k, 4 * k + 4)) for k in range(T // 4)]
    m = torch.from_numpy(masks).float()
    lat = torch.stack([m[g].amax(0) for g in groups])
    t = F.avg_pool2d(lat[:, None], 16)[:, 0] > 0.1
    t = F.max_pool2d(t.float()[:, None], 3, 1, 1)[:, 0] > 0
    if invert:
        t = ~t
    return t.reshape(len(groups), -1).to(device)


def frame_source(args, meta, edits):
    """yields (t, pil, original frame index); --clip mode drops k tail frames of every shot."""
    if args.clip:
        m = meta[args.clip]
        sf = m["shot_frames"]
        starts = np.concatenate([[0], np.cumsum(sf)[:-1]]).astype(int)
        keep = set(np.concatenate([np.arange(a, a + n - min(args.k, n - 5)) for a, n in zip(starts, sf)]).tolist())
        path, total = m["video"], sum(sf)
    else:
        path, keep, total = args.data_path, None, None
    t = 0
    for i, frame in enumerate(imageio.get_reader(path)):
        if total is not None and i >= total:
            break
        if keep is None or i in keep:
            yield t, Image.fromarray(frame), i
            t += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--masks", default=None, help="per-frame SAM3 npz (default: the pool's for --clip/--edit)")
    for a in ("src_prompt", "trg_prompt", "src_word", "trg_word"):
        ap.add_argument(f"--{a}", default=None)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--mask", choices=["prop", "sam"], default="prop")
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--reset_at_cut", choices=["none", "all", "keep_sink"], default="none")
    ap.add_argument("--edit_mem", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--pool", default=POOL)
    args = ap.parse_args()

    meta = {m["name"]: m for m in json.load(open(f"{args.pool}/manifest_aligned_v2.json"))}
    edits = {c["clip"]: c for c in json.load(open(f"{args.pool}/edits.json"))}
    if args.clip:
        c, e = edits[args.clip], edits[args.clip]["edits"][args.edit]
        args.src_prompt, args.trg_prompt = c["src_prompt"], e["trg_prompt"]
        args.src_word, args.trg_word = e["src_word"], e["trg_word"]
        args.invert = bool(e.get("sam3_invert"))
        args.masks = args.masks or f"{args.pool}/sam3_masks_v2/{args.clip}__{e['sam3_prompt'].replace(' ', '_')}.npz"
    sam_all = np.load(args.masks)["masks"]

    parser = ies.build_parser()
    margs = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x", "--trg_prompt", "x",
                               "--src_word", "x", "--trg_word", "x", "--step", str(args.step), "--seed", str(args.seed),
                               "--fg_boost_factor", "2", "--blend_power", "2", "--force_low_memory", "off"])
    pipe, low_memory, device, _ = ies.setup(margs)
    set_seed(args.seed)
    sess = EditStreamSession(pipe, args.src_prompt, args.trg_prompt, args.src_word, args.trg_word, device,
                             sog_mask="gate", reset_at_cut=args.reset_at_cut, edit_mem=args.edit_mem,
                             low_memory=low_memory)
    svae = StreamVAE(pipe.vae)
    codec = LiveCodec()

    orig_of = []           # output frame t -> original frame index (for SAM lookups / metrics)
    sam_at = lambda t: _sam(sam_all, orig_of[t])
    prop = PropMask(sam_at)
    out_frames, cuts, chunk_log = [], [], []
    pending = []           # (x [3, H, W], mask [H, W], arrival time) of the open chunk
    shot_idx, chunk_in_shot = 0, 0

    def run_chunk(items, n_real):
        nonlocal chunk_in_shot
        first = chunk_in_shot == 0
        px = torch.stack([x for x, _, _ in items], 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        z = svae.encode(px)
        tok = token_mask_v0(np.stack([mk for _, mk, _ in items]), first, args.invert, device)
        y = sess.step(z, tok, new_shot=first and shot_idx > 0)
        frames = svae.decode(y)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        out_frames.extend(frames[:n_real].cpu())
        chunk_log.append({"shot": shot_idx, "frames": n_real, "padded": len(items) - n_real,
                          "process_ms": (t1 - t0) * 1e3, "edit_ms": sess.chunk_ms[-1],
                          "latency_ms": (t1 - items[n_real - 1][2]) * 1e3})
        chunk_in_shot += 1

    t_start = time.perf_counter()
    for t, pil, i in frame_source(args, meta, edits):
        arrival = time.perf_counter()
        orig_of.append(i)
        rgb = np.asarray(pil.resize((W, H), Image.BILINEAR)) if pil.size != (W, H) else np.asarray(pil)
        key, cells = codec.push(np.ascontiguousarray(rgb))
        if key and t > 0:  # a cut: close the old shot
            cuts.append(t)
            if pending:
                n = len(pending)
                run_chunk(pending + [pending[-1]] * ((9 if chunk_in_shot == 0 else 12) - n), n)
            pending, shot_idx, chunk_in_shot = [], shot_idx + 1, 0
            svae.new_shot()
        mk = _sam(sam_all, i) if args.mask == "sam" else prop.push(t, t == 0 or (cuts and cuts[-1] == t), cells)
        pending.append((TRANSFORM(pil), mk, arrival))
        if len(pending) == (9 if chunk_in_shot == 0 else 12):
            run_chunk(pending, len(pending))
            pending = []
    if pending:
        n = len(pending)
        run_chunk(pending + [pending[-1]] * ((9 if chunk_in_shot == 0 else 12) - n), n)
    total_s = time.perf_counter() - t_start

    video = torch.stack(out_frames)  # [T, 3, H, W]
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    write_video(args.save_path, rearrange(video, 't c h w -> t h w c') * 255, fps=16)

    T = len(out_frames)
    rep = {"frames": T, "cuts": cuts, "keyframes": prop.keyframes if args.mask == "prop" else "every frame",
           "sam_calls_per_100f": (len(prop.keyframes) / T * 100) if args.mask == "prop" else 100.0,
           "total_s": total_s, "s_per_frame": total_s / T, "chunks": chunk_log,
           "max_latency_ms": max(c["latency_ms"] for c in chunk_log)}
    if args.clip:  # true cuts and background PSNR near / away from them
        sf = meta[args.clip]["shot_frames"]
        rep["true_cuts"] = np.cumsum([n - min(args.k, n - 5) for n in sf])[:-1].tolist()
        src = [np.asarray(p.resize((W, H), Image.BILINEAR)) for _, p, _ in frame_source(args, meta, edits)]
        out = (rearrange(video, 't c h w -> t h w c').numpy() * 255).round().astype(np.uint8)
        near = np.zeros(T, bool)
        for c in rep["true_cuts"]:
            near[max(0, c - 6):c + 6] = True
        ps = []
        for t in range(T):
            m = _sam(sam_all, orig_of[t]) ^ args.invert
            bg = ~(cv2.dilate(m.astype(np.uint8), np.ones((33, 33), np.uint8)) > 0)
            mse = ((out[t].astype(np.float32) - src[t].astype(np.float32)) ** 2).mean(-1)[bg].mean() if bg.mean() > 0.05 else np.nan
            ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-6)) if np.isfinite(mse) else np.nan)
        ps = np.array(ps)
        rep.update(bg_psnr=float(np.nanmean(ps)), bg_psnr_near_cut=float(np.nanmean(ps[near])),
                   bg_psnr_away=float(np.nanmean(ps[~near])), cuts_ok=rep["cuts"] == rep["true_cuts"])
    if args.compare:
        a = (rearrange(video, 't c h w -> t h w c').numpy() * 255).round()
        b = np.stack([np.asarray(f) for f in imageio.get_reader(args.compare)]).astype(np.float32)
        n = min(len(a), len(b))
        d = np.abs(a[:n] - b[:n])
        rep["compare"] = {"frames": [len(a), len(b)], "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
                          "per_frame_mean": d.mean((1, 2, 3)).round(3).tolist()}
    json.dump(rep, open(args.save_path[:-4] + "_stream.json", "w"), indent=1)
    lat = [c["latency_ms"] for c in chunk_log]
    print(f"STREAM frames {T} cuts {cuts} (true {rep.get('true_cuts')}) chunks {len(chunk_log)} "
          f"s/frame {rep['s_per_frame']:.3f} latency ms median {np.median(lat):.0f} max {max(lat):.0f} "
          f"sam/100f {rep['sam_calls_per_100f']:.1f}", flush=True)
    if args.clip:
        print(f"  bg_psnr {rep['bg_psnr']:.2f} near cut {rep['bg_psnr_near_cut']:.2f} away {rep['bg_psnr_away']:.2f} "
              f"cuts_ok {rep['cuts_ok']}")
    if args.compare:
        print(f"  vs {args.compare}: mean |diff| {rep['compare']['mean_abs_diff']:.3f} max {rep['compare']['max_abs_diff']:.0f}")


def _sam(sam_all, i):
    m = sam_all[i].astype(bool)
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return m


if __name__ == "__main__":
    main()
