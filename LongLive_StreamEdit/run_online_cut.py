"""Online shot handling end to end, plus the cut-offset sweep.

For one source stream (a manifest clip, optionally with every shot shortened by k tail frames so that the
cuts land at every position inside a 12-frame chunk):
  1. cuts      detected online from the live-style x264 re-encode: an I-frame = a new shot (stream_cut.py)
  2. mask      the chosen mask (docs/mask_propagation_2026-10-11.md): SAM3 on keyframes (here: the
               per-frame SAM3 mask of that frame), motion-vector propagation, schedule a16r1.15,
               with keyframes at the DETECTED shot starts
  3. padding   every shot of n frames is padded to the next 12m-3 frames by repeating its last frame (and
               mask), so each shot starts a fresh chunk grid and a fresh VAE latent grid; the padding only
               depends on the cut, i.e. it is causal
  4. edit      the existing per-shot pipeline (--shot_frames = padded lengths, mask v0, SOG gate, reset none)
  5. strip     padded frames are dropped -> <tag>.mp4 has exactly the source frames
  6. metrics   background PSNR vs the source (outside the per-frame SAM3 mask dilated by 16 px) for frames
               within 6 frames of a cut and for the others; written to <tag>_metrics.json

Runs: EDITS x K (default k = 0, 3, 6, 9; a shot keeps at least 5 frames). k = 0 is the aligned clip (no
padding) = the current pipeline.
Outputs /root/autodl-tmp/e0_runs/online_cut/<clip>/<edit>/k<k>.mp4 (+ _metrics.json, _plan.json); resumable.
Run from LongLive_StreamEdit/:  python run_online_cut.py [--k 0,3,6,9] [--edits clip:edit,...] [--prep_only]
--prep_only builds the padded inputs and plans without loading the model (CPU check).
"""
import argparse
import json
import os
import time
import traceback
from types import SimpleNamespace

import av
import cv2
import numpy as np

import mask_prop_eval as mpe
from stream_cut import codec_cuts

POOL = "/root/autodl-tmp/pool_v3"
OUT = "/root/autodl-tmp/e0_runs/online_cut"
EDITS = ["226154551_5_viewpoint_4shot_12.6s:replacement", "468286765_89_absence_3shot_13.9s:pattern",
         "26422501_15_absence_6shot_12.5s:pattern"]
NEAR = 6  # frames on each side of a cut counted as "near the cut"


def read_rgb(video, T):
    out = []
    with av.open(video) as c:
        for fr in c.decode(video=0):
            out.append(cv2.resize(fr.to_ndarray(format="rgb24"), (mpe.W, mpe.H), interpolation=cv2.INTER_AREA))
            if len(out) == T:
                break
    return out


def write_mp4(path, frames, lossless=False):
    c = av.open(path, "w")
    st = c.add_stream("libx264", rate=16)
    st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
    st.options = {"qp": "0"} if lossless else {"crf": "18"}
    for f in frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()


def padded_len(n):
    """smallest 12m - 3 >= n (m >= 1)."""
    return max(9, 12 * -(-(n + 3) // 12) - 3)


def prepare(clip, edit, k, meta, edits, ffmpeg, d, pool=POOL):
    """Build <d>/k<k>_src.mp4 (shortened stream), k<k>_pad.mp4 + k<k>_mask.npz (padded) and the plan."""
    m, e = meta[clip], edits[clip]["edits"][edit]
    sf = m["shot_frames"]
    starts = np.concatenate([[0], np.cumsum(sf)[:-1]]).astype(int)
    ks = [min(k, n - 5) for n in sf]  # drop k tail frames per shot, keep at least 5
    keep = np.concatenate([np.arange(a, a + n - kk) for a, n, kk in zip(starts, sf, ks)])
    T0 = sum(sf)
    rgb0 = read_rgb(m["video"], T0)
    sam0 = mpe.load_masks(f"{pool}/sam3_masks_v2/{clip}__{e['sam3_prompt'].replace(' ', '_')}.npz", T0)
    rgb, sam = [rgb0[i] for i in keep], sam0[keep]
    T = len(rgb)
    src = f"{d}/k{k}_src.mp4"
    write_mp4(src, rgb, lossless=True)

    # 1. online cuts (the stream as a live encoder would see it)
    cuts = codec_cuts(src, ffmpeg, T)
    true_cuts = (np.cumsum([n - kk for n, kk in zip(sf, ks)])[:-1]).tolist()
    shots = np.diff([0] + cuts + [T]).tolist()

    # 2. chosen mask, keyframes at the detected shot starts
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb])
    raw, _ = mpe.load_mv_maps(src, T, ffmpeg)
    pred, key, _ = mpe.run("mv", "a16r1.15", sam, shots, gray, {"raw": raw}, None,
                           SimpleNamespace(search=12, area_ratio=1.15))

    # 3. pad every shot to 12m - 3 frames by repeating its last frame
    idx, is_pad, s = [], [], 0
    for n in shots:
        N = padded_len(n)
        idx += list(range(s, s + n)) + [s + n - 1] * (N - n)
        is_pad += [False] * n + [True] * (N - n)
        s += n
    write_mp4(f"{d}/k{k}_pad.mp4", [rgb[i] for i in idx], lossless=True)
    np.savez_compressed(f"{d}/k{k}_mask.npz", masks=pred[idx])
    plan = {"clip": clip, "edit": edit, "k": k, "frames": T, "true_cuts": true_cuts, "detected_cuts": cuts,
            "shots": shots, "padded_shots": [padded_len(n) for n in shots], "pad_frames": int(sum(is_pad)),
            "keyframes": np.nonzero(key)[0].tolist(), "is_pad": is_pad}
    json.dump(plan, open(f"{d}/k{k}_plan.json", "w"))
    return plan, rgb, sam


def strip_and_score(out_padded, plan, rgb, sam, final):
    """drop padded frames, write the final video, background PSNR near / away from cuts."""
    frames = read_rgb(out_padded, len(plan["is_pad"]))
    kept = [f for f, p in zip(frames, plan["is_pad"]) if not p]
    assert len(kept) == plan["frames"], (len(kept), plan["frames"])
    write_mp4(final, kept)
    cuts = plan["true_cuts"]
    near = np.zeros(plan["frames"], bool)
    for c in cuts:
        near[max(0, c - NEAR):c + NEAR] = True
    ps = []
    for t in range(plan["frames"]):
        bg = ~(cv2.dilate(sam[t].astype(np.uint8), np.ones((33, 33), np.uint8)) > 0)
        if bg.mean() < 0.05:
            ps.append(np.nan)
            continue
        mse = ((kept[t].astype(np.float32) - rgb[t].astype(np.float32)) ** 2).mean(-1)[bg].mean()
        ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-6)))
    ps = np.array(ps)
    f = lambda sel: float(np.nanmean(ps[sel])) if np.isfinite(ps[sel]).any() else None
    return {"bg_psnr": f(np.ones_like(near)), "bg_psnr_near_cut": f(near), "bg_psnr_away": f(~near),
            "cuts_ok": plan["detected_cuts"] == plan["true_cuts"], "pad_frames": plan["pad_frames"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="0,3,6,9")
    ap.add_argument("--edits", default=",".join(EDITS))
    ap.add_argument("--prep_only", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pool", default=POOL)
    args = ap.parse_args()
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    meta = {m["name"]: m for m in json.load(open(f"{args.pool}/manifest_aligned_v2.json"))}
    edits = {c["clip"]: c for c in json.load(open(f"{args.pool}/edits.json"))}
    jobs = [(x.split(":")[0], x.split(":")[1], int(k)) for x in args.edits.split(",") for k in args.k.split(",")]

    if not args.prep_only:
        import torch
        import inference_edit_streamedit as ies
        from utils.misc import set_seed
        parser = ies.build_parser()
        common = ["--fg_boost_factor", "2", "--blend_power", "2", "--step", "15", "--seed", "0",
                  "--force_low_memory", "off"]
        base = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x", "--trg_prompt", "x",
                                  "--src_word", "x", "--trg_word", "x"] + common)
        pipeline, low_memory, device, local_rank = ies.setup(base)

    for i, (clip, edit, k) in enumerate(jobs):
        d = f"{args.out}/{clip}/{edit}"
        os.makedirs(d, exist_ok=True)
        final = f"{d}/k{k}.mp4"
        if os.path.exists(final):
            continue
        t0 = time.time()
        plan, rgb, sam = prepare(clip, edit, k, meta, edits, ffmpeg, d, args.pool)
        print(f"[{i + 1}/{len(jobs)}] {clip[:24]} {edit:11s} k={k} cuts true {plan['true_cuts']} detected "
              f"{plan['detected_cuts']} shots {plan['shots']} -> {plan['padded_shots']} (+{plan['pad_frames']} pad)",
              flush=True)
        if args.prep_only:
            continue
        c, e = edits[clip], edits[clip]["edits"][edit]
        padded_out = f"{d}/k{k}_padded_out.mp4"
        argv = ["--data_path", f"{d}/k{k}_pad.mp4", "--save_path", padded_out, "--src_prompt", c["src_prompt"],
                "--trg_prompt", e["trg_prompt"], "--src_word", e["src_word"], "--trg_word", e["trg_word"],
                "--shot_frames", ",".join(map(str, plan["padded_shots"])), "--reset_at_cut", "none",
                "--oracle_mask", f"{d}/k{k}_mask.npz", "--oracle_version", "v0", "--oracle_dilate", "1",
                "--sog_mask", "gate", "--timing_out", f"{d}/k{k}_timing.json"] + common
        if e.get("sam3_invert"):
            argv.append("--oracle_invert")
        set_seed(0)
        try:
            ies.edit_one(parser.parse_args(argv), pipeline, low_memory, device, local_rank)
            met = strip_and_score(padded_out, plan, rgb, sam, final)
            json.dump(met, open(f"{d}/k{k}_metrics.json", "w"), indent=1)
            print(f"    bg_psnr {met['bg_psnr']:.2f}  near cut {met['bg_psnr_near_cut']:.2f}  away "
                  f"{met['bg_psnr_away']:.2f}  cuts_ok {met['cuts_ok']}  {time.time() - t0:.0f}s", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
        pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()

    if not args.prep_only:  # summary
        print(f"\n{'clip':26s} {'edit':11s} {'k':>2s} {'pad':>4s} {'bg':>6s} {'near':>6s} {'away':>6s} cuts_ok")
        for clip, edit, k in jobs:
            f = f"{args.out}/{clip}/{edit}/k{k}_metrics.json"
            if os.path.exists(f):
                r = json.load(open(f))
                print(f"{clip[:26]:26s} {edit:11s} {k:2d} {r['pad_frames']:4d} {r['bg_psnr']:6.2f} "
                      f"{r['bg_psnr_near_cut']:6.2f} {r['bg_psnr_away']:6.2f} {r['cuts_ok']}")
    print("ONLINE_CUT_DONE", flush=True)


if __name__ == "__main__":
    main()
