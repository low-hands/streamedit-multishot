"""Does keyframe-SAM3 + cheap propagation edit as well as per-frame SAM3?

Same setting C as run_mem_check.py (per-shot VAE, dilate 1, SOG gate, reset none, no memory), with the
pixel->token mask conversion chosen by --v0 (default) or --v1 (see --oracle_version);
only the mask npz given to --oracle_mask changes:
  C        per-frame SAM3 (sam3_masks_v2)
  P_mv12   SAM3 every 12 frames + codec motion-vector propagation   (mask_prop_eval.py --save_pred mv:12)
  P_mva24  SAM3 adaptive (area x1.25, <= 24 frames) + motion vectors (--save_pred mv:a24r1.25)
Every edit in edits.json whose mask exists in all prediction dirs is run.
Outputs e0_runs/maskprop_edit_v0/s15 (v1: e0_runs/maskprop_edit/s15)/<clip>/<edit>/<mode>.mp4
(+ _timing.json); resumable. Then:  python eval_e0.py /root/autodl-tmp/e0_runs/maskprop_edit_v0/s15
Run from LongLive_StreamEdit/:  python run_maskprop_edit.py [--v1] [only_clip_or_edit_substring]
"""
import json
import os
import sys
import time
import traceback

import torch

import inference_edit_streamedit as ies
from utils.misc import set_seed

POOL = "/root/autodl-tmp/pool_v3"
PRED = "/root/autodl-tmp/e0_runs/maskprop_v2"
VERSION = "v1" if "--v1" in sys.argv else "v0"
OUT = "/root/autodl-tmp/e0_runs/maskprop_edit/s15" if VERSION == "v1" else "/root/autodl-tmp/e0_runs/maskprop_edit_v0/s15"
ONLY = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
MODES = {"C": f"{POOL}/sam3_masks_v2", "P_mv12": f"{PRED}/pred_mv_12", "P_mva24": f"{PRED}/pred_mv_a24r1.25"}

edits = json.load(open(f"{POOL}/edits.json"))
meta = {m["name"]: m for m in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}

jobs = []
for c in edits:
    for ek, e in c["edits"].items():
        name = f"{c['clip']}__{e['sam3_prompt'].replace(' ', '_')}.npz"
        if ONLY and ONLY not in c["clip"] and ONLY != ek:
            continue
        if all(os.path.exists(f"{d}/{name}") for d in MODES.values()):
            jobs += [(c, ek, e, mode, f"{d}/{name}") for mode, d in MODES.items()]
print(f"{len(jobs)} runs, mask {VERSION}, -> {OUT}", flush=True)

parser = ies.build_parser()
common = ["--fg_boost_factor", "2", "--blend_power", "2", "--step", "15", "--seed", "0", "--force_low_memory", "off"]
base = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x",
                          "--trg_prompt", "x", "--src_word", "x", "--trg_word", "x"] + common)
pipeline, low_memory, device, local_rank = ies.setup(base)
os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/../runs.jsonl", "a")
for i, (c, ek, e, mode, npz) in enumerate(jobs):
    m = meta[c["clip"]]
    d = f"{OUT}/{c['clip']}/{ek}"
    os.makedirs(d, exist_ok=True)
    mp4 = f"{d}/{mode}.mp4"
    if os.path.exists(mp4):
        continue
    argv = ["--data_path", m["video"], "--save_path", mp4, "--src_prompt", c["src_prompt"], "--trg_prompt", e["trg_prompt"],
            "--src_word", e["src_word"], "--trg_word", e["trg_word"],
            "--shot_frames", ",".join(str(n) for n in m["shot_frames"]), "--reset_at_cut", "none",
            "--oracle_mask", npz, "--oracle_version", VERSION, "--oracle_dilate", "1", "--sog_mask", "gate",
            "--timing_out", f"{d}/{mode}_timing.json"] + common
    if e.get("sam3_invert"):
        argv.append("--oracle_invert")
    set_seed(0)
    t0, rc, err = time.time(), 0, None
    try:
        ies.edit_one(parser.parse_args(argv), pipeline, low_memory, device, local_rank)
    except Exception:
        rc, err = 1, traceback.format_exc()
        print(err, flush=True)
    pipeline.vae.model.clear_cache()
    torch.cuda.empty_cache()
    rec = {"clip": c["clip"], "edit": ek, "mode": mode, "version": VERSION, "mask": npz, "exit": rc, "seconds": round(time.time() - t0), "video": mp4}
    if err:
        rec["error"] = err[-500:]
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[{i + 1}/{len(jobs)}] {c['clip'][:24]} {ek:11s} {mode:8s} exit={rc} {rec['seconds']}s", flush=True)
print("MASKPROP_EDIT_DONE", flush=True)
