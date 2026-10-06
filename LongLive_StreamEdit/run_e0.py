"""E0 batch runner (one process, models loaded once): what happens at a cut?

Every active edit x reset_at_cut {none, all, keep_sink}. Our current setting in all runs:
per-shot VAE, SAM3 grounding with the v1 mask conversion (1 token attention dilation, no
edit-type-dependent settings), SOG mode given on the command line (gate | replace | velocity).
Run from LongLive_StreamEdit/:
    python run_e0.py <steps> <sog_mode> [only]      only = comma-separated substrings of clip/edit/reset
Outputs e0_runs/main_<sog>/s<steps>/<clip>/<edit>/<reset>.mp4 (+ _timing.json, _viz.pt).
x0 previews are decoded only for the first chunk after every cut (where ghosting shows up).
Resumable: a run whose .mp4 already exists is skipped; one JSON line per run in runs.jsonl.
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
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SOG = sys.argv[2] if len(sys.argv) > 2 else "gate"
ONLY = sys.argv[3] if len(sys.argv) > 3 else None
OUT = f"/root/autodl-tmp/e0_runs/main_{SOG}"
SEED = 0
RESETS = ["none", "all", "keep_sink"]

edits = json.load(open(f"{POOL}/edits.json"))
aligned = {m["name"]: m for m in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}

jobs = []
for c in edits:
    if "deferred" in c:
        continue
    meta = aligned[c["clip"]]
    for ek, e in c["edits"].items():
        for r in RESETS:
            jobs.append((c, meta, ek, e, r))
if ONLY:
    jobs = [j for j in jobs if any(o in f"{j[0]['clip']}/{j[2]}/{j[4]}" for o in ONLY.split(","))]

parser = ies.build_parser()
common = ["--fg_boost_factor", "2", "--blend_power", "2", "--step", str(STEPS),
          "--seed", str(SEED), "--force_low_memory", "off"]
base = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x",
                          "--trg_prompt", "x", "--src_word", "x", "--trg_word", "x"] + common)
pipeline, low_memory, device, local_rank = ies.setup(base)
print(f"{len(jobs)} runs, steps={STEPS}, sog={SOG}, low_memory={low_memory}", flush=True)

os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/runs.jsonl", "a")
for i, (c, meta, ek, e, r) in enumerate(jobs):
    d = f"{OUT}/s{STEPS}/{c['clip']}/{ek}"
    os.makedirs(d, exist_ok=True)
    mp4 = f"{d}/{r}.mp4"
    if os.path.exists(mp4):
        continue
    npz = f"{POOL}/sam3_masks_v2/{c['clip']}__{e['sam3_prompt'].replace(' ', '_')}.npz"
    argv = ["--data_path", meta["video"], "--save_path", mp4,
            "--src_prompt", c["src_prompt"], "--trg_prompt", e["trg_prompt"],
            "--src_word", e["src_word"], "--trg_word", e["trg_word"],
            "--shot_frames", ",".join(str(n) for n in meta["shot_frames"]),
            "--reset_at_cut", r,
            "--oracle_mask", npz, "--oracle_version", "v1", "--oracle_dilate", "1", "--sog_mask", SOG,
            "--timing_out", f"{d}/{r}_timing.json", "--viz_out", f"{d}/{r}_viz.pt",
            "--viz_x0_chunks", ",".join(str(k) for k in meta["cut_chunks"]), "--viz_x0_steps", "0,6,14"] + common
    if e.get("sam3_invert"):
        argv.append("--oracle_invert")
    args = parser.parse_args(argv)
    set_seed(SEED)
    t0 = time.time()
    rc, err = 0, None
    try:
        ies.edit_one(args, pipeline, low_memory, device, local_rank)
    except Exception:
        rc, err = 1, traceback.format_exc()
        print(err, flush=True)
    pipeline.vae.model.clear_cache()
    torch.cuda.empty_cache()
    rec = {"i": i, "clip": c["clip"], "edit": ek, "type": e["type"], "sog": SOG, "reset": r,
           "steps": STEPS, "seed": SEED, "exit": rc, "seconds": round(time.time() - t0), "video": mp4}
    if err:
        rec["error"] = err[-500:]
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[{i + 1}/{len(jobs)}] {c['clip'][:24]} {ek:11s} {r:9s} exit={rc} {rec['seconds']}s", flush=True)
print("E0_DONE", flush=True)
