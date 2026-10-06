"""SOG pilot: which SOG foreground to use with SAM3 grounding (velocity / gate / replace).

3 edits x 3 SOG variants at reset_at_cut=none, plus the puppet colour edit at reset=all.
Run from LongLive_StreamEdit/:  python run_pilot_sog.py [steps]. Resumable.
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
OUT = "/root/autodl-tmp/e0_runs/pilot_sog"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEED = 0
SOGS = ["velocity", "gate", "replace"]
CONFIGS = [  # (clip, edit key, reset)
    ("468286765_89_absence_3shot_13.9s", "control", "none"),
    ("468286765_89_absence_3shot_13.9s", "control", "all"),
    ("226154551_5_viewpoint_4shot_12.6s", "replacement", "none"),
    ("26422501_15_absence_6shot_12.5s", "pattern", "none"),
]

edits = {c["clip"]: c for c in json.load(open(f"{POOL}/edits.json"))}
aligned = {m["name"]: m for m in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}
jobs = [(clip, ek, r, sog) for clip, ek, r in CONFIGS for sog in SOGS]

parser = ies.build_parser()
common = ["--fg_boost_factor", "2", "--blend_power", "2", "--step", str(STEPS),
          "--seed", str(SEED), "--force_low_memory", "off"]
base = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x",
                          "--trg_prompt", "x", "--src_word", "x", "--trg_word", "x"] + common)
pipeline, low_memory, device, local_rank = ies.setup(base)
print(f"{len(jobs)} runs, steps={STEPS}, low_memory={low_memory}", flush=True)

os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/runs.jsonl", "a")
for i, (clip, ek, r, sog) in enumerate(jobs):
    c, meta = edits[clip], aligned[clip]
    e = c["edits"][ek]
    d = f"{OUT}/s{STEPS}/{clip}/{ek}"
    os.makedirs(d, exist_ok=True)
    mp4 = f"{d}/{r}_sam3_sog-{sog}.mp4"
    if os.path.exists(mp4):
        continue
    npz = f"{POOL}/sam3_masks_v2/{clip}__{e['sam3_prompt'].replace(' ', '_')}.npz"
    argv = ["--data_path", meta["video"], "--save_path", mp4,
            "--src_prompt", c["src_prompt"], "--trg_prompt", e["trg_prompt"],
            "--src_word", e["src_word"], "--trg_word", e["trg_word"],
            "--shot_frames", ",".join(str(n) for n in meta["shot_frames"]),
            "--reset_at_cut", r, "--sog_mask", sog,
            "--oracle_mask", npz, "--oracle_dilate", "2" if e["type"] == "replacement" else "1"] + common
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
    rec = {"i": i, "clip": clip, "edit": ek, "reset": r, "sog": sog, "steps": STEPS, "seed": SEED,
           "exit": rc, "seconds": round(time.time() - t0), "video": mp4}
    if err:
        rec["error"] = err[-500:]
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[{i + 1}/{len(jobs)}] {clip[:24]} {ek:11s} {r:5s} sog={sog:8s} exit={rc} {rec['seconds']}s", flush=True)
print("PILOT_DONE", flush=True)
