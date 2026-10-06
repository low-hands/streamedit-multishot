"""E0 batch runner (one process, models loaded once).

Every active edit x reset_at_cut {none, all, keep_sink}, grounding = SAM3 masks.
Run from LongLive_StreamEdit/:  python run_e0.py [steps]
Resumable: a run whose .mp4 already exists is skipped. One JSON line per finished run is
appended to runs.jsonl. The seed is re-set before every run, so runs are comparable with
each other (not bit-for-bit with separate CLI runs, whose RNG is consumed by model init).
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
OUT = "/root/autodl-tmp/e0_runs/main"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
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

parser = ies.build_parser()
common = ["--fg_boost_factor", "2", "--blend_power", "2", "--step", str(STEPS),
          "--seed", str(SEED), "--force_low_memory", "off"]
base = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x",
                          "--trg_prompt", "x", "--src_word", "x", "--trg_word", "x"] + common)
pipeline, low_memory, device, local_rank = ies.setup(base)
print(f"{len(jobs)} runs, steps={STEPS}, low_memory={low_memory}", flush=True)

os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/runs.jsonl", "a")
for i, (c, meta, ek, e, r) in enumerate(jobs):
    d = f"{OUT}/s{STEPS}/{c['clip']}/{ek}"
    os.makedirs(d, exist_ok=True)
    mp4 = f"{d}/{r}_sam3.mp4"
    if os.path.exists(mp4):
        continue
    npz = f"{POOL}/sam3_masks_v2/{c['clip']}__{e['sam3_prompt'].replace(' ', '_')}.npz"
    argv = ["--data_path", meta["video"], "--save_path", mp4,
            "--src_prompt", c["src_prompt"], "--trg_prompt", e["trg_prompt"],
            "--src_word", e["src_word"], "--trg_word", e["trg_word"],
            "--shot_frames", ",".join(str(n) for n in meta["shot_frames"]),
            "--reset_at_cut", r,
            "--oracle_mask", npz, "--oracle_dilate", "2" if e["type"] == "replacement" else "1",
            "--save_latents", f"{d}/{r}_sam3_latents.pt"] + common
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
    rec = {"i": i, "clip": c["clip"], "edit": ek, "type": e["type"], "grounding": "sam3", "reset": r,
           "steps": STEPS, "seed": SEED, "exit": rc, "seconds": round(time.time() - t0), "video": mp4}
    if err:
        rec["error"] = err[-500:]
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[{i + 1}/{len(jobs)}] {c['clip'][:24]} {ek:11s} {r:9s} exit={rc} {rec['seconds']}s", flush=True)
print("E0_DONE", flush=True)
