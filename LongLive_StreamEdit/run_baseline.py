"""Baseline batch: stock StreamEdit vs our setting (velocity / gate SOG), descriptive prompts.

A  stock CLI (no E0 flags: joint VAE, cross-attn grounding, stock SOG)
B  per-shot VAE + SAM3 attention + SOG velocity, reset none
C  per-shot VAE + SAM3 attention + SOG gate,     reset none
plus the puppet colour edit in mode A without "holding a green cup" in the prompts.
Every run writes <tag>.mp4, <tag>_timing.json and <tag>_viz.pt.
Run from LongLive_StreamEdit/:  python run_baseline.py [steps] [only_tag_substring]
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
OUT = "/root/autodl-tmp/e0_runs/baseline"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
SEED = 0
CONFIGS = [  # (clip, edit key, x0 chunks to decode)
    ("26422501_15_absence_6shot_12.5s", "pattern", "3,7,9,12,14"),
    ("226154551_5_viewpoint_4shot_12.6s", "replacement", "4,9,12,14"),
    ("468286765_89_absence_3shot_13.9s", "control", "5,9,10,14"),
]
MODES = {
    "A_stock": [],
    "B_ours_velocity_v1": ["shot", "--sog_mask", "velocity", "--oracle_version", "v1"],
    "C_ours_gate": ["shot", "--sog_mask", "gate"],
    "C_ours_gate_v1": ["shot", "--sog_mask", "gate", "--oracle_version", "v1"],
}

edits = {c["clip"]: c for c in json.load(open(f"{POOL}/edits.json"))}
aligned = {m["name"]: m for m in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}

jobs = []
for clip, ek, chunks in CONFIGS:
    for tag, extra in MODES.items():
        jobs.append((clip, ek, chunks, tag, extra, None))
jobs.append(("468286765_89_absence_3shot_13.9s", "control", "5,9,10,14", "A_stock_nocup", [],
             " holding a green cup"))
if ONLY:  # comma-separated substrings of clip/edit/mode
    jobs = [j for j in jobs if any(o in f"{j[0]}/{j[1]}/{j[3]}" for o in ONLY.split(","))]

parser = ies.build_parser()
common = ["--fg_boost_factor", "2", "--blend_power", "2", "--step", str(STEPS),
          "--seed", str(SEED), "--force_low_memory", "off"]
base = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x",
                          "--trg_prompt", "x", "--src_word", "x", "--trg_word", "x"] + common)
pipeline, low_memory, device, local_rank = ies.setup(base)
print(f"{len(jobs)} runs, steps={STEPS}, low_memory={low_memory}", flush=True)

os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/runs.jsonl", "a")
for i, (clip, ek, chunks, tag, extra, drop) in enumerate(jobs):
    c, meta = edits[clip], aligned[clip]
    e = c["edits"][ek]
    d = f"{OUT}/s{STEPS}/{clip}/{ek}"
    os.makedirs(d, exist_ok=True)
    mp4 = f"{d}/{tag}.mp4"
    if os.path.exists(mp4):
        continue
    sp, tp = c["src_prompt"], e["trg_prompt"]
    if drop:
        assert drop in sp and drop in tp
        sp, tp = sp.replace(drop, ""), tp.replace(drop, "")
    argv = ["--data_path", meta["video"], "--save_path", mp4,
            "--src_prompt", sp, "--trg_prompt", tp, "--src_word", e["src_word"], "--trg_word", e["trg_word"],
            "--timing_out", f"{d}/{tag}_timing.json", "--viz_out", f"{d}/{tag}_viz.pt",
            "--viz_x0_chunks", chunks, "--viz_x0_steps", "0,6,7,14"] + common
    if extra and extra[0] == "shot":
        npz = f"{POOL}/sam3_masks_v2/{clip}__{e['sam3_prompt'].replace(' ', '_')}.npz"
        argv += ["--shot_frames", ",".join(str(n) for n in meta["shot_frames"]), "--reset_at_cut", "none",
                 "--oracle_mask", npz,
                 # v1 is edit-type agnostic (1 token everywhere); v0 keeps the old per-type dilation
                 "--oracle_dilate", "1" if "v1" in tag else ("2" if e["type"] == "replacement" else "1")] + extra[1:]
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
    rec = {"i": i, "clip": clip, "edit": ek, "mode": tag, "steps": STEPS, "seed": SEED,
           "exit": rc, "wall_s": round(time.time() - t0), "video": mp4}
    if err:
        rec["error"] = err[-800:]
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[{i + 1}/{len(jobs)}] {clip[:24]} {ek:11s} {tag:16s} exit={rc} {rec['wall_s']}s", flush=True)
print("BASELINE_DONE", flush=True)
