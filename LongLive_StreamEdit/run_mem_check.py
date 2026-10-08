"""Edit memory v1 check on more edits: C (no memory) vs M (edit memory), same prompts (edits.json as is).

C = per-shot VAE, SAM3 v1 mask (dilate 1), SOG gate, reset none.   M = C + --edit_mem 1024 --mem_delta 3.
Run from LongLive_StreamEdit/:  python run_mem_check.py
Outputs e0_runs/memcheck/s15/<clip>/<edit>/<C|M>.mp4 (+ _timing.json, _viz.pt). Resumable.
"""
import json
import os
import time
import traceback

import torch

import inference_edit_streamedit as ies
from utils.misc import set_seed

POOL = "/root/autodl-tmp/pool_v3"
OUT = "/root/autodl-tmp/e0_runs/memcheck/s15"
EDITS = [("26422501_15_absence_6shot_12.5s", "pattern"),
         ("28329869_34_absence_3shot_9.5s", "material"),
         ("247382794_50_absence_4shot_12.5s", "pattern"),
         ("41279729_43_simple_5shot_14.5s", "material")]
edits = {c["clip"]: c for c in json.load(open(f"{POOL}/edits.json"))}
meta = {m["name"]: m for m in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}

parser = ies.build_parser()
common = ["--fg_boost_factor", "2", "--blend_power", "2", "--step", "15", "--seed", "0", "--force_low_memory", "off"]
base = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x",
                          "--trg_prompt", "x", "--src_word", "x", "--trg_word", "x"] + common)
pipeline, low_memory, device, local_rank = ies.setup(base)
os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/../runs.jsonl", "a")
jobs = [(c, ek, mode) for c, ek in EDITS for mode in ("C", "M")]
for i, (clip, ek, mode) in enumerate(jobs):
    c, m, e = edits[clip], meta[clip], edits[clip]["edits"][ek]
    d = f"{OUT}/{clip}/{ek}"
    os.makedirs(d, exist_ok=True)
    mp4 = f"{d}/{mode}.mp4"
    if os.path.exists(mp4):
        continue
    npz = f"{POOL}/sam3_masks_v2/{clip}__{e['sam3_prompt'].replace(' ', '_')}.npz"
    argv = ["--data_path", m["video"], "--save_path", mp4, "--src_prompt", c["src_prompt"], "--trg_prompt", e["trg_prompt"],
            "--src_word", e["src_word"], "--trg_word", e["trg_word"],
            "--shot_frames", ",".join(str(n) for n in m["shot_frames"]), "--reset_at_cut", "none",
            "--oracle_mask", npz, "--oracle_version", "v1", "--oracle_dilate", "1", "--sog_mask", "gate",
            "--timing_out", f"{d}/{mode}_timing.json", "--viz_out", f"{d}/{mode}_viz.pt",
            "--viz_x0_chunks", ",".join(str(k) for k in m["cut_chunks"]), "--viz_x0_steps", "0,6,14"] + common
    if mode == "M":
        argv += ["--edit_mem", "1024", "--mem_delta", "3"]
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
    rec = {"clip": clip, "edit": ek, "mode": mode, "exit": rc, "seconds": round(time.time() - t0), "video": mp4}
    if err:
        rec["error"] = err[-500:]
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[{i + 1}/{len(jobs)}] {clip[:24]} {ek:9s} {mode} exit={rc} {rec['seconds']}s", flush=True)
print("MEMCHECK_DONE", flush=True)
