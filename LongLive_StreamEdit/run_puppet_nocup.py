"""Puppet clip, prompts without " holding a green cup", one process for several runs.

Usage (from LongLive_StreamEdit/): python run_puppet_nocup.py <edit_key>:<mode> [<edit_key>:<mode> ...]
  mode A = stock CLI (joint VAE, cross-attn grounding, stock SOG)
  mode C = our setting (per-shot VAE, SAM3 v1 mask, SOG gate, reset none)
  mode M = C + edit memory (1024 tokens/layer, first version)        reset none
  mode MR = C + edit memory, full reset at every cut (forget the scene, keep the edit)
  mode C0 = C without any mask dilation (attention token mask and SOG latent mask)
Outputs e0_runs/baseline/s15/<clip>/<edit_key>/<A_stock|C_ours_gate_v1>_nocup.mp4 (+ timing, viz).
"""
import json
import os
import sys

import inference_edit_streamedit as ies
from utils.misc import set_seed

POOL = "/root/autodl-tmp/pool_v3"
CLIP = "468286765_89_absence_3shot_13.9s"
DROP = " holding a green cup"
c = [c for c in json.load(open(f"{POOL}/edits.json")) if c["clip"] == CLIP][0]
meta = {m["name"]: m for m in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}[CLIP]

parser = ies.build_parser()
common = ["--fg_boost_factor", "2", "--blend_power", "2", "--step", "15", "--seed", "0", "--force_low_memory", "off"]
base = parser.parse_args(["--data_path", "x", "--save_path", "x", "--src_prompt", "x",
                          "--trg_prompt", "x", "--src_word", "x", "--trg_word", "x"] + common)
pipeline, low_memory, device, local_rank = ies.setup(base)

for job in sys.argv[1:]:
    ek, mode = job.split(":")
    e = c["edits"][ek]
    sp, tp = c["src_prompt"], e["trg_prompt"]
    assert DROP in sp and DROP in tp
    sp, tp = sp.replace(DROP, ""), tp.replace(DROP, "")
    tag = {"A": "A_stock_nocup", "C": "C_ours_gate_v1_nocup", "M": "M_mem_nocup", "MR": "M_mem_resetall_nocup",
           "C0": "C_ours_gate_v1_nodil_nocup"}[mode]
    d = f"/root/autodl-tmp/e0_runs/baseline/s15/{CLIP}/{ek}"
    os.makedirs(d, exist_ok=True)
    mp4 = f"{d}/{tag}.mp4"
    if os.path.exists(mp4):
        continue
    print(f"== {ek} {tag}\nSRC: {sp}\nTRG: {tp}", flush=True)
    argv = ["--data_path", meta["video"], "--save_path", mp4, "--src_prompt", sp, "--trg_prompt", tp,
            "--src_word", e["src_word"], "--trg_word", e["trg_word"],
            "--timing_out", f"{d}/{tag}_timing.json", "--viz_out", f"{d}/{tag}_viz.pt",
            "--viz_x0_chunks", "5,9,10,14", "--viz_x0_steps", "0,6,7,14"] + common
    if mode != "A":
        npz = f"{POOL}/sam3_masks_v2/{CLIP}__{e['sam3_prompt'].replace(' ', '_')}.npz"
        argv += ["--shot_frames", ",".join(str(n) for n in meta["shot_frames"]),
                 "--reset_at_cut", "all" if mode == "MR" else "none",
                 "--oracle_mask", npz, "--oracle_version", "v1", "--sog_mask", "gate"]
        # C0: no dilation at all (attention token mask and SOG latent mask straight from SAM3)
        argv += ["--oracle_dilate", "0", "--oracle_lat_dilate", "0"] if mode == "C0" else ["--oracle_dilate", "1"]
        if mode in ("M", "MR"):
            argv += ["--edit_mem", "1024", "--mem_delta", "3"]
    set_seed(0)
    ies.edit_one(parser.parse_args(argv), pipeline, low_memory, device, local_rank)
    pipeline.vae.model.clear_cache()
    print(f"done {ek} {tag}", flush=True)
print("PUPPET_NOCUP_DONE", flush=True)
