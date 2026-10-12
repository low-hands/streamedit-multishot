"""Where do stream_edit.py and the batch pipeline part ways? Stage-by-stage check on one shot.

On the first shot of a clip (per-frame SAM3 mask, v0, SOG gate, reset none, seed 0):
  1 frames   diffusers load_video (batch) vs imageio.get_reader (stream): pixel difference
  2 encode   whole-shot vae.encode_to_latent vs StreamVAE chunk by chunk (9, 12, 12, ...)
  3 masks    v0 token masks of the whole shot (inference_edit_streamedit.py) vs token_mask_v0 per chunk
  4 edit     same input latents and masks: pipeline.inference (batch loop) vs EditStreamSession.step,
             both from set_seed(0); also step() on the StreamVAE latents
  5 decode   whole-shot decode_to_pixel vs StreamVAE decode chunk by chunk, on the same latents
Run from LongLive_StreamEdit/:  python stream_check.py [--clip ...] [--edit ...]
"""
import argparse
import json

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import load_video
from PIL import Image

import inference_edit_streamedit as ies
from pipeline.edit_stream import EditStreamSession, StreamVAE
from stream_edit import TRANSFORM, token_mask_v0, _sam
from utils.misc import set_seed

POOL = "/root/autodl-tmp/pool_v3"


def stat(name, a, b):
    d = (a.float() - b.float()).abs()
    print(f"  {name:44s} max {d.max().item():.4g}  mean {d.mean().item():.4g}  "
          f"(scale {a.float().abs().mean().item():.3g})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="226154551_5_viewpoint_4shot_12.6s")
    ap.add_argument("--edit", default="replacement")
    args = ap.parse_args()
    c = {x["clip"]: x for x in json.load(open(f"{POOL}/edits.json"))}[args.clip]
    e = c["edits"][args.edit]
    m = {x["name"]: x for x in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}[args.clip]
    n = m["shot_frames"][0]
    sam = np.load(f"{POOL}/sam3_masks_v2/{args.clip}__{e['sam3_prompt'].replace(' ', '_')}.npz")["masks"]
    margs = ies.build_parser().parse_args(
        ["--data_path", "x", "--save_path", "x", "--src_prompt", "x", "--trg_prompt", "x", "--src_word", "x",
         "--trg_word", "x", "--step", "15", "--seed", "0", "--fg_boost_factor", "2", "--blend_power", "2",
         "--force_low_memory", "off"])
    pipe, low_memory, device, _ = ies.setup(margs)
    print(f"clip {args.clip} edit {args.edit}: first shot, {n} frames", flush=True)

    # 1 frames
    fb = load_video(m["video"])[:n]
    fs = [Image.fromarray(f) for _, f in zip(range(n), imageio.get_reader(m["video"]))]
    print("1 frames", flush=True)
    stat("load_video vs imageio (uint8)", torch.from_numpy(np.stack([np.asarray(x) for x in fb])),
         torch.from_numpy(np.stack([np.asarray(x) for x in fs])))
    px = torch.stack([TRANSFORM(x) for x in fb], 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    # 2 encode
    print("2 encode", flush=True)
    pipe.vae.model.clear_cache()
    zA = pipe.vae.encode_to_latent(px).to(device=device, dtype=torch.bfloat16)
    pipe.vae.model.clear_cache()
    sv = StreamVAE(pipe.vae)
    sv.new_shot()
    edges = [0, 9] + list(range(21, n + 1, 12))
    zB = torch.cat([sv.encode(px[:, :, a:b]) for a, b in zip(edges[:-1], edges[1:])], 1)
    stat("whole-shot vs chunked encode", zA, zB)

    # 3 masks (batch v0 code from inference_edit_streamedit.py)
    print("3 masks", flush=True)
    mm = torch.from_numpy(sam[:n]).float()
    g = [[0]] + [list(range(4 * k - 3, 4 * k + 1)) for k in range(1, (n - 1) // 4 + 1)]
    t = F.avg_pool2d(torch.stack([mm[x].amax(0) for x in g])[:, None], 16)[:, 0] > 0.1
    t = F.max_pool2d(t.float()[:, None], 3, 1, 1)[:, 0] > 0
    tokA = t.reshape(len(g), -1).to(device)
    tokB = torch.cat([token_mask_v0(np.stack([_sam(sam, i) for i in range(a, b)]), a == 0, False, device)
                      for a, b in zip(edges[:-1], edges[1:])])
    print(f"  token masks equal: {bool((tokA == tokB).all())}  ({int((tokA != tokB).sum())} differ)", flush=True)

    # 4 edit loop, same inputs
    print("4 edit", flush=True)
    kw = dict(src_prompts=c["src_prompt"], trg_prompts=e["trg_prompt"], src_trigger_words=e["src_word"],
              trg_trigger_words=e["trg_word"])
    set_seed(0)
    _, yA = pipe.inference(src_video=zA, return_latents=True, wo_video_decode=True, low_memory=low_memory,
                           fg_boost_factor=2.0, blend_power=2.0, oracle_token_masks=tokA, sog_mask="gate",
                           oracle_attn="replace", **kw)

    def run_session(z, tok):
        set_seed(0)
        s = EditStreamSession(pipe, c["src_prompt"], e["trg_prompt"], e["src_word"], e["trg_word"], device,
                              sog_mask="gate", low_memory=low_memory)
        outs, i = [], 0
        while i < z.shape[1]:
            outs.append(s.step(z[:, i:i + 3], tok[i:i + 3]))
            i += 3
        return torch.cat(outs, 1)

    yB = run_session(zA, tokA)
    stat("inference() vs step(), same latents", yA, yB)
    set_seed(0)
    _, yA2 = pipe.inference(src_video=zA, return_latents=True, wo_video_decode=True, low_memory=low_memory,
                            fg_boost_factor=2.0, blend_power=2.0, oracle_token_masks=tokA, sog_mask="gate",
                            oracle_attn="replace", **kw)
    stat("inference() vs inference() (noise floor)", yA, yA2)
    yC = run_session(zB, tokB)
    stat("step() on chunked-encode latents vs batch", yA, yC)
    for j in range(0, yA.shape[1], 3):
        d = (yA[:, j:j + 3].float() - yB[:, j:j + 3].float()).abs().mean().item()
        print(f"    chunk {j // 3}: inference vs step mean {d:.4g}", flush=True)

    # 5 decode
    print("5 decode", flush=True)
    pipe.vae.model.clear_cache()
    vA = (pipe.vae.decode_to_pixel(yA, use_cache=False)[0] * 0.5 + 0.5).clamp(0, 1)
    sv.new_shot()
    vB = torch.cat([sv.decode(yA[:, i:i + 3]) for i in range(0, yA.shape[1], 3)])
    stat("whole-shot vs chunked decode (x255)", vA * 255, vB * 255)


if __name__ == "__main__":
    main()
