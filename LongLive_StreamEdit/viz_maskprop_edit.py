"""Side-by-side view of run_maskprop_edit.py outputs with the masks each run actually used.

2 x 2 panels per frame:
  source + per-frame SAM3 mask (green)    | C        edit + its mask (per-frame SAM3, green)
  P_mv12 edit + its mask (red)            | P_mva24  edit + its mask (red)
On the P panels the per-frame SAM3 mask is drawn too (thin green), so drift shows as red vs green.
Header: mode, frame, shot, KEY on SAM3 keyframes of that P run, pixel IoU of its mask vs per-frame SAM3.
--tokens draws masks at 16 px token level (what attention uses) instead of pixel level.

Writes <OUT>/<clip>/<edit>/cmp_masks.mp4 (the cmp_ prefix keeps eval_e0.py away from it); skips edits
whose three videos are not all there yet. CPU only.
Run from LongLive_StreamEdit/:  python viz_maskprop_edit.py [--tokens] [only_clip_or_edit_substring]
"""
import json
import os
import sys

import av
import cv2
import numpy as np

from mask_prop_eval import H, W, load_masks, pool_ge

POOL = "/root/autodl-tmp/pool_v3"
PRED = "/root/autodl-tmp/e0_runs/maskprop_v2"
OUT = "/root/autodl-tmp/e0_runs/maskprop_edit/s15"
MASKS = {"C": f"{POOL}/sam3_masks_v2", "P_mv12": f"{PRED}/pred_mv_12", "P_mva24": f"{PRED}/pred_mv_a24r1.25"}
TOKENS = "--tokens" in sys.argv
ONLY = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
GREEN, RED, YELLOW = (40, 255, 40), (255, 50, 50), (255, 220, 0)


def frames(path, T):
    out = []
    with av.open(path) as c:
        for fr in c.decode(video=0):
            out.append(cv2.resize(fr.to_ndarray(format="rgb24"), (W, H), interpolation=cv2.INTER_AREA))
            if len(out) == T:
                break
    return out


def as_drawn(m):
    return np.kron(pool_ge(m[None].astype(np.float32), 16)[0], np.ones((16, 16), bool)) if TOKENS else m


def outline(img, m, color, width):
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnt, -1, color, width)


def panel(img, mask, color, ref, header, key):
    img = img.copy()
    if ref is not None:
        outline(img, ref, GREEN, 1)
    outline(img, mask, color, 3)
    img = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (W // 2, 22), (0, 0, 0), -1)
    cv2.putText(img, header, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if key:
        cv2.putText(img, "KEY (SAM3)", (W // 2 - 105, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
    return img


def render(clip, ek, e, m):
    d = f"{OUT}/{clip}/{ek}"
    vids = {k: f"{d}/{k}.mp4" for k in MASKS}
    if not all(os.path.exists(v) for v in vids.values()):
        return False
    T = sum(m["shot_frames"])
    name = f"{clip}__{e['sam3_prompt'].replace(' ', '_')}.npz"
    masks, keys = {}, {}
    for k, root in MASKS.items():
        masks[k] = load_masks(f"{root}/{name}", T)
        z = np.load(f"{root}/{name}")
        keys[k] = set(z["keyframes"].tolist()) if "keyframes" in z else set(range(T))
    if e.get("sam3_invert"):  # background edit: the edited region is the complement
        masks = {k: ~v for k, v in masks.items()}
    src = frames(m["video"], T)
    out = {k: frames(v, T) for k, v in vids.items()}
    T = min([T, len(src)] + [len(v) for v in out.values()])
    starts = np.concatenate([[0], np.cumsum(m["shot_frames"])[:-1]]).astype(int)
    path = f"{d}/cmp_masks{'_tokens' if TOKENS else ''}.mp4"
    c = av.open(path, "w")
    st = c.add_stream("libx264", rate=16)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "20"}
    for t in range(T):
        shot = int(np.searchsorted(starts, t, side="right") - 1)
        ref = as_drawn(masks["C"][t])
        tiles = [panel(src[t], ref, GREEN, None, f"source  f{t} shot{shot}", False),
                 panel(out["C"][t], ref, GREEN, None, f"C (SAM3 every frame)  f{t}", False)]
        for k in ("P_mv12", "P_mva24"):
            p = as_drawn(masks[k][t])
            u = (p | ref).sum()
            iou = f"{(p & ref).sum() / u:.2f}" if u else "-"
            tiles.append(panel(out[k][t], p, RED, ref, f"{k}  f{t}  IoU {iou}", t in keys[k]))
        canvas = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
        for pkt in st.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    print("wrote", path, flush=True)
    return True


if __name__ == "__main__":
    meta = {m["name"]: m for m in json.load(open(f"{POOL}/manifest_aligned_v2.json"))}
    n = 0
    for c in json.load(open(f"{POOL}/edits.json")):
        for ek, e in c["edits"].items():
            if ONLY and ONLY not in c["clip"] and ONLY != ek:
                continue
            n += render(c["clip"], ek, e, meta[c["clip"]])
    print(f"{n} comparison videos")
