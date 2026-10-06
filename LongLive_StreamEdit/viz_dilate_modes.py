"""Draw the v1 attention (token) and SOG (latent) masks for three dilation settings, no generation.

The conversion below is a copy of inference_edit_streamedit.py (v1 branch, glitch filter omitted:
it changes nothing on the puppet frames shown). Usage: python viz_dilate_modes.py
"""
import types

import numpy as np
import torch
import torch.nn.functional as _pool
from PIL import Image, ImageDraw

CLIP = "468286765_89_absence_3shot_13.9s"
SHOTS = [117, 45, 33]
mb = np.load(f"/root/autodl-tmp/pool_v3/sam3_masks_v2/{CLIP}__yellow_dress.npz")["masks"].astype(bool)


def convert(args):
    _m = torch.from_numpy(mb).float()
    toks, lats, _s = [], [], 0
    for _n in SHOTS:
        _g = [[0]] + [list(range(4 * k - 3, 4 * k + 1)) for k in range(1, (_n - 1) // 4 + 1)]
        _px = torch.stack([(_m[_s:_s + _n][g].mean(0) >= 0.5).float() for g in _g])
        _lat = _pool.avg_pool2d(_px[:, None], 8)[:, 0] >= 0.5
        _tok = _pool.avg_pool2d(_lat.float()[:, None], 2)[:, 0] >= 0.5

        def _grow(x, r):
            f = x.float()[:, None]
            d = _pool.max_pool2d(f, 2 * r + 1, 1, r)
            if args.oracle_dilate_mode == "nogap":
                c = -_pool.max_pool2d(-d, 2 * r + 1, 1, r)
                d = d * (1 - ((c > 0) & (f == 0)).float())
            return d[:, 0] > 0
        _lat0 = _lat
        if args.oracle_dilate > 0:
            _tok = _grow(_tok, args.oracle_dilate)
            if args.oracle_dilate_mode == "nogap":
                _r2 = 2 * args.oracle_dilate
                _d2 = _pool.max_pool2d(_lat0.float()[:, None], 2 * _r2 + 1, 1, _r2)
                _gap = ((-_pool.max_pool2d(-_d2, 2 * _r2 + 1, 1, _r2)) > 0)[:, 0] & ~_lat0
                _tok = _tok & ~(_pool.avg_pool2d(_gap.float()[:, None], 2)[:, 0] > 0.5)
        if args.oracle_lat_dilate > 0:
            _lat = _grow(_lat, args.oracle_lat_dilate)
        toks.append(_tok); lats.append(_lat)
        _s += _n
    return torch.cat(toks), torch.cat(lats)


modes = [("dilate 1 (old)", dict(oracle_dilate=1, oracle_lat_dilate=1, oracle_dilate_mode="plain")),
         ("no dilate", dict(oracle_dilate=0, oracle_lat_dilate=0, oracle_dilate_mode="plain")),
         ("nogap", dict(oracle_dilate=1, oracle_lat_dilate=1, oracle_dilate_mode="nogap"))]
res = [(n, *convert(types.SimpleNamespace(**a))) for n, a in modes]


def frame_latent(t):
    for s0, n, l0 in [(0, 117, 0), (117, 45, 30), (162, 33, 42)]:
        if s0 <= t < s0 + n:
            j = t - s0
            return l0 + (0 if j == 0 else (j + 3) // 4)


def panel(t, name, tok, lat, crop):
    l = frame_latent(t)
    s = np.stack([mb[t].astype(np.uint8) * 90] * 3, -1)                     # SAM pixel mask, grey
    a = np.kron(tok[l].numpy().astype(np.uint8), np.ones((16, 16), np.uint8))[:480, :832] > 0
    g = np.kron(lat[l].numpy().astype(np.uint8), np.ones((8, 8), np.uint8))[:480, :832] > 0
    s[a, 0] = 255                                                           # red = attention mask
    s[g, 1] = np.maximum(s[g, 1], 200)                                      # green = SOG mask
    y0, y1, x0, x1 = crop
    im = Image.fromarray(s[y0:y1, x0:x1]).resize((300, int(300 * (y1 - y0) / (x1 - x0))), Image.NEAREST)
    ImageDraw.Draw(im).text((3, 2), f"{name} f{t}", fill=(255, 255, 255))
    return np.asarray(im)


rows = []
for t, crop in [(40, (150, 480, 180, 420)), (100, (150, 480, 180, 420)), (166, (200, 460, 230, 470)),
                (170, (200, 460, 230, 470)), (174, (200, 460, 230, 470))]:
    ps = [panel(t, n, tk, lt, crop) for n, tk, lt in res]
    h = min(p.shape[0] for p in ps)
    rows.append(np.concatenate([p[:h] for p in ps], 1))
w = min(r.shape[1] for r in rows)
Image.fromarray(np.concatenate([r[:, :w] for r in rows], 0)).save("/tmp/dilate_modes.jpg", quality=92)
for n, tk, lt in res:
    print(f"{n:15s} token coverage {tk.float().mean():.4f}  latent coverage {lt.float().mean():.4f}")
