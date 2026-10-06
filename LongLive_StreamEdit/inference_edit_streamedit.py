import argparse
import torch
import os
from pathlib import Path 

import json
import time
from collections import OrderedDict
from omegaconf import OmegaConf
import peft
import numpy as np
from PIL import Image
from einops import rearrange
import torch.distributed as dist
from torchvision import transforms
from torchvision.io import write_video

from pipeline import (
    EditCausalInferencePipeline
)
from utils.misc import set_seed
from utils.lora_utils import configure_lora_for_model
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

from diffusers.utils import load_video


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)

def find_closest_num_frame(x, a=4, b=3):
    max_m = (x + a - 1) // (a * b)
    while max_m > 0:
        y = a * b * max_m - a + 1
        if y <= x:
            return y
        max_m -= 1

def load_pipe(args):
    config = OmegaConf.load(args.config_path)
    config['model_kwargs']['timestep_shift'] = args.flow_shift
    config['denoising_step_list'] = np.arange(1000, 0, -1000 / args.step).astype(int).tolist()
    config['noise_alpha_fg'] = args.noise_alpha_fg
    config['dump_masks_dir'] = args.dump_masks_dir
    # infinity relative rope
    config['model_kwargs']['use_infinite_attention'] = getattr(args, 'use_infinite_attention', False)

    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        os.environ["NCCL_CROSS_NIC"] = "1"
        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
        os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.constants.default_pg_timeout,
            )
        set_seed(config.seed + local_rank)
        config.distributed = True  # Mark as distributed for pipeline
        if rank == 0:
            print(f"[Rank {rank}] Initialized distributed processing on device {device}")
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda")
        set_seed(config.seed)
        config.distributed = False  # Mark as non-distributed
        print(f"Single GPU mode on device {device}")

    print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
    low_memory = get_cuda_free_memory_gb(device) < 40
    if getattr(args, "force_low_memory", None) is not None:
        low_memory = args.force_low_memory == "on"
    print(f"low_memory={low_memory}")

    torch.set_grad_enabled(False)

    # Initialize pipeline
    # Note: checkpoint loading is now handled inside the pipeline __init__ method
    pipeline = EditCausalInferencePipeline(config, device=device)

    # Load generator checkpoint
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
        if config.use_ema:
            def _clean_key(name: str) -> str:
                """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
                name = name.replace("_fsdp_wrapped_module.", "")
                return name

            cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
            missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
            if local_rank == 0:
                if len(missing) > 0:
                    print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
                if len(unexpected) > 0:
                    print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
        else:
            pipeline.generator.load_state_dict(raw_gen_state_dict)

    # --------------------------- LoRA support (optional) ---------------------------

    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None) and configure_lora_for_model is not None:
        if local_rank == 0:
            print(f"LoRA enabled with config: {config.adapter}")
            print("Applying LoRA to generator (inference)...")
        
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=(local_rank == 0),
        )

        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            if local_rank == 0:
                print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
            if local_rank == 0:
                print("LoRA weights loaded for generator")
        else:
            if local_rank == 0:
                print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

        pipeline.is_lora_enabled = True

    return pipeline, low_memory, device, local_rank


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--src_prompt", type=str, required=True)
    parser.add_argument("--trg_prompt", type=str, required=True)
    parser.add_argument("--src_word", type=str, required=True)
    parser.add_argument("--trg_word", type=str, required=True)
    
    # first frame condition, triple_first_frame=True for LongLive
    parser.add_argument("--first_frame_edit", type=str, default=None)
    parser.add_argument("--triple_first_frame", action="store_true", default=True)

    # hyper-parameters
    parser.add_argument("--fg_boost_factor", type=float, default=2.0, help='CrossAttn Boosting')
    parser.add_argument("--blend_power", type=float, default=2.0, help='rho')
    parser.add_argument("--dump_masks_dir", type=str, default=None,
                        help='write per-chunk cross-attention masks here (PNG)')
    parser.add_argument("--noise_alpha_fg", type=float, default=None,
                        help='temporal-noise correlation inside editing regions; '
                             'None (default) keeps the original uniform correlation, '
                             '0 fully decorrelates the foreground across chunks')

    # model settings
    parser.add_argument("--step", type=int, default=15, help='1~1000')
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--use_infinite_attention", action="store_true", default=False)

    parser.add_argument("--config_path", type=str, default='configs/longlive_inference.yaml')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--shot_frames", type=str, default=None,
                        help="E0: comma-separated frames per shot (each 12k-3). Encodes/decodes each shot "
                             "separately and derives the cut chunks. Unset = stock behaviour.")
    parser.add_argument("--reset_at_cut", choices=["none", "all", "keep_sink"], default="none")
    parser.add_argument("--oracle_mask", type=str, default=None,
                        help="E0: npz with `masks` bool [T,H,W] (source video) replacing cross-attn grounding")
    parser.add_argument("--oracle_dilate", type=int, default=1, help="token dilation radius for --oracle_mask")
    parser.add_argument("--oracle_version", choices=["v0", "v1"], default="v0",
                        help="pixel->latent/token conversion of --oracle_mask (see patch_mask_v1.py)")
    parser.add_argument("--oracle_invert", action="store_true", default=False,
                        help="use the complement of the (dilated) oracle mask, e.g. background edits")
    parser.add_argument("--oracle_attn", choices=["replace", "union_trg"], default="replace",
                        help="attention-side mask with --oracle_mask: SAM3 only, or SAM3 OR target cross-attn")
    parser.add_argument("--sog_mask", choices=["velocity", "gate", "replace", "union"], default="velocity",
                        help="SOG foreground: stock velocity gap, oracle-gated velocity gap, or the oracle mask")
    parser.add_argument("--timing_out", type=str, default=None, help="json with encode/diffusion/decode timings")
    parser.add_argument("--viz_out", type=str, default=None, help=".pt with SOG/attention masks and x0 frames")
    parser.add_argument("--viz_x0_chunks", type=str, default="", help="chunks whose one-step x0 is decoded")
    parser.add_argument("--viz_x0_steps", type=str, default="0,6,7,14", help="0-based denoising steps for x0")
    parser.add_argument("--save_latents", type=str, default=None, help="optional .pt path for output latents")
    parser.add_argument("--force_low_memory", choices=["on", "off"], default=None,
                        help="Override the free-VRAM < 40 GB heuristic. Unset = stock behaviour.")
    return parser


def setup(args):
    pipeline, low_memory, device, local_rank = load_pipe(args)
    # Move pipeline to appropriate dtype and device
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline, low_memory, device, local_rank


def edit_one(args, pipeline, low_memory, device, local_rank):
    # Create output directory (only on main process to avoid race conditions)
    if local_rank == 0:
        os.makedirs(Path(args.save_path).parent, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # load video
    src_video = load_video(args.data_path)
    if args.first_frame_edit is not None:
        src_first_frame = src_video[0]
        trg_first_frame = Image.open(args.first_frame_edit).convert('RGB')
    else:
        src_first_frame = None
        trg_first_frame = None

    height = src_video[0].size[1]
    width = src_video[0].size[0]
    num_frames = len(src_video)
    shot_frames = [int(x) for x in args.shot_frames.split(",")] if args.shot_frames else None
    if shot_frames:
        assert all((n + 3) % 12 == 0 for n in shot_frames), f"each shot must be 12k-3 frames: {shot_frames}"
        assert sum(shot_frames) <= num_frames, (sum(shot_frames), num_frames)
        new_len = sum(shot_frames)
    else:
        new_len = find_closest_num_frame(num_frames)
    src_video = src_video[: new_len]
    num_frames = len(src_video)
    print(num_frames, height, width)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # AE
    torch.cuda.synchronize(); _t_enc0 = time.perf_counter()
    src_video_tensor = torch.stack([transform(img) for img in src_video], dim=1).unsqueeze(0)
    if shot_frames:  # ✨ E0: one VAE encode per shot -> no causal-conv context crosses a cut
        parts, _s = [], 0
        for _n in shot_frames:
            pipeline.vae.model.clear_cache()
            parts.append(pipeline.vae.encode_to_latent(
                src_video_tensor[:, :, _s:_s + _n].to(device=device, dtype=torch.bfloat16)
            ).to(device=device, dtype=torch.bfloat16))
            _s += _n
        video_latents = torch.cat(parts, dim=1)
        shot_latents = [p.shape[1] for p in parts]
        assert all(l % 3 == 0 for l in shot_latents), shot_latents
        cut_chunks = set(int(c) for c in np.cumsum(shot_latents)[:-1] // 3)
        print(f"E0 shots: frames={shot_frames} latents={shot_latents} cut_chunks={sorted(cut_chunks)}")
    else:
        video_latents = pipeline.vae.encode_to_latent(
            src_video_tensor.to(device=device, dtype=torch.bfloat16)
        ).to(device=device, dtype=torch.bfloat16)
        shot_latents, cut_chunks = None, None

    torch.cuda.synchronize(); _t_enc = time.perf_counter() - _t_enc0
    oracle_token_masks = None
    oracle_latent_masks = None
    if args.oracle_mask and args.oracle_version == "v1":
        assert shot_frames, "--oracle_mask needs --shot_frames"
        _mb = np.load(args.oracle_mask)["masks"][:new_len].astype(bool).copy()
        assert _mb.shape[0] == new_len, (_mb.shape, new_len)
        _s, _fixed = 0, 0
        for _n in shot_frames:  # causal glitch filter, restarted at every cut
            _prev = None
            for _t in range(_s, _s + _n):
                _cur = _mb[_t]
                if _prev is not None and _prev.sum() > 0:
                    _iou = (_cur & _prev).sum() / max((_cur | _prev).sum(), 1)
                    if _cur.sum() > 2 * _prev.sum() and _iou < 0.5:
                        _mb[_t] = _prev; _cur = _prev; _fixed += 1
                _prev = _cur
            _s += _n
        _m = torch.from_numpy(_mb).float()
        _toks, _lats, _s = [], [], 0
        _pool = torch.nn.functional
        for _n in shot_frames:
            _g = [[0]] + [list(range(4 * k - 3, 4 * k + 1)) for k in range(1, (_n - 1) // 4 + 1)]
            _px = torch.stack([(_m[_s:_s + _n][g].mean(0) >= 0.5).float() for g in _g])   # majority over frames
            _lat = _pool.avg_pool2d(_px[:, None], 8)[:, 0] >= 0.5                        # [L, 60, 104]
            _tok = _pool.avg_pool2d(_lat.float()[:, None], 2)[:, 0] >= 0.5               # [L, 30, 52]
            if args.oracle_dilate > 0:
                _r = args.oracle_dilate
                _tok = _pool.max_pool2d(_tok.float()[:, None], 2 * _r + 1, 1, _r)[:, 0] > 0
            _lat = _pool.max_pool2d(_lat.float()[:, None], 3, 1, 1)[:, 0] > 0           # +1 latent cell
            if args.oracle_invert:
                _tok, _lat = ~_tok, ~_lat
            _toks.append(_tok.reshape(len(_g), -1)); _lats.append(_lat)
            _s += _n
        oracle_token_masks = torch.cat(_toks).to(device)
        oracle_latent_masks = torch.cat(_lats).to(device)
        print(f"E0 oracle mask v1: glitch frames fixed={_fixed} token coverage={oracle_token_masks.float().mean():.3f} "
              f"latent coverage={oracle_latent_masks.float().mean():.3f}")
    elif args.oracle_mask:
        assert shot_frames, "--oracle_mask needs --shot_frames (latent/frame mapping restarts per shot)"
        _m = torch.from_numpy(np.load(args.oracle_mask)["masks"][:new_len]).float()
        assert _m.shape[0] == new_len, (_m.shape, new_len)
        _toks, _s = [], 0
        for _n in shot_frames:
            _g = [[0]] + [list(range(4 * k - 3, 4 * k + 1)) for k in range(1, (_n - 1) // 4 + 1)]
            _lat = torch.stack([_m[_s:_s + _n][g].amax(0) for g in _g])           # [L, H, W]
            _t = torch.nn.functional.avg_pool2d(_lat[:, None], 16)[:, 0] > 0.1   # [L, 30, 52]
            if args.oracle_dilate > 0:
                _r = args.oracle_dilate
                _t = torch.nn.functional.max_pool2d(_t.float()[:, None], 2 * _r + 1, 1, _r)[:, 0] > 0
            if args.oracle_invert:
                _t = ~_t
            _toks.append(_t.reshape(len(_g), -1))
            _s += _n
        oracle_token_masks = torch.cat(_toks).to(device)
        print(f"E0 oracle mask: {tuple(oracle_token_masks.shape)} coverage={oracle_token_masks.float().mean():.3f}")

    # first frame condition
    independent_first_frame = False
    triple_first_frame = False
    if args.first_frame_edit is not None:
        independent_first_frame = True
        triple_first_frame = False
        src_first_frame = pipeline.vae.encode_to_latent(
            transform(src_first_frame).unsqueeze(0).unsqueeze(2).to(video_latents)
        ).to(video_latents)
        trg_first_frame = pipeline.vae.encode_to_latent(
            transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(video_latents)
        ).to(video_latents)
        if args.triple_first_frame:
            independent_first_frame = False
            triple_first_frame = True
            src_first_frame = src_first_frame.repeat_interleave(3, dim=1)   # [B, F, C, H, W]
            trg_first_frame = trg_first_frame.repeat_interleave(3, dim=1)   # [B, F, C, H, W]

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    torch.cuda.synchronize(); _t_inf0 = time.perf_counter()
    edit_video = pipeline.inference(
        src_video=video_latents,
        src_prompts=args.src_prompt,
        trg_prompts=args.trg_prompt,
        src_trigger_words=args.src_word,
        trg_trigger_words=args.trg_word,
        return_latents=True,
        wo_video_decode=True,
        profile=False,
        low_memory=low_memory,

        independent_first_frame=independent_first_frame,
        triple_first_frame=triple_first_frame,
        src_initial_latent=src_first_frame,
        trg_initial_latent=trg_first_frame,

        fg_boost_factor=args.fg_boost_factor,
        blend_power=args.blend_power,
        cut_chunks=cut_chunks,
        reset_at_cut=args.reset_at_cut,
        oracle_token_masks=oracle_token_masks,
        sog_mask=args.sog_mask,
        oracle_attn=args.oracle_attn,
        oracle_latent_masks=oracle_latent_masks,
        viz=bool(args.viz_out),
        viz_x0_chunks=[int(x) for x in args.viz_x0_chunks.split(",") if x],
        viz_x0_steps=[int(x) for x in args.viz_x0_steps.split(",") if x],
    )
    torch.cuda.synchronize(); _t_inf = time.perf_counter() - _t_inf0
    _, out_latents = edit_video
    if args.save_latents:
        torch.save(out_latents.cpu(), args.save_latents)
    # ✨ E0: per-shot decode; without --shot_frames one segment = the stock joint decode
    _segs = shot_latents if shot_frames else [out_latents.shape[1]]
    torch.cuda.synchronize(); _t_dec0 = time.perf_counter()
    vids, _s = [], 0
    for _l in _segs:
        pipeline.vae.model.clear_cache()
        _v = pipeline.vae.decode_to_pixel(out_latents[:, _s:_s + _l], use_cache=False)
        vids.append((_v * 0.5 + 0.5).clamp(0, 1))
        _s += _l
    edit_video = torch.cat(vids, dim=1)
    torch.cuda.synchronize(); _t_dec = time.perf_counter() - _t_dec0
    _tm = getattr(pipeline, "timing", None) or {}
    timing = {"frames": int(edit_video.shape[1]), "encode_s": _t_enc, "inference_s": _t_inf,
              "diffusion_s": sum(_tm.get("chunk_ms", [])) / 1e3, "chunk_ms": _tm.get("chunk_ms", []),
              "viz_overhead_s": _tm.get("viz_ms", 0.0) / 1e3, "decode_s": _t_dec}
    timing["total_s"] = _t_enc + _t_inf - timing["viz_overhead_s"] + _t_dec
    timing["s_per_frame"] = timing["total_s"] / timing["frames"]
    print("TIMING", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in timing.items() if k != "chunk_ms"}))
    if args.timing_out:
        json.dump(timing, open(args.timing_out, "w"), indent=1)
    if args.viz_out and getattr(pipeline, "viz", None) is not None:
        V = pipeline.viz
        _seg_start = np.concatenate([[0], np.cumsum(_segs)[:-1]]).astype(int)
        x0_frames = {}
        for (c, st), lat in V["x0"].items():
            a = int(_seg_start[np.searchsorted(_seg_start, 3 * c, side="right") - 1])
            L = out_latents[:, a:3 * c + 3].clone()
            L[:, 3 * c - a:] = lat.unsqueeze(0).to(L)
            pipeline.vae.model.clear_cache()
            fr = ((pipeline.vae.decode_to_pixel(L, use_cache=False)[0] * 0.5 + 0.5).clamp(0, 1) * 255).byte().cpu()
            n_chunk = 9 if 3 * c == a else 12
            fr = fr[-n_chunk:].permute(0, 2, 3, 1)
            x0_frames[(c, st)] = fr[[0, n_chunk // 2]]
        torch.save({"sog_fg": [torch.stack(s) for s in V["sog_fg"]],
                    "sog_corr": [torch.stack(s) for s in V["sog_corr"]],
                    "attn_src": V["attn_src"], "attn_tinj": V["attn_tinj"],
                    "x0_frames": x0_frames, "seg_latents": [int(x) for x in _segs]}, args.viz_out)
        pipeline.vae.model.clear_cache()

    # Clear VAE cache
    pipeline.vae.model.clear_cache()
    write_video(
        args.save_path, 
        rearrange(edit_video[0], 't c h w -> t h w c').cpu() * 255, 
        fps=16
    )


if __name__ == '__main__':
    args = build_parser().parse_args()
    edit_one(args, *setup(args))
