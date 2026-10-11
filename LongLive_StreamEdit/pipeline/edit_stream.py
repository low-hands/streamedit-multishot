"""✨ Chunk-by-chunk entry to EditCausalInferencePipeline, for real streaming.

EditStreamSession holds the state that EditCausalInferencePipeline.inference keeps in locals and
step() runs exactly one iteration of its temporal loop (Step 3) on the chunk it is given, so source
latents and masks can arrive while the stream is running. The arithmetic is the loop body of
inference() with oracle grounding (token masks given per chunk); viz, mask dumps and the first-frame
condition are left out. With the same seed and the same chunks the outputs match inference().

StreamVAE encodes / decodes chunk by chunk with the causal caches of the Wan VAE, restarting both at a
shot start (the first chunk of a shot is 1 + 4 + 4 frames, later chunks 4 + 4 + 4).
"""
import time
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from utils.memory import get_cuda_free_memory_gb  # noqa: F401  (same import surface as inference)
from .utils import find_phrase_token_indices


class EditStreamSession:
    def __init__(self, pipe, src_prompt, trg_prompt, src_word, trg_word, device, dtype=torch.bfloat16,
                 fg_boost_factor=2.0, blend_power=2.0, sog_mask="gate", oracle_attn="replace",
                 reset_at_cut="none", edit_mem=0, mem_delta=3, mask_layers: Iterable = range(20),
                 enhance_layers: Iterable = range(30), fg_scale=1.0, reuse_noise_temporal_mean=True,
                 low_memory=False, height=60, width=104):
        self.p, self.device, self.dtype = pipe, device, dtype
        self.fg_boost_factor, self.blend_power = fg_boost_factor, blend_power
        self.sog_mask, self.oracle_attn, self.reset_at_cut = sog_mask, oracle_attn, reset_at_cut
        self.mask_layers, self.enhance_layers, self.fg_scale = mask_layers, enhance_layers, fg_scale
        self.low_memory, self.height, self.width = low_memory, height, width
        p, B = pipe, 1
        src_prompts, trg_prompts, src_words, trg_words = p._check_prompts(src_prompt, trg_prompt, src_word, trg_word)
        self.conditional_dict = p.text_encoder(text_prompts=src_prompts + trg_prompts)
        self.src_conditional_dict = p.text_encoder(text_prompts=src_prompts)
        self.trg_conditional_dict = p.text_encoder(text_prompts=trg_prompts)

        self.kv_cache_src = p._initialize_kv_cache(batch_size=B, dtype=dtype, device=device)
        self.kv_cache_trg = p._initialize_kv_cache(batch_size=B, dtype=dtype, device=device)
        self.trg_fg_mask_cache = p._initialize_trg_fg_mask_cache(batch_size=B, device=device)
        self.crossattn_cache_src = p._initialize_crossattn_cache(batch_size=B, dtype=dtype, device=device)
        self.crossattn_cache_trg = p._initialize_crossattn_cache(batch_size=B, dtype=dtype, device=device)
        self.crossattn_cache_dual = p._initialize_crossattn_cache(batch_size=B * 2, dtype=dtype, device=device)
        p._initialize_noise_statistics(reuse_noise_temporal_mean)

        tok = p.text_encoder.tokenizer.tokenizer
        self.tok_src = find_phrase_token_indices(tok, src_prompts, src_words)
        self.tok_trg = find_phrase_token_indices(tok, trg_prompts, trg_words)
        p.generator.model.local_attn_size = p.local_attn_size
        p._set_all_modules_max_attention_size(p.local_attn_size)

        from wan.modules import causal_model as _cm
        self._cm = _cm
        _cm.EDIT_MEM.update(on=edit_mem > 0, write=None, read=False, cap=edit_mem, delta=mem_delta, k={}, v={}, hw={})
        for i, blk in enumerate(p.generator.model.blocks):
            blk.self_attn.layer_idx = i
        self.current_start_frame, self.pos0, self.chunk_idx = 0, 0, -1
        self.chunk_ms = []

    @torch.no_grad()
    def step(self, src_input, token_mask, new_shot=False):
        """src_input [1, F, C, H, W] source latents of this chunk; token_mask [F, H/2 * W/2] bool (oracle
        grounding, already dilated / inverted). new_shot: this chunk starts a shot that is not the first.
        Returns the edited latents [1, F, C, H, W]."""
        p, cm = self.p, self._cm
        torch.cuda.synchronize(); t0 = time.perf_counter()
        self.chunk_idx += 1
        B, Fn = 1, src_input.shape[1]
        height, width = self.height, self.width
        kv_cache_src, kv_cache_trg, trg_fg_mask_cache = self.kv_cache_src, self.kv_cache_trg, self.trg_fg_mask_cache
        if new_shot and self.reset_at_cut != "none":
            if self.reset_at_cut == "all":
                self.pos0 = self.current_start_frame
            p._reset_states_at_cut(kv_cache_src, kv_cache_trg, trg_fg_mask_cache,
                                   (self.current_start_frame - self.pos0) * p.frame_seq_length,
                                   keep_sink=(self.reset_at_cut == "keep_sink"))
        start = (self.current_start_frame - self.pos0) * p.frame_seq_length
        denoised_pred = src_input
        context_timestep = torch.ones([B, Fn], device=src_input.device, dtype=torch.float32) * p.args.context_noise

        shared_dict_dual = dict()
        p._kv_cache_to(kv_cache_trg, 'cuda', self.low_memory)
        kv_cache_dual = p._concat_kv_cache(kv_cache_src, kv_cache_trg, shared_dict=shared_dict_dual)
        p._kv_cache_to(kv_cache_trg, 'cpu', self.low_memory)

        p._register_crossattn_mask_gatherer(self.crossattn_cache_src, self.tok_src, layers=self.mask_layers,
                                            fg_scale=self.fg_scale)
        p.generator(noisy_image_or_video=src_input, conditional_dict=self.src_conditional_dict,
                    timestep=context_timestep, kv_cache=kv_cache_src, crossattn_cache=self.crossattn_cache_src,
                    current_start=start)
        _, src_fg_mask_bin, _, _ = p._aggregate_crossattn_mask(self.crossattn_cache_src)
        src_fg_mask_bin = token_mask.reshape(1, -1).to(src_fg_mask_bin.device).expand_as(src_fg_mask_bin).clone()
        p._inject_masks_to_kv_cache(kv_cache_dual, trg_fg_mask_cache, src_fg_mask_bin)
        src_fg_mask_map = p._mask_reshape(src_fg_mask_bin, size=(Fn, height, width))
        inloop_trg_fg_mask = src_fg_mask_bin
        cm.EDIT_MEM["read"] = bool(cm.EDIT_MEM["k"])

        steps = p.denoising_step_list
        noisy_pred_input = None
        for index, current_timestep in enumerate(steps):
            timestep = torch.ones([B * 2, Fn], device=src_input.device) * current_timestep
            timestep_next = steps[index + 1] / 1000 if (index < len(steps) - 1) else 0
            shared_dict_dual['current_timestep'] = float(timestep_next)
            shared_dict_dual['current_timestep_index'] = index
            shared_dict_dual['total_timestep'] = len(steps)
            shared_dict_dual['blend_power'] = self.blend_power

            fwd_noise = torch.randn_like(src_input)
            fwd_noise = p._reuse_noise_statistics(fwd_noise, index, fg_mask=src_fg_mask_map)
            fwd_trg_noise = fwd_noise
            if index == len(steps) // 2:
                p._register_crossattn_mask_gatherer(self.crossattn_cache_dual, self.tok_src + self.tok_trg,
                                                    layers=self.mask_layers, fg_scale=self.fg_scale)
            if self.fg_boost_factor != 1.0:
                p._register_crossattn_enhancement(self.crossattn_cache_dual, self.tok_src + self.tok_trg,
                                                  layers=self.enhance_layers, fg_boost_factor=self.fg_boost_factor,
                                                  current_src_fg_mask=inloop_trg_fg_mask)
            noisy_src_input = p.scheduler.add_noise(src_input.flatten(0, 1), fwd_noise.flatten(0, 1),
                                                    timestep[:B]).unflatten(0, src_input.shape[:2])
            noisy_pred_input = p.scheduler.add_noise(denoised_pred.flatten(0, 1), fwd_trg_noise.flatten(0, 1),
                                                     timestep[B:]).unflatten(0, denoised_pred.shape[:2])
            noisy_input = torch.cat([noisy_src_input, noisy_pred_input], dim=0)
            velocity_pred, _ = p.generator(noisy_image_or_video=noisy_input, conditional_dict=self.conditional_dict,
                                           timestep=timestep, kv_cache=kv_cache_dual,
                                           crossattn_cache=self.crossattn_cache_dual, current_start=start)
            t_i = current_timestep / 1000
            v_src, v_trg = velocity_pred.chunk(2, dim=0)
            v_gt = fwd_noise - src_input

            fg_mask = (v_trg - v_src).abs().mean(dim=2, keepdim=True)     # [B, F, 1, H, W]
            data_dims = list(range(fg_mask.ndim))[1:]
            if self.sog_mask == "velocity":
                fg_mask = (fg_mask - fg_mask.amin(dim=data_dims, keepdim=True)) / \
                    (fg_mask.amax(dim=data_dims, keepdim=True) - fg_mask.amin(dim=data_dims, keepdim=True) + 1e-7)
            else:  # SOG grounded on the token mask (v0: no separate latent mask)
                _o = token_mask.float().reshape(Fn, 1, height // 2, width // 2).to(fg_mask.device)
                _o = F.interpolate(_o, size=(height, width), mode="nearest")
                _o = F.avg_pool2d(_o, 3, stride=1, padding=1, count_include_pad=False)
                _o = _o.unsqueeze(0).to(fg_mask)
                if self.sog_mask == "replace":
                    fg_mask = _o.expand_as(fg_mask)
                elif self.sog_mask == "union":
                    _v = (fg_mask - fg_mask.amin(dim=data_dims, keepdim=True)) / \
                        (fg_mask.amax(dim=data_dims, keepdim=True) - fg_mask.amin(dim=data_dims, keepdim=True) + 1e-7)
                    fg_mask = torch.maximum(_o.expand_as(_v), _v)
                else:  # gate
                    _in = _o > 0.5
                    _lo = torch.where(_in, fg_mask, torch.full_like(fg_mask, float("inf"))).amin(dim=data_dims, keepdim=True)
                    _hi = torch.where(_in, fg_mask, torch.full_like(fg_mask, float("-inf"))).amax(dim=data_dims, keepdim=True)
                    _norm = ((fg_mask - _lo) / (_hi - _lo + 1e-7)).clamp(0, 1)
                    _norm = torch.where(torch.isfinite(_norm), _norm, torch.zeros_like(_norm))
                    fg_mask = _o * _norm
            bg_mask = 1 - fg_mask
            v_t = v_trg + bg_mask * (v_gt - v_src)
            denoised_pred = noisy_pred_input - t_i * v_t

            if index == len(steps) // 2:
                _, inloop_src_trg_fg_mask_bin, _, _ = p._aggregate_crossattn_mask(
                    self.crossattn_cache_dual, size=(Fn, height, width), scale_factor=16)
                inloop_trg_fg_mask_bin = inloop_src_trg_fg_mask_bin.chunk(2, dim=0)[1]
                inloop_trg_fg_mask = src_fg_mask_bin if self.oracle_attn == "replace" \
                    else (inloop_trg_fg_mask_bin | src_fg_mask_bin)
                p._inject_masks_to_kv_cache(kv_cache_dual, trg_fg_mask_cache, inloop_trg_fg_mask)

        del kv_cache_dual
        p._kv_cache_to(kv_cache_trg, 'cuda', self.low_memory)
        p._register_crossattn_mask_gatherer(self.crossattn_cache_trg, self.tok_trg, layers=self.mask_layers,
                                            fg_scale=self.fg_scale)
        cm.EDIT_MEM["read"] = False
        if cm.EDIT_MEM["on"]:  # edit memory: write object tokens
            _om = token_mask.float().reshape(Fn, 1, height // 2, width // 2)
            _om = -F.max_pool2d(-_om, 3, 1, 1)
            cm.EDIT_MEM["write"] = _om.reshape(-1) > 0.5
        p.generator(noisy_image_or_video=denoised_pred, conditional_dict=self.trg_conditional_dict,
                    timestep=context_timestep, kv_cache=kv_cache_trg, crossattn_cache=self.crossattn_cache_trg,
                    current_start=start)
        cm.EDIT_MEM["write"] = None
        _, trg_fg_mask_bin, _, _ = p._aggregate_crossattn_mask(self.crossattn_cache_trg)
        current_trg_fg_mask = src_fg_mask_bin if self.oracle_attn == "replace" else (trg_fg_mask_bin | src_fg_mask_bin)
        p._update_trg_fg_mask_cache(trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg)
        p._kv_cache_to(kv_cache_trg, 'cpu', self.low_memory)

        self.current_start_frame += Fn
        torch.cuda.synchronize()
        self.chunk_ms.append((time.perf_counter() - t0) * 1e3)
        return denoised_pred


class StreamVAE:
    """Chunk-wise Wan VAE with its causal caches; new_shot() restarts both encoder and decoder."""

    def __init__(self, vae):
        self.vae, self.m = vae, vae.model
        self.first = True

    def new_shot(self):
        self.m.clear_cache()
        self.first = True

    @torch.no_grad()
    def encode(self, px):
        """px [1, 3, T, H, W] in [-1, 1]; T = 9 for the first chunk of a shot, 12 otherwise -> [1, L, 16, h, w]."""
        m, T = self.m, px.shape[2]
        edges = ([0, 1] + list(range(5, T + 1, 4))) if self.first else list(range(0, T + 1, 4))
        assert edges[-1] == T, (T, self.first)
        outs = []
        for a, b in zip(edges[:-1], edges[1:]):
            m._enc_conv_idx = [0]
            outs.append(m.encoder(px[:, :, a:b], feat_cache=m._enc_feat_map, feat_idx=m._enc_conv_idx))
        mu, _ = m.conv1(torch.cat(outs, 2)).chunk(2, dim=1)
        mean = self.vae.mean.to(device=px.device, dtype=px.dtype).view(1, m.z_dim, 1, 1, 1)
        inv_std = (1.0 / self.vae.std.to(device=px.device, dtype=px.dtype)).view(1, m.z_dim, 1, 1, 1)
        mu = (mu - mean) * inv_std
        self.first = False
        return mu.float().permute(0, 2, 1, 3, 4).to(px.dtype)

    @torch.no_grad()
    def decode(self, z):
        """z [1, L, 16, h, w] -> frames [T, 3, H, W] in [0, 1] (decoder cache carried across chunks of a shot)."""
        x = self.vae.decode_to_pixel(z, use_cache=True)[0]
        return (x * 0.5 + 0.5).clamp(0, 1)
