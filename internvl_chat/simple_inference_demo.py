#!/usr/bin/env python3
"""
InternVL‑Chat 双编码器推理 Demo
================================
用法示例
--------
python demo_internvl_dual.py \
    --checkpoint /path/to/dual_encoder_checkpoint \
    --input      /path/to/your/image_or_video.jpg \
    --question   "Describe what you see in detail."
"""

import argparse
import os
import random
import warnings
from typing import List, Union

import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoTokenizer, GenerationConfig

from internvl.model import InternVLChatConfig, InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.train.vggt_preprocess import load_and_preprocess_images as preprocess_images2
from internvl.conversation import get_conv_template

# ===== special tokens =====
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------
# 1. 载入多模态数据
# ------------------------------------------------------------------
def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode == "RGBA":          # 转白底
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _get_frame_indices(num_frames, vlen, sample='rand', fix_start=None,
                       input_fps=1.0, max_num_frames=-1) -> List[int]:
    """从视频里抽帧（与训练采样保持一致）"""
    if sample in ['rand', 'middle']:
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(0, vlen, acc_samples + 1).astype(int)
        ranges = list(zip(intervals[:-1], intervals[1:] - 1))

        if sample == 'rand':
            frame_ids = [random.choice(range(lo, hi + 1)) if lo <= hi else lo
                         for lo, hi in ranges]
        elif sample == 'middle':
            frame_ids = [(lo + hi) // 2 for lo, hi in ranges]
        else:
            raise NotImplementedError

        # 不足时补最后一帧
        if len(frame_ids) < num_frames:
            frame_ids += [frame_ids[-1]] * (num_frames - len(frame_ids))

    elif sample.startswith('fps'):
        out_fps = float(sample[3:])
        duration = vlen / input_fps
        delta = 1 / out_fps
        secs = np.arange(delta / 2, duration + delta / 2, delta)
        frame_ids = np.around(secs * input_fps).astype(int)
        frame_ids = [i for i in frame_ids if i < vlen]
        if 0 < max_num_frames < len(frame_ids):
            frame_ids = frame_ids[:max_num_frames]
    else:
        raise ValueError(sample)

    return frame_ids


def load_video_frames(video_path: str,
                      min_frames=8,
                      max_frames=16,
                      sampling='rand') -> List[Image.Image]:
    """读取视频并抽帧为 PIL 图片列表"""
    try:
        if video_path.endswith('.gif'):
            gif = imageio.get_reader(video_path)
            vlen = len(gif)
            t_frames = random.randint(min_frames, max_frames)
            ids = _get_frame_indices(t_frames, vlen, sampling)
            frames = []
            for i, frame_np in enumerate(gif):
                if i in ids:
                    if frame_np.ndim == 3 and frame_np.shape[2] == 4:
                        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2RGB)
                    elif frame_np.ndim == 2:
                        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
                    frames.append(Image.fromarray(frame_np))
            return frames
        else:
            vr = VideoReader(video_path, num_threads=1)
            vlen = len(vr)
            fps = vr.get_avg_fps()
            t_frames = random.randint(min_frames, max_frames)
            t_frames = min(t_frames, vlen) or vlen
            ids = _get_frame_indices(t_frames, vlen, sampling, input_fps=fps,
                                     max_num_frames=max_frames)
            ids = [i for i in ids if i < vlen]
            if not ids:
                return []
            batch = vr.get_batch(ids).asnumpy()
            return [Image.fromarray(frame) for frame in batch]
    except Exception as e:
        print(f"[load_video_frames] {e}")
        return []


# ------------------------------------------------------------------
# 2. 视觉预处理
# ------------------------------------------------------------------
def preprocess_images(images: List[Image.Image],
                      image_size=448,
                      dynamic_size=True,
                      use_thumbnail=True,
                      max_patches=4):
    """返回两个 encoder 对应的 pixel_values"""
    # --- Encoder‑1 ---
    transform = build_transform(is_train=False, input_size=image_size)
    all_proc = []
    for img in images:
        if dynamic_size:
            all_proc.extend(dynamic_preprocess(img, 1, max_patches,
                                               image_size, use_thumbnail))
        else:
            all_proc.append(img)
    pv1 = torch.stack([transform(i) for i in all_proc])          # (K,3,H,W)

    # --- Encoder‑2 (VGGT) ---
    pv2 = preprocess_images2(images, mode='pad')                 # (K2,3,H,W)
    pv2 = pv2.unsqueeze(0)                                       # (1,S,C,H,W)

    # --- VGGT window 对齐 ---
    B, S = pv2.shape[:2]
    WSIZE = 2
    if S % WSIZE:
        tgt = (S + WSIZE - 1) // WSIZE * WSIZE
        pad = tgt - S
        pv2 = torch.nn.functional.pad(pv2,
                                      (0, 0, 0, 0, 0, 0, 0, pad))  # seq 维填 0
        print(f"[preprocess] pad frames {S}->{pv2.shape[1]}")

    # --- dtype & device ---
    # Use consistent dtype that works with both models
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device('cuda')
    
    pv1 = pv1.to(device=device, dtype=dtype)
    pv2 = pv2.to(device=device, dtype=dtype)
    
    return pv1, pv2


# ------------------------------------------------------------------
# 3. 加载模型 & tokenizer
# ------------------------------------------------------------------
def load_model_and_tokenizer(ckpt):
    print("⇢ Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)

    special = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
               QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
               REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    n_new = tok.add_tokens(special, special_tokens=True)

    print("⇢ Loading config")
    cfg = InternVLChatConfig.from_pretrained(ckpt)
    if cfg.llm_config.model_type == 'internlm2':
        cfg.llm_config.attn_implementation = 'flash_attention_2'
    else:
        cfg.llm_config._attn_implementation = 'flash_attention_2'

    print("⇢ Loading model")
    model = InternVLChatModel.from_pretrained(
        ckpt,
        config=cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map='cuda'
    )

    model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    if n_new:
        model.language_model.resize_token_embeddings(len(tok))
        with torch.no_grad():
            emb = model.language_model.get_output_embeddings().weight
            emb[-n_new:] = emb[:-n_new].mean(dim=0, keepdim=True)

    # Fix vision_model2 parameter mapping: load .weight -> .gamma
    print("⇢ Fixing vision_model2 parameter mapping (.weight -> .gamma)...")
    try:
        import glob
        from safetensors import safe_open
        
        # Find safetensor files
        safetensor_files = glob.glob(os.path.join(ckpt, "model-*.safetensors"))
        if not safetensor_files:
            print("   Warning: No safetensor files found for parameter mapping")
        else:
            print(f"   Found {len(safetensor_files)} safetensor files")
            
            # First, let's see what LayerScale parameters the model actually has
            model_gamma_params = []
            model_weight_params = []
            for name, param in model.named_parameters():
                if 'vision_model2' in name and ('ls1' in name or 'ls2' in name):
                    if '.gamma' in name:
                        model_gamma_params.append(name)
                    elif '.weight' in name:
                        model_weight_params.append(name)
            
            print(f"   Model has {len(model_gamma_params)} .gamma LayerScale params")
            print(f"   Model has {len(model_weight_params)} .weight LayerScale params")
            if model_gamma_params:
                print(f"   Sample .gamma param: {model_gamma_params[0]}")
            if model_weight_params:
                print(f"   Sample .weight param: {model_weight_params[0]}")
            
            # Load checkpoint weights - look for all vision_model2 LayerScale parameters
            checkpoint_weights = {}
            checkpoint_gamma_keys = []
            checkpoint_weight_keys = []
            
            for file in safetensor_files:
                with safe_open(file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if 'vision_model2' in key and ('ls1' in key or 'ls2' in key):
                            checkpoint_weights[key] = f.get_tensor(key)
                            if '.gamma' in key:
                                checkpoint_gamma_keys.append(key)
                            elif '.weight' in key:
                                checkpoint_weight_keys.append(key)
            
            print(f"   Checkpoint has {len(checkpoint_gamma_keys)} .gamma LayerScale keys")
            print(f"   Checkpoint has {len(checkpoint_weight_keys)} .weight LayerScale keys")
            if checkpoint_gamma_keys:
                print(f"   Sample checkpoint .gamma key: {checkpoint_gamma_keys[0]}")
            if checkpoint_weight_keys:
                print(f"   Sample checkpoint .weight key: {checkpoint_weight_keys[0]}")
            
            # Try to map parameters
            mapped_count = 0
            with torch.no_grad():
                # First try direct gamma -> gamma mapping
                for name in model_gamma_params:
                    if name in checkpoint_weights:
                        param = dict(model.named_parameters())[name]
                        weight_tensor = checkpoint_weights[name].to(param.device, dtype=param.dtype)
                        param.copy_(weight_tensor)
                        mapped_count += 1
                        if mapped_count <= 3:
                            print(f"   Direct mapped: {name}")
                
                # Then try weight -> gamma mapping
                for name in model_gamma_params:
                    weight_name = name.replace('.gamma', '.weight')
                    if weight_name in checkpoint_weights:
                        param = dict(model.named_parameters())[name]
                        weight_tensor = checkpoint_weights[weight_name].to(param.device, dtype=param.dtype)
                        param.copy_(weight_tensor)
                        mapped_count += 1
                        if mapped_count <= 6:
                            print(f"   Cross mapped: {weight_name} -> {name}")
            
            if mapped_count > 6:
                print(f"   ... and {mapped_count - 6} more mappings")
            print(f"   Successfully mapped {mapped_count} parameters total")
            
    except Exception as e:
        print(f"   Warning: Parameter mapping failed: {e}")
        import traceback
        traceback.print_exc()

    model = model.cuda().eval()
    return model, tok


# ------------------------------------------------------------------
# 4. 给模型打补丁：支持双编码器 chat / generate
# ------------------------------------------------------------------
def patch_model_chat_method(model):
    """给实例注入 chat & generate_with_dual_encoder"""

    import types

    # ---------- generate ----------
    def generate_with_dual_encoder(self,
                                   pixel_values=None,
                                   pixel_values2=None,
                                   attention_mask2=None,
                                   input_ids=None,
                                   attention_mask=None,
                                   visual_features=None,
                                   generation_config=None,
                                   output_hidden_states=None,
                                   **gen_kwargs):
        """双编码器推理"""
        if isinstance(generation_config, dict) or generation_config is None:
            generation_config = GenerationConfig(**(generation_config or {}))

        # —— 视觉特征 —— #
        if pixel_values is not None:
            vit1 = (visual_features
                    if visual_features is not None
                    else self.extract_feature(pixel_values))      # (Pk,Tk,E) ‑> reshape 后 (N1,E)

            if pixel_values2 is not None:
                # Fix dtype mismatch issue: ensure pixel_values2 has the same dtype as the model
                model_dtype = next(self.vision_model2.parameters()).dtype
                pixel_values2 = pixel_values2.to(model_dtype)
                
                # Fix the normalization buffers dtype issue in Aggregator
                if hasattr(self.vision_model2, '_resnet_mean') and hasattr(self.vision_model2, '_resnet_std'):
                    if self.vision_model2._resnet_mean.dtype != model_dtype:
                        self.vision_model2._resnet_mean = self.vision_model2._resnet_mean.to(model_dtype)
                        self.vision_model2._resnet_std = self.vision_model2._resnet_std.to(model_dtype)
                
                vit2 = self.extract_feature2(pixel_values2, attention_mask2)  # (1,T2,E)
                vit_all = torch.cat([vit2.reshape(-1, vit2.size(-1)),
                                     vit1.reshape(-1, vit1.size(-1))], dim=0)
            else:
                vit_all = vit1.reshape(-1, vit1.size(-1))
        else:
            vit_all = None

        # —— 文本嵌入 —— #
        txt = self.language_model.get_input_embeddings()(input_ids)   # (B,N,E)
        B, N, C = txt.shape
        txt = txt.view(B * N, C)

        if vit_all is not None:
            ids_flat = input_ids.view(B * N)
            sel = (ids_flat == self.img_context_token_id)
            assert sel.sum() == vit_all.size(0), \
                f"视觉 token ({vit_all.size(0)}) ≠ 选中位置 ({sel.sum()})"
            txt[sel] = vit_all.to(txt.device)
        txt = txt.view(B, N, C)

        # —— 调底层 LLM 生成 —— #
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = getattr(self.language_model.config, 'bos_token_id', 1)
        
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = getattr(self.language_model.config, 'eos_token_id', 2)
        
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = getattr(self.language_model.config, 'pad_token_id', 
                                                   generation_config.eos_token_id)
        
        outs = self.language_model.generate(
            inputs_embeds=txt,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **gen_kwargs,
        )
        return outs

    # ---------- chat ----------
    def chat_with_dual_encoder(self,
                               tokenizer,
                               pixel_values,
                               question,
                               generation_config,
                               pixel_values2=None,
                               attention_mask2=None,
                               history=None,
                               return_history=False,
                               num_patches_list=None,
                               IMG_START_TOKEN='<img>',
                               IMG_END_TOKEN='</img>',
                               IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                               verbose=False):
        """模仿原始 chat，但支持第二路 encoder"""
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        tmpl = get_conv_template(self.template)
        tmpl.system_message = self.system_message
        eos_id = tokenizer.convert_tokens_to_ids(tmpl.sep.strip())

        # —— 构造 prompt —— #
        hist = history or []
        for q, a in hist:
            tmpl.append_message(tmpl.roles[0], q)
            tmpl.append_message(tmpl.roles[1], a)
        tmpl.append_message(tmpl.roles[0], question)
        tmpl.append_message(tmpl.roles[1], None)
        prompt = tmpl.get_prompt()

        # —— 插入 <image> token 总数 —— #
        n_img2 = 13 * 13 + 1    # 训练时写死 170
        for n_patch in num_patches_list:
            total = self.num_image_token * n_patch + n_img2
            img_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total + IMG_END_TOKEN
            prompt = prompt.replace('<image>', img_tokens, 1)
            if verbose:
                print(f"[prompt] tokens: enc1={self.num_image_token * n_patch}, "
                      f"enc2={n_img2}, total={total}")

        model_inp = tokenizer(prompt, return_tensors='pt')
        device = next(self.parameters()).device
        input_ids = model_inp['input_ids'].to(device)
        attention_mask = model_inp['attention_mask'].to(device)

        gen_cfg = GenerationConfig(**generation_config)
        gen_cfg.eos_token_id = eos_id

        outs = self.generate_with_dual_encoder(
            pixel_values=pixel_values,
            pixel_values2=pixel_values2,
            attention_mask2=attention_mask2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )

        txt = tokenizer.batch_decode(outs, skip_special_tokens=True)[0]
        answer = txt.split(tmpl.sep.strip())[0].strip()
        hist.append((question, answer))

        if return_history:
            return answer, hist
        else:
            if verbose:
                print("[Chat] user:", question)
                print("[Chat] bot :", answer)
            return answer

    # —— 绑定到实例 —— #
    model.generate_with_dual_encoder = types.MethodType(generate_with_dual_encoder, model)
    model.chat = types.MethodType(chat_with_dual_encoder, model)
    return model


# ------------------------------------------------------------------
# 5. CLI 调用
# ------------------------------------------------------------------
def simple_chat(model, tokenizer,
                pixel_values, pixel_values2,
                question, max_tokens=1000):
    """外部简单包装一下"""
    if pixel_values2 is not None:
        # attention_mask2: 1 表示非零帧
        attn2 = (pixel_values2.abs().sum(dim=(2, 3, 4)) > 0)
    else:
        attn2 = None

    gen_cfg = dict(
        max_new_tokens=max_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=getattr(tokenizer, 'bos_token_id', tokenizer.eos_token_id),
        eos_token_id=tokenizer.eos_token_id,
    )

    return model.chat(tokenizer=tokenizer,
                      pixel_values=pixel_values,
                      pixel_values2=pixel_values2,
                      attention_mask2=attn2,
                      question=question,
                      generation_config=gen_cfg,
                      verbose=True)


def main():
    parser = argparse.ArgumentParser(description='InternVL‑Chat Dual‑Encoder Demo')
    parser.add_argument('--checkpoint', default='/mnt/chensenda/codes/VLN/InternVL/internvl_chat/work_dirs/internvl_chat_dual_encoder/internvl_chat_dual_encoder_2b_mix_stage1/checkpoint-4200',
                        help='Path to dual‑encoder checkpoint')
    parser.add_argument('--input', required=True,
                        help='Image / video path')
    parser.add_argument('--question', default='Describe what you see in detail.',
                        help='Prompt for the model')
    parser.add_argument('--num-frames', type=int, default=8,
                        help='Num video frames (sampling)')
    parser.add_argument('--max-patches', type=int, default=4,
                        help='Dynamic patches per image')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Generation length')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    # ---- load model ----
    model, tok = load_model_and_tokenizer(args.checkpoint)
    model = patch_model_chat_method(model)

    # ---- load media ----
    inp = args.input.lower()
    if inp.endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif')):
        imgs = load_video_frames(args.input,
                                 min_frames=args.num_frames,
                                 max_frames=args.num_frames,
                                 sampling='rand')
        if not imgs:
            raise RuntimeError("Video load failed")
        frame_info = '\n'.join([f'Frame-{i+1}: <image>' for i in range(len(imgs))])
        q = args.question.replace('<video>', frame_info) if '<video>' in args.question \
            else frame_info + '\n' + args.question
    else:
        imgs = [load_image(args.input)]
        q = '<image>\n' + args.question if '<image>' not in args.question else args.question

    # ---- preprocess ----
    print("⇢ Preprocessing images / frames")
    pv1, pv2 = preprocess_images(imgs,
                                 image_size=model.config.force_image_size
                                 or model.config.vision_config.image_size,
                                 dynamic_size=True,
                                 use_thumbnail=model.config.use_thumbnail,
                                 max_patches=args.max_patches)
    print(f"Encoder‑1 tensor : {tuple(pv1.shape)}")
    print(f"Encoder‑2 tensor : {tuple(pv2.shape)}")

    # ---- inference ----
    print("⇢ Generating ...")
    answer = simple_chat(model, tok, pv1, pv2,
                         question=q, max_tokens=args.max_tokens)

    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == '__main__':
    main()
