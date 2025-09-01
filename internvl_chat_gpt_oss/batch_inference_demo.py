#!/usr/bin/env python3
"""
InternVL‑Chat Dual Encoder Batch Inference Demo
===============================================
Usage examples:
--------
# Single input with multiple questions
python batch_inference_demo.py \
    --input "/path/to/image_or_video.jpg" \
    --questions "Describe what you see." "What is the main object?" \
    --output results.json

# Multiple inputs with multiple questions  
python batch_inference_demo.py \
    --inputs "/path/to/image1.jpg" "/path/to/video1.mp4" \
    --questions "Describe what you see." "What is happening?" \
    --output results.json

# From file lists
python batch_inference_demo.py \
    --input-file inputs.txt \
    --question-file questions.txt \
    --output results.json
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Commented out to avoid conflicts with other scripts
import random
import warnings
from typing import List, Union, Dict, Any
import json
from tqdm import tqdm

import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from transformers import AutoTokenizer, GenerationConfig

from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.model.internvl_chat.conversation import get_conv_template

# ===== special tokens =====
# 特殊tokens的定义
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
# 1. Load multimodal data
# ------------------------------------------------------------------
def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    if img.mode == "RGBA":          # Convert to white background
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _get_frame_indices(num_frames, vlen, sample='rand', fix_start=None,
                       input_fps=1.0, max_num_frames=-1) -> List[int]:
    """Sample frames from video (consistent with training sampling)"""
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

        # Pad with last frame if insufficient
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
    """Load video and extract frames as PIL image list"""
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


def load_media_batch(input_paths: List[str], num_frames: int = 32) -> List[List[Image.Image]]:
    """Load multiple media files and return list of image lists"""
    media_batch = []
    
    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"Warning: File not found: {input_path}")
            media_batch.append([])
            continue
            
        inp = input_path.lower()
        if inp.endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif')):
            imgs = load_video_frames(input_path,
                                   min_frames=num_frames,
                                   max_frames=num_frames,
                                   sampling='rand')
            if not imgs:
                print(f"Warning: Video load failed: {input_path}")
                media_batch.append([])
            else:
                media_batch.append(imgs)
        else:
            try:
                imgs = [load_image(input_path)]
                media_batch.append(imgs)
            except Exception as e:
                print(f"Warning: Image load failed: {input_path}, {e}")
                media_batch.append([])
    
    return media_batch


# ------------------------------------------------------------------
# 2. Visual preprocessing
# ------------------------------------------------------------------
def preprocess_images(images: List[Image.Image],
                      image_size=448,
                      dynamic_size=True,
                      use_thumbnail=True,
                      max_patches=4):
    """Return pixel_values for both encoders"""
    # --- Encoder‑1 ---
    transform = build_transform(is_train=False, input_size=image_size)
    all_proc = []
    for img in images:
        if dynamic_size:
            all_proc.extend(dynamic_preprocess(img, 1, max_patches,
                                               image_size, use_thumbnail))
        else:
            all_proc.append(img)
    # 这里一次性加入128张图片，然后进行处理，size [128, 3, 448, 448]
    pv1 = torch.stack([transform(i) for i in all_proc])          # (K,3,H,W)

    # --- Encoder‑2 (VGGT) ---
    # B2 C H W, B2和B1的维度相同，都是128，图像的H和W不同需要对齐
    pv2 = preprocess_images2(images, mode='pad')                 # (K2,3,H,W)
    pv2 = pv2.unsqueeze(0)                                       # (1,S,C,H,W)

    # 这里不能整除的话要做窗口的填充
    # --- VGGT window alignment ---
    B, S = pv2.shape[:2]
    WSIZE = 2
    if S % WSIZE:
        tgt = (S + WSIZE - 1) // WSIZE * WSIZE
        pad = tgt - S
        pv2 = torch.nn.functional.pad(pv2,
                                      (0, 0, 0, 0, 0, 0, 0, pad))  # pad seq dim
        print(f"[preprocess] pad frames {S}->{pv2.shape[1]}")

    # --- dtype & device ---
    # Use consistent dtype that works with both models
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device('cuda')
    
    pv1 = pv1.to(device=device, dtype=dtype)
    pv2 = pv2.to(device=device, dtype=dtype)
    
    return pv1, pv2


def preprocess_media_batch(media_batch: List[List[Image.Image]],
                          image_size: int = 448,
                          dynamic_size: bool = True,
                          use_thumbnail: bool = True,
                          max_patches: int = 4):
    """Preprocess batch of media for both encoders"""
    batch_pv1 = []
    batch_pv2 = []
    
    for imgs in media_batch:
        if not imgs:  # Empty image list
            batch_pv1.append(None)
            batch_pv2.append(None)
            continue
            
        pv1, pv2 = preprocess_images(imgs, image_size, dynamic_size, use_thumbnail, max_patches)
        batch_pv1.append(pv1)
        batch_pv2.append(pv2)
    
    return batch_pv1, batch_pv2


# ------------------------------------------------------------------
# 3. Load model & tokenizer
# ------------------------------------------------------------------
def load_model_and_tokenizer(ckpt):
    print("⇢ Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)

    # 特殊token
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

    # 拓展token标记
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
            
            # 学习率最开始1e-4，后面联调1e-5
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
# 4. Patch model: support dual encoder chat / generate
# ------------------------------------------------------------------
def patch_model_batch_method(model):
    """Inject batch chat & generate_with_dual_encoder methods"""

    import types

    # ---------- batch generate ----------
    # 这个是批量推理的代码
    def batch_generate_with_dual_encoder(self,
                                        batch_pixel_values=None,
                                        batch_pixel_values2=None,
                                        batch_attention_mask2=None,
                                        input_ids=None,
                                        attention_mask=None,
                                        generation_config=None,
                                        **gen_kwargs):
        """Batch dual encoder inference"""
        if isinstance(generation_config, dict) or generation_config is None:
            generation_config = GenerationConfig(**(generation_config or {}))

        batch_size = len(batch_pixel_values) if batch_pixel_values else input_ids.shape[0]
        use_compression = getattr(self, 'use_vision_compression', False) and hasattr(self, 'vision_compressor')

        # Process visual features for each sample
        batch_vit_all = []
        for i in range(batch_size):
            pixel_values = batch_pixel_values[i] if batch_pixel_values else None
            pixel_values2 = batch_pixel_values2[i] if batch_pixel_values2 else None
            attention_mask2 = batch_attention_mask2[i] if batch_attention_mask2 else None

            if pixel_values is not None:
                vit1 = self.extract_feature(pixel_values)  # [N1, T1, C]
                # 双流1的feature的维度
                print(f"vit1: {vit1.shape}")
                if pixel_values2 is not None:
                    # Fix dtype mismatch
                    model_dtype = next(self.vision_model2.parameters()).dtype
                    pixel_values2 = pixel_values2.to(model_dtype)

                    # Fix normalization buffers dtype
                    if hasattr(self.vision_model2, '_resnet_mean') and hasattr(self.vision_model2, '_resnet_std'):
                        if self.vision_model2._resnet_mean.dtype != model_dtype:
                            self.vision_model2._resnet_mean = self.vision_model2._resnet_mean.to(model_dtype)
                            self.vision_model2._resnet_std = self.vision_model2._resnet_std.to(model_dtype)

                    vit2 = self.extract_feature2(pixel_values2, attention_mask2)  # [N2, T2, C]
                    # [128, 50, 3584]  ->   [B, T2, C]
                    print(f"vit2: {vit2.shape}")
                    if use_compression:
                        # Try joint compression when frame counts align; otherwise compress separately and concat
                        if vit1.shape[0] == vit2.shape[0]:
                            combined = torch.cat([vit2, vit1], dim=1)  # [N, T2+T1, C]
                            # 四个一组压缩了一下  变为[32, 114, 3584]
                            comp = self.vision_compressor(combined)     # [G, Q, C]
                            vit_all = comp.reshape(-1, comp.size(-1))   # [G*Q, C]
                            print(f"vit_all: {vit_all.shape}")
                        else:
                            # comp_v2 = self.vision_compressor(vit2)      # [G2, Q, C]
                            # comp_v1 = self.vision_compressor(vit1)      # [G1, Q, C]
                            # vit_all = torch.cat([
                            #     comp_v2.reshape(-1, comp_v2.size(-1)),
                            #     comp_v1.reshape(-1, comp_v1.size(-1))
                            # ], dim=0)
                            raise ValueError("vit1.shape[0] != vit2.shape[0]")
                    else:
                        vit_all = torch.cat([
                            vit2.reshape(-1, vit2.size(-1)),
                            vit1.reshape(-1, vit1.size(-1))
                        ], dim=0)
                else:
                    if use_compression:
                        comp_v1 = self.vision_compressor(vit1)  # [G1, Q, C]
                        vit_all = comp_v1.reshape(-1, comp_v1.size(-1))
                    else:
                        vit_all = vit1.reshape(-1, vit1.size(-1))
            else:
                vit_all = None

            batch_vit_all.append(vit_all)

        # Process text embeddings
        txt = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = txt.shape

        # Replace image context tokens
        for i in range(B):
            if batch_vit_all[i] is not None:
                vit_all = batch_vit_all[i]
                ids_row = input_ids[i]
                sel = (ids_row == self.img_context_token_id)

                if sel.sum() != vit_all.size(0):
                    print(f"Warning: Visual tokens ({vit_all.size(0)}) != selected positions ({sel.sum()}) for sample {i}")
                    # Handle mismatch by truncating or padding
                    if vit_all.size(0) > sel.sum():
                        vit_all = vit_all[:sel.sum()]
                    elif vit_all.size(0) < sel.sum():
                        # Pad with repeat last token
                        pad_size = sel.sum() - vit_all.size(0)
                        pad_tokens = vit_all[-1:].repeat(pad_size, 1)
                        vit_all = torch.cat([vit_all, pad_tokens], dim=0)

                # 视觉特征替换context占据的文本特征
                txt[i][sel] = vit_all.to(txt.device)

        # Set generation config defaults
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = getattr(self.language_model.config, 'bos_token_id', 1)
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = getattr(self.language_model.config, 'eos_token_id', 2)
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = getattr(self.language_model.config, 'pad_token_id', 
                                                   generation_config.eos_token_id)

        # Generate
        outs = self.language_model.generate(
            inputs_embeds=txt,
            attention_mask=attention_mask,
            generation_config=generation_config,
            use_cache=True,
            **gen_kwargs,
        )
        return outs

    # ---------- batch chat ----------
    @torch.no_grad()
    # 推理无梯度回传
    def batch_chat_with_dual_encoder(self,
                                   tokenizer,
                                   batch_pixel_values,
                                   batch_questions,
                                   generation_config,
                                   batch_pixel_values2=None,
                                   batch_attention_mask2=None,
                                   IMG_START_TOKEN='<img>',
                                   IMG_END_TOKEN='</img>',
                                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                                   verbose=False):
        """Batch chat with dual encoder support"""
        batch_size = len(batch_questions)
        use_compression = getattr(self, 'use_vision_compression', False) and hasattr(self, 'vision_compressor')

        # Build prompts for each sample
        batch_prompts = []
        for i in range(batch_size):
            question = batch_questions[i]
            pixel_values = batch_pixel_values[i] if batch_pixel_values else None
            pixel_values2 = batch_pixel_values2[i] if batch_pixel_values2 else None
            attn2 = batch_attention_mask2[i] if batch_attention_mask2 is not None else None

            # Add <image> if not present
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question

            # Get conversation template
            tmpl = get_conv_template(self.template)
            tmpl.system_message = self.system_message
            tmpl.append_message(tmpl.roles[0], question)
            tmpl.append_message(tmpl.roles[1], None)
            prompt = tmpl.get_prompt()

            # Calculate tokens for image placeholders
            n_img2 = 7 * 7 + 1  # Fixed in training for encoder2

            total = 0
            if pixel_values is not None:
                n_patch = pixel_values.shape[0]
                n_patch2 = pixel_values2.shape[1] if pixel_values2 is not None else 0

                if use_compression:
                    # Estimate number of compressed groups and total tokens
                    comp_ratio = getattr(self, 'compression_ratio', 4)
                    q_tokens = getattr(self.vision_compressor, 'num_query', 64)
                    print(f"comp_ratio: {comp_ratio}, q_tokens: {q_tokens}")
                    if pixel_values2 is not None and attn2 is not None:
                        valid_frames = int(attn2.sum().item())
                        groups = (valid_frames + comp_ratio - 1) // comp_ratio
                        total = groups * q_tokens
                        print(f"total: {total}")
                    elif pixel_values2 is not None and n_patch == n_patch2:
                        groups = (n_patch + comp_ratio - 1) // comp_ratio
                        total = groups * q_tokens
                    elif pixel_values2 is not None and n_patch != n_patch2:
                        # Compress each stream separately and sum tokens
                        groups1 = (n_patch + comp_ratio - 1) // comp_ratio
                        groups2 = (n_patch2 + comp_ratio - 1) // comp_ratio
                        total = (groups1 + groups2) * q_tokens
                    else:
                        groups = (n_patch + comp_ratio - 1) // comp_ratio
                        total = groups * q_tokens

                else:
                    if pixel_values2 is not None and n_patch == n_patch2:
                        total = (self.num_image_token + n_img2) * n_patch
                    else:
                        total = self.num_image_token * n_patch + (n_img2 if pixel_values2 is not None else 0)

                if verbose:
                    print(f"[prompt {i}] tokens: total={total} (compression={'on' if use_compression else 'off'})")

                img_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total + IMG_END_TOKEN
                prompt = prompt.replace('<image>', img_tokens, 1)

            batch_prompts.append(prompt)

        # Tokenize with left padding
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        
        # Set pad_token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set padding side to left for generation
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
            
        # Tokenize batch with left padding
        model_inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=14096
        )
        
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        
        device = next(self.parameters()).device
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        
        # Generation config
        tmpl = get_conv_template(self.template)
        eos_id = tokenizer.convert_tokens_to_ids(tmpl.sep.strip())
        gen_cfg = GenerationConfig(**generation_config)
        gen_cfg.eos_token_id = eos_id
        
        # Generate
        outs = self.batch_generate_with_dual_encoder(
            batch_pixel_values=batch_pixel_values,
            batch_pixel_values2=batch_pixel_values2,
            batch_attention_mask2=batch_attention_mask2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )
        
        # Decode responses
        responses = []
        for i in range(batch_size):
            txt = tokenizer.decode(outs[i], skip_special_tokens=True)
            answer = txt.split(tmpl.sep.strip())[0].strip()
            responses.append(answer)
            
            if verbose:
                print(f"[Chat {i}] user: {batch_questions[i]}")
                print(f"[Chat {i}] bot: {answer}")
        
        return responses

    # Bind methods
    model.batch_generate_with_dual_encoder = types.MethodType(batch_generate_with_dual_encoder, model)
    model.batch_chat_with_dual_encoder = types.MethodType(batch_chat_with_dual_encoder, model)

    return model


# ------------------------------------------------------------------
# 5. Batch inference utilities
# ------------------------------------------------------------------
def batch_inference(model, tokenizer, media_batch, questions, max_tokens=512, batch_size=4):
    """Perform batch inference with chunking"""
    # 这里的带batch是通过转化得来的，标准数据格式，[S，C，H，W]
    batch_pv1, batch_pv2 = preprocess_media_batch(
        media_batch,
        image_size=model.config.force_image_size or model.config.vision_config.image_size,
        dynamic_size=True,
        use_thumbnail=model.config.use_thumbnail,
        max_patches=1  # For video, set to 1
    )
    
    all_results = []
    total_samples = len(questions)
    
    # Process in batches
    for i in tqdm(range(0, total_samples, batch_size), desc="Batch inference"):
        batch_end = min(i + batch_size, total_samples)
        
        # Get batch data
        batch_q = questions[i:batch_end]
        batch_p1 = batch_pv1[i:batch_end]
        batch_p2 = batch_pv2[i:batch_end]
        
        # Create attention masks
        batch_attn2 = []
        for pv2 in batch_p2:
            if pv2 is not None:
                attn2 = (pv2.abs().sum(dim=(2, 3, 4)) > 0)
                batch_attn2.append(attn2)
            else:
                batch_attn2.append(None)
        
        # Generation config
        gen_cfg = {
            'max_new_tokens': max_tokens,
            'do_sample': False,
            'num_beams': 1,
            'pad_token_id': tokenizer.eos_token_id,
            'bos_token_id': getattr(tokenizer, 'bos_token_id', tokenizer.eos_token_id),
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        # Inference
        try:
            answers = model.batch_chat_with_dual_encoder(
                tokenizer=tokenizer,
                batch_pixel_values=batch_p1,
                batch_pixel_values2=batch_p2,
                batch_attention_mask2=batch_attn2,
                batch_questions=batch_q,
                generation_config=gen_cfg,
                verbose=False
            )
            
            # Store results
            for j, answer in enumerate(answers):
                all_results.append({
                    'index': i + j,
                    'question': batch_q[j],
                    'answer': answer,
                    'input_path': None  # Will be filled later
                })
                
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {e}")
            # Add empty results for failed batch
            for j in range(len(batch_q)):
                all_results.append({
                    'index': i + j,
                    'question': batch_q[j],
                    'answer': f"Error: {str(e)}",
                    'input_path': None
                })
    
    return all_results


def parse_file_list(file_path: str) -> List[str]:
    """Parse file containing list of paths"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def main():
    parser = argparse.ArgumentParser(description='InternVL‑Chat Dual‑Encoder Batch Demo')
    parser.add_argument('--checkpoint', default='/mnt/chensenda/codes/VLN/InternVL_video/internvl_chat/work_dirs/internvl_chat_dual_compressor/internvl_chat_dual_compressor_8b_mix_s3_2/checkpoint-11600',
                        help='Path to dual‑encoder checkpoint')
    
    # Input options
    parser.add_argument('--input', type=str, help='Single input path')
    parser.add_argument('--inputs', nargs='+', help='Multiple input paths')
    parser.add_argument('--input-file', type=str, help='File containing input paths')
    
    # Question options
    parser.add_argument('--question', type=str, help='Single question')
    parser.add_argument('--questions', nargs='+', help='Multiple questions')
    parser.add_argument('--question-file', type=str, help='File containing questions')
    
    # Processing options
    parser.add_argument('--num-frames', type=int, default=128, help='Number of video frames')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max generation tokens')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    # Parse inputs
    if args.input:
        input_paths = [args.input]
    elif args.inputs:
        input_paths = args.inputs
    elif args.input_file:
        input_paths = parse_file_list(args.input_file)
    else:
        raise ValueError("Must provide --input, --inputs, or --input-file")
    # 输入视频序列对
    input_paths = ["/mnt/chensenda/codes/VLN/ScanQA/Scannet/mp4/scene0169_00.mp4"]

    # input_paths = ["/mnt/chensenda/codes/VLN/ScanQA/Scannet/mp4/scene0169_00.mp4","/mnt/chensenda/codes/VLN/ScanQA/Scannet/mp4/scene0496_00.mp4","/mnt/chensenda/codes/VLN/ScanQA/Scannet/mp4/scene0169_00.mp4","/mnt/chensenda/codes/VLN/ScanQA/Scannet/mp4/scene0496_00.mp4"]
    # Parse questions
    if args.question:
        questions = [args.question]
    elif args.questions:
        questions = args.questions
    elif args.question_file:
        questions = parse_file_list(args.question_file)
    else:
        raise ValueError("Must provide --question, --questions, or --question-file")
    # 问题序列对
    questions = ["I am facing a cabinet, while there are several chairs in a row on my right and another one behind me. What is on the 5 o'clock of the trash can that is far away on my right? Quick answer."]
    # questions = ["I am facing a cabinet, while there are several chairs in a row on my right and another one behind me. What is on the 5 o'clock of the trash can that is far away on my right? Quick answer.","I am sitting on a chair facing the table with the blackboard behind me and a chair on my left within reach. Which direction should I go if I want to exit the room? Answer in few words.","I am facing a cabinet, while there are several chairs in a row on my right and another one behind me. What is on the 5 o'clock of the trash can that is far away on my right? Quick answer.","I am sitting on a chair facing the table with the blackboard behind me and a chair on my left within reach. Which direction should I go if I want to exit the room? Answer in few words.",]
    
    # Expand inputs/questions to match
    if len(input_paths) == 1 and len(questions) > 1:
        # Single input, multiple questions
        input_paths = input_paths * len(questions)
    elif len(questions) == 1 and len(input_paths) > 1:
        # Multiple inputs, single question
        questions = questions * len(input_paths)
    elif len(input_paths) != len(questions):
        raise ValueError(f"Number of inputs ({len(input_paths)}) must match number of questions ({len(questions)})")

    print(f"Processing {len(input_paths)} input-question pairs")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    model = patch_model_batch_method(model)

    # Load media
    print("Loading media files...")
    media_batch = load_media_batch(input_paths, args.num_frames)

    # Run batch inference
    print("Running batch inference...")
    results = batch_inference(model, tokenizer, media_batch, questions, 
                            args.max_tokens, args.batch_size)

    # Add input paths to results
    for i, result in enumerate(results):
        result['input_path'] = input_paths[i]

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        print("\n" + "="*80)
        print("BATCH INFERENCE RESULTS")
        print("="*80)
        for result in results:
            print(f"\nInput: {result['input_path']}")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print("-" * 40)


if __name__ == '__main__':
    main()
