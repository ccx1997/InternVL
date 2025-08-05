# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
import torch
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .aggregator import Aggregator as VGGTEncoder

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


# refer perciever, qformer; causal?
class QFormerBlock(nn.Module):
    """QFormer block with self-attention, cross-attention and FFN"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        assert self.head_dim % 4 == 0
        
        # Self-attention
        # self.self_attn_ln = nn.LayerNorm(hidden_size)
        # self.self_query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.self_key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.self_value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.self_out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Cross-attention
        self.cross_attn_ln = nn.LayerNorm(hidden_size)
        self.cross_query_proj = nn.Linear(hidden_size, hidden_size//4, bias=False)
        self.cross_key_proj = nn.Linear(hidden_size, hidden_size//4, bias=False)
        self.cross_value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # FFN
        self.ffn_ln = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.dropout = dropout
        
        # Learnable scaling parameters to prevent gradient vanishing
        # self.self_attn_scale = nn.Parameter(torch.ones(hidden_size) * 0.1)
        self.cross_attn_scale = nn.Parameter(torch.ones(hidden_size) * 0.001)
        self.ffn_scale = nn.Parameter(torch.ones(hidden_size) * 0.001)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using the same pattern as line 372"""
        def _initialize_weights_helper(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.01, a=-1.0, b=1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        
        self.apply(_initialize_weights_helper)
    
    def forward(self, query, key_value):
        """
        Args:
            query: [B, query_len, hidden_size]
            key_value: [B, kv_len, hidden_size]
        """
        B, query_len, _ = query.shape
        # Self-attention
        # residual = query·
        # query = self.self_attn_ln(query)
        
        # # Self-attention computation
        # q = self.self_query_proj(query)
        # k = self.self_key_proj(query)
        # v = self.self_value_proj(query)
        
        # # Reshape for multi-head attention
        # q = q.view(B, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        # k = k.view(B, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        # v = v.view(B, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # # Scaled dot-product attention
        # attn_output = F.scaled_dot_product_attention(
        #     q, k, v,
        #     dropout_p=self.dropout if self.training else 0.0,
        #     is_causal=False
        # )
        
        # # Reshape back
        # attn_output = attn_output.transpose(1, 2).contiguous().view(B, query_len, self.hidden_size)
        # attn_output = self.self_out_proj(attn_output)
        
        # # Residual connection with learnable scaling
        # query = residual + self.self_attn_scale * attn_output
        
        # Cross-attention
        residual = query
        query = self.cross_attn_ln(query)
        
        B, kv_len, _ = key_value.shape
        
        # Cross-attention computation
        q = self.cross_query_proj(query)
        k = self.cross_key_proj(key_value)
        v = self.cross_value_proj(key_value)
        
        # Reshape for multi-head attention
        q = q.view(B, query_len, self.num_heads, self.head_dim//4).transpose(1, 2)
        k = k.view(B, kv_len, self.num_heads, self.head_dim//4).transpose(1, 2)
        v = v.view(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, query_len, self.hidden_size)
        attn_output = self.cross_out_proj(attn_output)
        
        # Residual connection with learnable scaling
        query = residual + self.cross_attn_scale * attn_output
        
        # FFN
        residual = query
        query = self.ffn_ln(query)
        ffn_output = self.ffn(query)
        
        # Residual connection with learnable scaling
        output = residual + self.ffn_scale * ffn_output
        return output


class VisionCompressor(nn.Module):
    """Vision compressor using QFormer architecture"""
    def __init__(self, hidden_size, compression_ratio=4, num_layers=6, num_heads=8, num_query=64):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.hidden_size = hidden_size
        self.num_query = num_query
        
        # Learnable query tokens for compression
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query, hidden_size)
        )
        
        # QFormer blocks
        self.qformer_blocks = nn.ModuleList([
            QFormerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        # Mark as uninitialized - will be initialized by parent model
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for VisionCompressor only, not recursively for all submodules"""
        # Initialize query tokens with safe normal distribution
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Initialize QFormer blocks using the same pattern as line 372
        def _initialize_weights_helper(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.01, a=-1.0, b=1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        
        for block in self.qformer_blocks:
            block.apply(_initialize_weights_helper)
    
    def forward(self, embeddings, is_video=False):
        """
        Compress vision embeddings using QFormer with static computation graph
        
        Args:
            embeddings: [B, seq_len, hidden_size] or [total_frames, seq_len, hidden_size]
            is_video: bool, whether input is video frames (deprecated - all inputs treated as video)
            
        Returns:
            compressed_embeddings: [padded_groups, seq_len, hidden_size]
        """ 
        B, seq_len, hidden_size = embeddings.shape
        
        actual_groups = (B + self.compression_ratio - 1) // self.compression_ratio
        padded_frames = actual_groups * self.compression_ratio
        
        padding_needed = padded_frames - B
        if padding_needed > 0:
            last_frame = embeddings[-1:].expand(padding_needed, -1, -1)
            padded_embeddings = torch.cat([embeddings, last_frame], dim=0)
        else:
            padded_embeddings = embeddings
            
        group_embeddings = padded_embeddings.view(actual_groups, self.compression_ratio, seq_len, hidden_size)
        
        group_embeddings_flat = group_embeddings.view(actual_groups, self.compression_ratio * seq_len, hidden_size)
        
        # Add query tokens to the first image embeddings
        compressed_tokens = self.query_tokens.expand(actual_groups, -1, -1)  # [actual_groups, num_query, hidden_size]
        
        # Add first embedding from each group to query tokens
        first_embeddings = group_embeddings[:, 0, :, :]  # [actual_groups, seq_len, hidden_size]
        compressed_tokens = compressed_tokens + first_embeddings
        
        for block in self.qformer_blocks:
            compressed_tokens = block(compressed_tokens, group_embeddings_flat)
        
        return compressed_tokens


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'DinoVisionTransformer', 'VGGTBlock', 'QFormerBlock']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2)) // 4
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        
        # Compression parameters
        self.use_vision_compression = getattr(config, 'use_vision_compression', True)
        self.compression_ratio = getattr(config, 'compression_ratio', 4)
        
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'use_vision_compression: {self.use_vision_compression}')
        logger.info(f'compression_ratio: {self.compression_ratio}')
        
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        # another image encoder
        embed_dim2=1024
        self.vision_model2 = VGGTEncoder(embed_dim=embed_dim2, img_size=518, patch_size=14)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.vision_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        self.downsample2 = nn.Conv2d(embed_dim2 * 2, embed_dim2 * 4, kernel_size=3, stride=3, padding=1)
        self.mlp2_patch = nn.Sequential(
            nn.LayerNorm(embed_dim2 * 4),
            nn.Linear(embed_dim2 * 4, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        self.vision_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.mlp2_camera = nn.Sequential(
            nn.LayerNorm(embed_dim2 * 2),
            nn.Linear(embed_dim2 * 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # Vision compression modules
        if self.use_vision_compression:
            self.vision_compressor = VisionCompressor(
                hidden_size=llm_hidden_size,
                compression_ratio=self.compression_ratio,
                num_layers=1,
                num_heads=8,
                num_query=114  # 64 + 50
            )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self._initialize_vision_projector()

    def _compress_image_flags(self, image_flags):
        """
        Compress image flags according to compression ratio.
        If image_flags is tensor([0]), return tensor([0]).
        If image_flags length > 1, compress by grouping and taking max of each group.
        """
        if image_flags.shape[0] == 1:
            # If single flag, return as is
            return image_flags
        
        # Pad the tensor to make it divisible by compression_ratio
        total_length = image_flags.shape[0]
        padded_length = ((total_length + self.compression_ratio - 1) // self.compression_ratio) * self.compression_ratio
        
        if padded_length > total_length:
            # Pad with the last value
            padding = image_flags[-1:].expand(padded_length - total_length)
            padded_flags = torch.cat([image_flags, padding], dim=0)
        else:
            padded_flags = image_flags
        
        # Reshape to groups and take max of each group
        grouped_flags = padded_flags.view(-1, self.compression_ratio)
        compressed_flags = torch.max(grouped_flags, dim=1)[0]
        
        return compressed_flags

    def _initialize_vision_projector(self):
        def _initialize_weights(m):
            # if isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, std=1 / math.sqrt(m.weight.size(1)))
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Conv2d):
            #     fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            #     std = 1 / math.sqrt(fan_in)
            #     nn.init.normal_(m.weight, mean=0.0, std=std)
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.01, a=-1.0, b=1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.downsample2.apply(_initialize_weights)
        self.mlp2_patch.apply(_initialize_weights)
        self.mlp2_camera.apply(_initialize_weights)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values2: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask2: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        image_flags2: Optional[torch.LongTensor] = None,
        num_tiles: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # print("-----------------begin forward-----------------")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # image_flags = image_flags.squeeze(-1)
        # image_flags2 = image_flags2.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        vit_embeds = self.extract_feature(pixel_values)
        # print("")
        # print(f"pixel_values.shape: {pixel_values.shape}")
        # print(f"pixel_values2.shape: {pixel_values2.shape}")
        # print(f"vit_embeds.shape: {vit_embeds.shape}")
        if not self.use_vision_compression:
            vit_embeds = vit_embeds[image_flags == 1]
        # print(f"vit_embeds_after.shape: {vit_embeds.shape}")
        # print(f"image_flags: {image_flags}")
        # print(f"image_flags2: {image_flags2}")
        # print(f"num_tiles:{num_tiles}")
        # video muti frames
        # pixel_values.shape: torch.Size([14, 3, 448, 448]
        # pixel_values2.shape: torch.Size([1, 14, 3, 518, 518])
        # vit_embeds.shape: torch.Size([14, 64, 3584])
        # img_embeds2.shape: torch.Size([14, 50, 3584])
        # video one frame
        # pixel_values.shape: torch.Size([1, 3, 448, 448])
        # pixel_values2.shape: torch.Size([1, 2, 3, 518, 518])
        # vit_embeds.shape: torch.Size([1, 64, 3584])
        # img_embeds2.shape: torch.Size([1, 50, 3584])

        # import ipdb;ipdb.set_trace()
        # for encoder2
        if pixel_values2 is not None:
            img_embeds2 = self.extract_feature2(pixel_values2, attention_mask2)
            # print(f"img_embeds2.shape: {img_embeds2.shape}")
            
            # print(f"img_embeds2_after.shape: {img_embeds2.shape}")
            
            # First concatenate the 2 embeddings, then compress
            # print(f"num_tiles: {num_tiles}")
            if num_tiles is None or isinstance(num_tiles, list) and all(tb is None for tb in num_tiles):
                # print("num_tiles is None or isinstance(num_tiles, list) and all(tb is None for tb in num_tiles)")
                # Concatenate embeddings first: [N, 50, C] + [N, 64, C] -> [N, 114, C]
                combined_embeds = torch.cat([img_embeds2, vit_embeds], dim=1)
                # print(f"combined_embeds.shape: {combined_embeds.shape}")
                
                if self.use_vision_compression:
                    # Compress the combined embeddings using single compressor
                    combined_embeds_compressed = self.vision_compressor(combined_embeds)
                    # print(f"combined_embeds_compressed.shape: {combined_embeds_compressed.shape}")
                    combined_embeds = combined_embeds_compressed
                    
                    # Compress image flags to match compressed embeddings
                    image_flags = self._compress_image_flags(image_flags)
                    image_flags2 = self._compress_image_flags(image_flags2)
                    # print(f"compressed image_flags: {image_flags}")
                    # print(f"compressed image_flags2: {image_flags2}")
                    
                    # Filter based on image_flags (both should be the same for combined processing)
                    combined_embeds = combined_embeds[image_flags == 1]
                
                vit_embeds = combined_embeds.reshape(-1, C)
                # print(f"vit_embeds_after.shape: {vit_embeds.shape}")
            else:
                # Handle tile-based processing
                # First, collect all tiles with concatenated embeddings
                all_combined_embeds = []
                idx_v1, idx_v2 = 0, 0
                for tb in num_tiles:
                    if tb is None:
                        continue
                    for nt in tb:
                        # Check if nt is 1, warn if not since current implementation only supports nt=1
                        if nt != 1:
                            import warnings
                            warnings.warn(f"Current implementation only supports num_tiles value of 1, but got {nt}. "
                                        f"This may cause unexpected behavior.", UserWarning)
                        
                        # Concatenate embeddings for each tile: [1, 50, C] + [nt, 64, C] -> [1, 50+64*nt, C]
                        tile_img_embeds2 = img_embeds2[idx_v2:idx_v2+1]  # [1, 50, C]
                        tile_vit_embeds = vit_embeds[idx_v1:idx_v1+nt]   # [nt, 64, C]
                        combined_tile = torch.cat([tile_img_embeds2, tile_vit_embeds], dim=1)  # [1, 50+64*nt, C]
                        all_combined_embeds.append(combined_tile)
                        idx_v1 += nt
                        idx_v2 += 1
                
                if len(all_combined_embeds) > 0:
                    # Stack all tiles for compression: [num_tiles, 50+64*nt, C]
                    combined_embeds = torch.cat(all_combined_embeds, dim=0)
                    
                    if self.use_vision_compression:
                        # Compress the combined embeddings
                        combined_embeds_compressed = self.vision_compressor(combined_embeds)
                        # print("compress video")
                        combined_embeds = combined_embeds_compressed
                        
                        # Compress image flags to match compressed embeddings
                        image_flags = self._compress_image_flags(image_flags)
                        image_flags2 = self._compress_image_flags(image_flags2)
                        
                        # Filter based on image_flags
                        combined_embeds = combined_embeds[image_flags == 1]
                        
                        # Update num_tiles to reflect compressed structure
                        if num_tiles is not None and isinstance(num_tiles, list):
                            updated_num_tiles = []
                            for batch_tiles in num_tiles:
                                if batch_tiles is None:
                                    updated_num_tiles.append(None)
                                else:
                                    # Treat all cases as video: compress tile count according to compression ratio
                                    total_tiles = sum(batch_tiles) if isinstance(batch_tiles, list) else batch_tiles
                                    if isinstance(total_tiles, int) and total_tiles > 0:
                                        # Calculate compressed groups
                                        compressed_groups = (total_tiles + self.compression_ratio - 1) // self.compression_ratio
                                        updated_num_tiles.append([1] * compressed_groups)  # Each group becomes 1 tile
                                    else:
                                        updated_num_tiles.append([1])  # Fallback to single tile
                            num_tiles = updated_num_tiles
                    
                    # Reshape for final output
                    vit_embeds = combined_embeds.reshape(-1, C)
                else:
                    # Fallback case when no tiles
                    vit_embeds = torch.empty(0, C, device=img_embeds2.device)

        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            # print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        del vit_embeds
        if pixel_values2 is not None:
            del img_embeds2
            
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # print("-----------------end forward-----------------")
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        h1 = w1 = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h1, w1, -1).permute(0, 3, 1, 2)
        vit_embeds = self.vision_pool(vit_embeds)
        vit_embeds = vit_embeds.permute(0, 2, 3, 1)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds
    
    def extract_feature2(self, pixel_values2, attention_mask2):
        # extract features by a parallel encoder
        self.vision_model2.eval()
        img_embeds_list, patch_start_idx = self.vision_model2(pixel_values2, attention_mask=attention_mask2)
        camera_pose_features = img_embeds_list[-1][:, :, 0:1]  # [B, S, 1, C]
        img_embeds2 = img_embeds_list[-1][:, :, patch_start_idx:]  # [B, S, P, C]
        del img_embeds_list
        # filter padding frame features
        B, S, P, C = img_embeds2.shape
        camera_pose_features = camera_pose_features.reshape(B * S, 1, C)[
            attention_mask2.flatten()]  # [N, 1, C], N is the number of valid frames
        img_embeds2 = img_embeds2.reshape(B * S, P, C)[attention_mask2.flatten()]  # [N, P, C]
        # downsample
        side = int(P ** 0.5)
        img_embeds2 = img_embeds2.reshape(-1, side, side, C).permute(0, 3, 1, 2) # [N, C, side, side]
        img_embeds2 = self.downsample2(img_embeds2)
        # import ipdb; ipdb.set_trace()
        img_embeds2 = img_embeds2.flatten(2).permute(0, 2, 1) # [N, T, C*3], T=side*side//9
        # project
        img_embeds2 = self.mlp2_patch(img_embeds2)
        camera_pose_features = self.mlp2_camera(camera_pose_features)
        # pool
        N, T, D = img_embeds2.shape
        w_or_h = int(T ** 0.5)
        img_embeds2 = img_embeds2.permute(0, 2, 1).reshape(N, D, w_or_h, w_or_h)
        img_embeds2 = torch.cat([img_embeds2, img_embeds2[:, :, -1:, :]], dim=2)
        img_embeds2 = torch.cat([img_embeds2, img_embeds2[:, :, :, -1:]], dim=-1)
        img_embeds2 = self.vision_pool2(img_embeds2)
        img_embeds2 = img_embeds2.permute(0, 2, 3, 1).reshape(N, -1, D)
        # concat
        img_embeds2 = torch.cat([camera_pose_features, img_embeds2], dim=1) # [N, 1+T, E]
        return img_embeds2

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device(self.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
