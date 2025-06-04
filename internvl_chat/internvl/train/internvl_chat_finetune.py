# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import math
import os
import random
import re
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional

import numpy as np
import cv2
import imageio
from decord import VideoReader

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_phi3_attention_class,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, ENV_CONTEXT_TOKEN,
                                      ENV_START_TOKEN, ENV_END_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, preprocess_mpt,
                                    preprocess_phi3)
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

# Import video encoder related modules
from internvl.model.videochat_flash.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, load_video
from internvl.model.videochat_flash.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from internvl.model.videochat_flash.conversation import conv_templates, SeparatorStyle

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ['rand', 'middle']: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif 'fps' in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def extract_frame_number(filename):
    # Extract the numeric part from the filename using regular expressions
    match = re.search(r'_(\d+).jpg$', filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    # Extract filenames from each path and sort by their numeric part
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    freeze_video_encoder: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the video encoder. Default is False.'},
    )
    train_llm_embed_only: bool = field(
        default=False,
        metadata={'help': 'Set to True to train only LLM embedding layers (input embedding and lm_head). Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=30,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    min_video_encoder_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video encoder. Default is 4.'},
    )
    max_video_encoder_frame: int = field(
        default=48,
        metadata={'help': 'The maximum number of frames for video encoder. Default is 16.'},
    )
    max_video_encoder_length: int = field(
        default=8192,
        metadata={'help': 'The maximum input length for video encoder. Default is 8192.'},
    )
    video_encoder_user_prompt: str = field(
        default="Compress the video into new or updated latent features!",
        metadata={'help': 'The user prompt for video encoder. Default is "Compress them into latent features!".'},
    )
    video_encoder_tokenizer_path: str = field(
        default="/mnt/models/VideoChat-Flash-Qwen2_5-2B_res448",
        metadata={'help': 'The tokenizer path for video encoder. Default is "/mnt/models/VideoChat-Flash-Qwen2_5-2B_res448".'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
        # hyperparameters for video encoder
        min_video_encoder_frame=4,
        max_video_encoder_frame=16,
        max_video_encoder_length=2048,
        video_encoder_user_prompt="Compress them into latent features!",
        video_encoder_tokenizer_path="/mnt/models/VideoChat-Flash-Qwen2_5-2B_res448",
        video_encoder_image_processor=None,  # video encoder image processor for processing frames
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

        # hyperparameters for video encoder
        self.min_video_encoder_frame = min_video_encoder_frame
        self.max_video_encoder_frame = max_video_encoder_frame
        self.max_video_encoder_length = max_video_encoder_length
        self.video_encoder_user_prompt = video_encoder_user_prompt
        self.video_encoder_tokenizer_path = video_encoder_tokenizer_path
        
        # Load video encoder tokenizer
        try:
            from transformers import AutoTokenizer
            self.video_encoder_tokenizer = AutoTokenizer.from_pretrained(
                video_encoder_tokenizer_path, trust_remote_code=True
            )
            logger.info(f'Loaded video encoder tokenizer from: {video_encoder_tokenizer_path}')
        except Exception as e:
            logger.warning(f'Failed to load video encoder tokenizer: {e}. Using main tokenizer as fallback.')
            self.video_encoder_tokenizer = tokenizer

        self.video_encoder_image_processor = video_encoder_image_processor

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def load_video(self, video_path, clip=None):
        if video_path.endswith('/'):  # Video is a folder of images
            image_list = sort_frames(os.listdir(video_path))
            frames_pil = []
            for image_name in image_list:
                fp = os.path.join(video_path, image_name)
                try:
                    frame = Image.open(fp).convert('RGB')
                    frames_pil.append(frame)
                except UnidentifiedImageError:
                    logger.warning(f"Skipping unidentified image: {fp}")
                    continue
            vlen = len(frames_pil)
            # Randomly select number of frames between min_num_frame and max_num_frame
            # self.max_num_frame and self.min_num_frame are attributes of LazySupervisedDataset
            t_num_frames = np.random.randint(self.min_num_frame, self.max_num_frame + 1)

            if vlen > t_num_frames:
                # self.sampling_method is an attribute of LazySupervisedDataset
                frame_indices = get_frame_indices(
                    t_num_frames, vlen, sample=self.sampling_method
                )
                selected_frames = [frames_pil[i] for i in frame_indices]
            else:
                selected_frames = frames_pil # Use all frames if less than t_num_frames
            return selected_frames

        elif video_path.endswith('.gif'): # Video is a GIF
            try:
                gif = imageio.get_reader(video_path)
                vlen = len(gif)
            except Exception as e:
                logger.error(f"Error loading GIF {video_path}: {e}")
                return [] # Return empty list on error

            t_num_frames = np.random.randint(self.min_num_frame, self.max_num_frame + 1)
            frame_indices = get_frame_indices(
                t_num_frames, vlen, sample=self.sampling_method
            )
            frames_pil = []
            try:
                for index, frame_data in enumerate(gif):
                    if index in frame_indices:
                        # Convert RGBA to RGB if necessary, handle other conversions
                        if frame_data.ndim == 3 and frame_data.shape[2] == 4:
                            frame = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2RGB)
                        elif frame_data.ndim == 2: # Grayscale to RGB
                            frame = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
                        else:
                            frame = frame_data
                        frame_pil = Image.fromarray(frame.astype(np.uint8))
                        frames_pil.append(frame_pil)
            except Exception as e:
                logger.error(f"Error processing GIF frames for {video_path}: {e}")
                return []
            return frames_pil

        else:  # Video is a standard video file (mp4, avi, etc.)
            try:
                video_reader = VideoReader(video_path, num_threads=1)
                vlen = len(video_reader)
                fps = video_reader.get_avg_fps()
                duration = vlen / float(fps)

                if clip:
                    start, end = clip
                    # Ensure start and end are within video duration
                    start = max(0, min(start, duration))
                    end = max(start, min(end, duration))
                    duration = end - start
                    vlen_clip = int(duration * fps)
                    start_index = int(start * fps)
                else:
                    vlen_clip = vlen
                    start_index = 0

                t_num_frames = np.random.randint(self.min_num_frame, self.max_num_frame + 1)
                # Ensure t_num_frames is not greater than available frames in the clip
                t_num_frames = min(t_num_frames, vlen_clip)
                if t_num_frames <= 0 and vlen_clip > 0 : # if min_num_frame makes t_num_frames too small or 0
                    t_num_frames = vlen_clip # use all frames in clip
                elif vlen_clip == 0:
                     logger.warning(f"Video clip has 0 frames: {video_path}, clip: {clip}")
                     return []


                frame_indices = get_frame_indices(
                    t_num_frames, vlen_clip, sample=self.sampling_method,
                    input_fps=fps, max_num_frames=self.max_num_frame # Pass max_num_frame from dataset args
                )
                actual_frame_indices = [f + start_index for f in frame_indices]
                # Ensure indices are within the bounds of the original video
                actual_frame_indices = [idx for idx in actual_frame_indices if idx < vlen]
                
                if not actual_frame_indices: # if all indices are out of bounds
                    logger.warning(f"No valid frames selected for {video_path}, clip: {clip}, indices: {frame_indices}, start_idx: {start_index}")
                    return []

                frames_np = video_reader.get_batch(actual_frame_indices).asnumpy()  # (T, H, W, C), np.uint8
                frames_pil = [Image.fromarray(frames_np[i]) for i in range(frames_np.shape[0])]
                return frames_pil
            except RuntimeError as e:
                logger.error(f"Decord runtime error for {video_path}: {e}")
                return []
            except Exception as e:
                logger.error(f"Error loading video {video_path}: {e}")
                return []

    def prepare_env_video_params(self, env_video_path):
        """
        Prepare video encoder parameters for env_video following VideoChatFlashQwenForCausalLM.chat() logic
        
        Args:
            env_video_path: Path to the environment video file
            
        Returns:
            video_enc_params: Dictionary containing parameters for VideoChatFlashQwenForCausalLM.forward
        """
        # Load video frames using VideoChat Flash's load_video function directly
        media_dict = {'video_read_type': 'decord'}
        frames, time_msg = load_video(env_video_path, max_num_frames=self.max_video_encoder_frame, media_dict=media_dict)
        
        if frames is None or len(frames) == 0:
            return None
            
        # Convert numpy frames to PIL Images for image_processor compatibility
        if isinstance(frames, np.ndarray):
            # frames is a numpy array of shape (T, H, W, C)
            frames_pil = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
        else:
            frames_pil = frames
            
        # Get image sizes (following chat method logic)
        image_sizes = [frames_pil[0].size[::-1]]  # PIL size is (width, height), convert to (height, width)
        
        # Prepare user prompt for video encoder
        user_prompt = self.video_encoder_user_prompt
        
        # Follow chat() method logic
        conv = conv_templates["qwen_2"].copy()
        
        # Add DEFAULT_IMAGE_TOKEN and time_msg to user_prompt following chat() method
        formatted_user_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {user_prompt}'
        
        conv.append_message(conv.roles[0], formatted_user_prompt)
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        
        # Use tokenizer_image_token as in chat() method
        input_ids = tokenizer_image_token(prompt, self.video_encoder_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        
        # Set pad_token_id if needed (following chat() method)
        if self.video_encoder_tokenizer.pad_token_id is None:
            if "qwen" in self.video_encoder_tokenizer.name_or_path.lower():
                self.video_encoder_tokenizer.pad_token_id = 151643
        
        # Create attention mask following chat() method
        attention_mask = input_ids.ne(self.video_encoder_tokenizer.pad_token_id).long()
        
        # Convert frames to the format expected by video encoder
        # Process frames using video_encoder's image_processor if available
        if self.video_encoder_image_processor is not None and hasattr(self.video_encoder_image_processor, 'preprocess'):
            try:
                # Process frames similar to chat() method
                processed_frames = self.video_encoder_image_processor.preprocess(
                    frames_pil, return_tensors="pt")["pixel_values"]
                # Convert to list format as expected by chat() method
                processed_frames = [processed_frames]
                frames = processed_frames
            except Exception as e:
                logger.warning(f"Failed to preprocess frames with video_encoder: {e}. Using raw frames.")
                frames = frames_pil
        else:
            frames = frames_pil
        
        # Prepare parameters following chat() method's generate call format
        video_enc_params = {
            'input_ids': input_ids,  # corresponds to inputs parameter
            'images': frames,  # corresponds to images parameter  
            'attention_mask': attention_mask,  # corresponds to attention_mask parameter
            'modalities': ["video"],  # corresponds to modalities parameter
            'image_sizes': image_sizes,  # corresponds to image_sizes parameter
            'max_num_frames': len(frames),
            'max_video_encoder_length': self.max_video_encoder_length,
            'user_prompt': user_prompt,  # keep original text for reference
        }
        
        return video_enc_params

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        if self.tcs_loader is not None:
            image_list = self.tcs_loader(
                video_path,
                image_type='video',
                max_num_frames=self.max_num_frame,
                min_num_frames=self.min_num_frame,
                sample=self.sampling_method,
                clip=data_item.get('clip', None))
        else:
            image_list = self.load_video(video_path)

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
        )
        return ret

    def get_env_context_count(self, video_enc_params):
        """
        Determine the number of ENV_CONTEXT tokens needed based on video encoder configuration.
        This should match the latent_len used in the video encoder model.
        """
        if video_enc_params is None:
            return 0
        # Use the configured video encoder latent length
        # This should match the video_encoder_latent_len in the model config (default 1024)
        return 1024  # Fixed number of context tokens to match video encoder output

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                
                # Check if env_video exists and prepare video encoder params
                video_enc_params = None
                if 'env_video' in data_item and data_item['env_video'] is not None and data_item['env_video'] != '':
                    env_video_path = os.path.join(self.root, data_item['env_video'])
                    video_enc_params = self.prepare_env_video_params(env_video_path)
                    
                    # Add environment context tokens to the first user message if env_video exists
                    if video_enc_params is not None and len(data_item['conversations']) > 0:
                        # Find the first human message
                        for conv_item in data_item['conversations']:
                            if conv_item['from'] == 'human':
                                # Get the number of ENV_CONTEXT tokens needed
                                env_context_count = self.get_env_context_count(video_enc_params)
                                # Add environment tokens before the user question
                                env_tokens = f"{ENV_START_TOKEN}{ENV_CONTEXT_TOKEN * env_context_count}{ENV_END_TOKEN}"
                                conv_item['value'] = env_tokens + '\n' + conv_item['value']
                                break
                
                # Process the data item based on its type (unchanged logic)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                
                # Add video encoder params to return dict
                ret['video_enc_params'] = video_enc_params
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
    video_encoder_image_processor=None,
):
    datasets = []
    lengths = []
    # Safely get distributed parameters
    if dist.is_available() and dist.is_initialized():
        data_rank = dist.get_rank()
        data_world_size = dist.get_world_size()
    else:
        data_rank = 0
        data_world_size = 1
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
            # hyperparameters for video encoder
            min_video_encoder_frame=data_args.min_video_encoder_frame,
            max_video_encoder_frame=data_args.max_video_encoder_frame,
            max_video_encoder_length=data_args.max_video_encoder_length,
            video_encoder_user_prompt=data_args.video_encoder_user_prompt,
            video_encoder_tokenizer_path=data_args.video_encoder_tokenizer_path,
            video_encoder_image_processor=video_encoder_image_processor,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def main():
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'pytorch')
    # Only initialize distributed when using multiple GPUs or distributed environment variables are set
    if torch.cuda.device_count() > 1 or 'RANK' in os.environ or 'LOCAL_RANK' in os.environ:
        init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    
    # Add environment context tokens for video encoder
    env_ctx_token_list = [ENV_CONTEXT_TOKEN, ENV_START_TOKEN, ENV_END_TOKEN]
    num_env_tokens = tokenizer.add_tokens(env_ctx_token_list, special_tokens=True)
    logger.info(f'new num_env_tokens: {num_env_tokens}')
    env_ctx_token_id = tokenizer.convert_tokens_to_ids(ENV_CONTEXT_TOKEN)
    env_start_token_id = tokenizer.convert_tokens_to_ids(ENV_START_TOKEN)
    env_end_token_id = tokenizer.convert_tokens_to_ids(ENV_END_TOKEN)
    num_new_tokens += num_env_tokens
    
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        from internvl.patch import apply_liger_kernel_to_internvit
        from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                               apply_liger_kernel_to_qwen2)
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info(f'Using flash_attention_2 for {config.llm_config.model_type}')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        llm = model_type.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id
    model.env_ctx_token_id = env_ctx_token_id
    model.env_start_token_id = env_start_token_id
    model.env_end_token_id = env_end_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    if hasattr(model, 'video_encoder') and model.video_encoder is not None:
        video_encoder_image_processor = model.video_encoder.get_vision_tower().image_processor
        
        # Apply VideoChatFlash optimizations only when video encoder is not frozen
        from internvl.model.videochat_flash.modeling_videochat_flash import VideoChatFlashQwenForCausalLM
        
        if isinstance(model.video_encoder, VideoChatFlashQwenForCausalLM) and not model_args.freeze_video_encoder:
            logger.info('Applying training optimizations for VideoChatFlashQwenForCausalLM...')
            
            # Enable training optimizations
            try:
                # Try the comprehensive optimization method first
                model.video_encoder.enable_training_optimizations()
                logger.info('✅ VideoChatFlash training optimizations enabled successfully')
                
                # Verify optimizations were applied
                gc_enabled = getattr(model.video_encoder.model, 'gradient_checkpointing', False)
                attn_impl = getattr(model.video_encoder.config, '_attn_implementation', 'unknown')
                logger.info(f'📊 Optimization Status:')
                logger.info(f'  - Gradient Checkpointing: {"✅" if gc_enabled else "❌"}')
                logger.info(f'  - Attention Implementation: {attn_impl}')
                logger.info(f'  - Training Mode: {"✅" if model.video_encoder.training else "❌"}')
                
            except Exception as e:
                logger.warning(f'⚠️ Failed to enable comprehensive training optimizations: {e}')
                logger.warning('Attempting manual optimization fallback...')
                
                # Fallback to manual optimization with better error handling
                optimization_success = []
                
                # Try Flash Attention 2
                try:
                    model.video_encoder.enable_flash_attention_2()
                    optimization_success.append('Flash Attention 2')
                    logger.info('✅ Flash Attention 2 enabled')
                except Exception as e2:
                    logger.warning(f'❌ Flash Attention 2 failed: {e2}')
                
                # Try Gradient Checkpointing with multiple approaches
                try:
                    model.video_encoder.enable_gradient_checkpointing()
                    optimization_success.append('Gradient Checkpointing')
                    logger.info('✅ Gradient Checkpointing enabled')
                except Exception as e3:
                    logger.warning(f'❌ Gradient Checkpointing failed: {e3}')
                    # Try direct attribute setting as last resort
                    try:
                        if hasattr(model.video_encoder, 'model'):
                            model.video_encoder.model.gradient_checkpointing = True
                        if hasattr(model.video_encoder, 'gradient_checkpointing'):
                            model.video_encoder.gradient_checkpointing = True
                        optimization_success.append('Basic Gradient Checkpointing')
                        logger.info('✅ Basic Gradient Checkpointing enabled as fallback')
                    except Exception as e4:
                        logger.error(f'❌ All gradient checkpointing methods failed: {e4}')
                
                # Set training mode
                try:
                    model.video_encoder.train()
                    optimization_success.append('Training Mode')
                    logger.info('✅ Training mode enabled')
                except Exception as e5:
                    logger.warning(f'❌ Failed to set training mode: {e5}')
                
                if optimization_success:
                    logger.info(f'✅ Manual optimizations applied: {", ".join(optimization_success)}')
                else:
                    logger.error('❌ All optimization attempts failed')
            
        elif isinstance(model.video_encoder, VideoChatFlashQwenForCausalLM) and model_args.freeze_video_encoder:
            logger.info('🔒 VideoChatFlash is frozen, skipping training optimizations')
        else:
            logger.info(f'ℹ️ Video encoder type: {type(model.video_encoder).__name__}')
            logger.info('ℹ️ No VideoChatFlash optimizations applied')
    else:
        video_encoder_image_processor = None
        logger.info('No video encoder found in the model')
    
    train_dataset = build_datasets(
        data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type, min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
        video_encoder_image_processor=video_encoder_image_processor)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False
    
    def _unfreeze_params(module):
        for param in module.parameters():
            param.requires_grad = True

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.train_llm_embed_only:
        # First freeze all LLM parameters
        _freeze_params(model.language_model)
        logger.info('All LLM parameters frozen for embed-only training')
        
        # Then unfreeze only embedding layers
        # Unfreeze input embedding layer
        if hasattr(model.language_model, 'embed_tokens'):
            _unfreeze_params(model.language_model.embed_tokens)
            logger.info('LLM input embedding layer (embed_tokens) unfrozen')
        elif hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'embed_tokens'):
            _unfreeze_params(model.language_model.model.embed_tokens)
            logger.info('LLM input embedding layer (model.embed_tokens) unfrozen')
        else:
            logger.warning('Could not find input embedding layer to unfreeze')
        
        # Unfreeze output layer (lm_head)
        if hasattr(model.language_model, 'lm_head'):
            _unfreeze_params(model.language_model.lm_head)
            logger.info('LLM output layer (lm_head) unfrozen')
        else:
            logger.warning('Could not find lm_head to unfreeze')

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if hasattr(model, 'video_encoder') and model.video_encoder is not None:
        if model_args.freeze_video_encoder:
            _freeze_params(model.video_encoder)
            logger.info('Video encoder parameters frozen')
        else:
            _unfreeze_params(model.video_encoder)
            logger.info('Video encoder parameters unfrozen')
    else:
        logger.warning('Video encoder not found or not initialized, skipping freeze_video_encoder')

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    # Check if this is the main process safely
    is_main_process = True
    if dist.is_available() and dist.is_initialized():
        is_main_process = dist.get_rank() == 0
    
    if is_main_process:
        params_requires_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_requires_grad.append(name)
        logger.info(f'Trainable parameters: {params_requires_grad}')

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Save model parameter information to params.txt
    params_file_path = os.path.join(training_args.output_dir, 'params.txt')
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    with open(params_file_path, 'w', encoding='utf-8') as f:
        f.write("Parameter_Name\tTensor_Size\tRequires_Grad\n")  # Header
        for name, param in model.named_parameters():
            tensor_size = list(param.shape)
            requires_grad = str(param.requires_grad)
            f.write(f"{name}\t{tensor_size}\t{requires_grad}\n")
    
    logger.info(f'Model parameter information saved to: {params_file_path}')
    
    # GPU memory monitoring and final optimization summary
    if is_main_process and torch.cuda.is_available():
        try:
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            gpu_allocated_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
            gpu_cached_mb = torch.cuda.memory_reserved(0) / 1024 / 1024
            
            logger.info('🖥️  GPU Memory Status:')
            logger.info(f'  - Total GPU Memory: {gpu_memory_mb:.0f} MB')
            logger.info(f'  - Allocated Memory: {gpu_allocated_mb:.0f} MB ({gpu_allocated_mb/gpu_memory_mb*100:.1f}%)')
            logger.info(f'  - Cached Memory: {gpu_cached_mb:.0f} MB ({gpu_cached_mb/gpu_memory_mb*100:.1f}%)')
        except Exception as e:
            logger.warning(f'Failed to get GPU memory info: {e}')
    
    # Final optimization summary before trainer initialization
    if is_main_process:
        logger.info('🚀 Final Training Configuration Summary:')
        logger.info(f'  - Mixed Precision (bf16): {"✅" if training_args.bf16 else "❌"}')
        logger.info(f'  - Gradient Checkpointing: {"✅" if model_args.grad_checkpoint else "❌"}')
        logger.info(f'  - Video Encoder Frozen: {"✅" if model_args.freeze_video_encoder else "❌"}')
        
        if hasattr(model, 'video_encoder') and model.video_encoder is not None:
            from internvl.model.videochat_flash.modeling_videochat_flash import VideoChatFlashQwenForCausalLM
            if isinstance(model.video_encoder, VideoChatFlashQwenForCausalLM):
                gc_enabled = getattr(model.video_encoder.model, 'gradient_checkpointing', False)
                attn_impl = getattr(model.video_encoder.config, '_attn_implementation', 'unknown')
                logger.info(f'  - VideoChatFlash Optimizations:')
                logger.info(f'    * Flash Attention 2: {"✅" if attn_impl == "flash_attention_2" else "❌"} ({attn_impl})')
                logger.info(f'    * Gradient Checkpointing: {"✅" if gc_enabled else "❌"}')
                logger.info(f'    * Training Mode: {"✅" if model.video_encoder.training else "❌"}')
        logger.info('🎯 Ready to start training!')

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
