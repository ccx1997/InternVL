import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from tqdm import tqdm
import traceback

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=16):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def load_model():
    """Load InternVL3-2B model and tokenizer"""
    path = '/mnt/models/InternVL3-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def process_question(model, tokenizer, video_path, question, base_path="/mnt/chengchangxu/data/VSI-Bench"):
    """Process a single question and return the model's answer"""
    try:
        # Load video
        full_video_path = os.path.join(base_path, video_path)
        if not os.path.exists(full_video_path):
            return f"Error: Video file not found: {full_video_path}"
        
        pixel_values, num_patches_list = load_video(full_video_path, num_segments=32, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        
        # Create video prefix
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        
        # Extract the actual question (remove <video> tag if present)
        if question.startswith('<video>\n'):
            question_text = question[8:]  # Remove '<video>\n'
        else:
            question_text = question
        
        # Combine video prefix with question
        full_question = video_prefix + question_text +"If you can answer with a number, answer with a number."
        
        # Generate response
        generation_config = dict(max_new_tokens=1024, do_sample=False, temperature=0.0)
        response, _ = model.chat(tokenizer, pixel_values, full_question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
        
        return response.strip()
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"Error for video {video_path}: {error_msg}")
        traceback.print_exc()
        return error_msg

def main():
    # Load test data
    test_file = "/mnt/chengchangxu/data/VSI-Bench/vsi_bench_test.jsonl"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model
    print("Loading InternVL3-8B model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!")
    
    # Process each test sample
    results = []
    
    for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
        sample_id = sample['id']
        video_path = sample['video']
        conversations = sample['conversations']
        question_type = sample['type']
        # if question_type != "object_counting":
        if "object_rel_direction" not in sample['type']:
            continue
        
        # Get the human question
        human_question = None
        for conv in conversations:
            if conv['from'] == 'human':
                human_question = conv['value']
                break
        
        if human_question is None:
            print(f"Warning: No human question found for sample {sample_id}")
            continue
        
        # Get model prediction
        prediction = process_question(model, tokenizer, video_path, human_question)
        
        # Create result entry
        result = {
            "idx": sample_id,
            "prediction": prediction
        }
        results.append(result)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_data)} samples")
    
    # Save results
    output_file = "./internvl3_8b_predictions_32frames-object_rel_direction.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    print(f"Total processed: {len(results)} samples")

if __name__ == "__main__":
    main() 