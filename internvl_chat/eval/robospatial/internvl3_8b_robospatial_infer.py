from itertools import count
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from tqdm import tqdm
import traceback
from datasets import load_dataset

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

def load_image(image_file, input_size=448, max_num=6):
    """Load and preprocess image for InternVL3-8B"""
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        # Handle PIL Image object directly
        image = image_file.convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    """Split model across multiple GPUs if available"""
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return None
    
    # This is a simplified version - adjust based on your GPU memory
    # For InternVL3-8B, you might need to adjust this
    return None

def load_model():
    """Load InternVL3-8B model and tokenizer"""
    path = '/mnt/models/InternVL3-8B'
    
    # Try to split model if multiple GPUs available
    device_map = split_model('InternVL3-8B')
    
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    
    # If no device_map, move to single GPU
    if device_map is None:
        model = model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

def process_question(model, tokenizer, image, question):
    """Process a single question and return the model's answer"""
    try:
        # Load and preprocess image
        pixel_values = load_image(image, max_num=8).to(torch.bfloat16).cuda()
        
        # Prepare question with image token
        full_question = f"<image>\n{question}"
        
        # Generate response
        generation_config = dict(max_new_tokens=1024, do_sample=False, temperature=0.0)
        response = model.chat(tokenizer, pixel_values, full_question, generation_config)
        
        return response.strip()
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"Error: {error_msg}")
        traceback.print_exc()
        return error_msg

def main():
    # Load RoboSpatial-Home dataset
    print("Loading RoboSpatial-Home dataset...")
    ds = load_dataset("chanhee-luke/RoboSpatial-Home")
    
    # Combine all splits into a single list
    all_data = []
    for split_name in ['context', 'compatibility', 'configuration']:
        for item in ds[split_name]:
            all_data.append(item)
    
    print(f"Loaded {len(all_data)} samples from RoboSpatial-Home dataset")
    
    # Load model
    print("Loading InternVL3-8B model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!")
    
    # Process each sample
    results = []
    count=0
    for i, sample in enumerate(tqdm(all_data, desc="Processing samples")):
        question = sample['question']
        image = sample['img']  # This is a PIL Image object
        category = sample['category']
        ground_truth = sample['answer']
        
        if image is None:
            print(f"Warning: No image found for sample {i}")
            continue
        # if i>5:
        #     break
        # Get model prediction
        prediction = process_question(model, tokenizer, image, question)
        
        # Create result entry
        result = {
            "idx": i,
            "prediction": prediction
        }
        results.append(result)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(all_data)} samples")
            print(f"Sample {i}: {question[:100]}...")
            print(f"Ground truth: {ground_truth}")
            print(f"Prediction: {prediction}")
            print("-" * 50)
    
    # Save results
    output_file = "/mnt/chensenda/codes/VLN/InternVL/internvl_chat/eval/robospatial/internvl3_8b_robospatial_predictions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    print(f"Total processed: {len(results)} samples")

if __name__ == "__main__":
    main() 