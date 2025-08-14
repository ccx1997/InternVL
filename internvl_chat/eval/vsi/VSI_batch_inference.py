#!/usr/bin/env python3
"""
Batch inference script for VSI-Bench dataset using InternVL Chat Dual Encoder
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
sys.path.append('/mnt/chensenda/codes/VLN/InternVL_video/internvl_chat')
import argparse
import json

import sys
import time
import random
random.seed(42)

import torch
from tqdm import tqdm
from typing import List, Dict, Any

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU utilization monitoring will be disabled.")

# Import functions from batch_inference_demo.py
from batch_inference_demo import (
    load_model_and_tokenizer,
    patch_model_batch_method,
    load_video_frames,
    load_image,
    preprocess_media_batch,
    batch_inference
)

# ... existing code ...

class GPUMonitor:
    """Monitor GPU performance metrics"""
    
    def __init__(self):
        self.memory_usage = []
        self.gpu_utilization = []
        self.iteration_times = []
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
            except:
                self.handles = None
                print("Warning: Failed to initialize pynvml")
        else:
            self.handles = None
    
    def record_metrics(self, start_time: float, end_time: float):
        """Record GPU metrics for current iteration"""
        # Record iteration time
        iteration_time = end_time - start_time
        self.iteration_times.append(iteration_time)
        
        # Record GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            self.memory_usage.append({
                'allocated': memory_allocated,
                'reserved': memory_reserved
            })
        
        # Record GPU utilization
        if self.handles:
            try:
                utilization_rates = []
                for handle in self.handles:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization_rates.append(util.gpu)
                self.gpu_utilization.append(utilization_rates)
            except:
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get average statistics"""
        stats = {}
        
        # Iteration speed statistics
        if self.iteration_times:
            stats['avg_iteration_time'] = sum(self.iteration_times) / len(self.iteration_times)
            stats['min_iteration_time'] = min(self.iteration_times)
            stats['max_iteration_time'] = max(self.iteration_times)
            stats['samples_per_second'] = 1.0 / stats['avg_iteration_time']
        
        # Memory statistics
        if self.memory_usage:
            avg_allocated = sum(mem['allocated'] for mem in self.memory_usage) / len(self.memory_usage)
            avg_reserved = sum(mem['reserved'] for mem in self.memory_usage) / len(self.memory_usage)
            max_allocated = max(mem['allocated'] for mem in self.memory_usage)
            max_reserved = max(mem['reserved'] for mem in self.memory_usage)
            
            stats['memory'] = {
                'avg_allocated_gb': avg_allocated,
                'avg_reserved_gb': avg_reserved,
                'max_allocated_gb': max_allocated,
                'max_reserved_gb': max_reserved
            }
        
        # GPU utilization statistics
        if self.gpu_utilization:
            # Average across all GPUs and all iterations
            all_utils = []
            for iteration_utils in self.gpu_utilization:
                all_utils.extend(iteration_utils)
            if all_utils:
                stats['avg_gpu_utilization_percent'] = sum(all_utils) / len(all_utils)
                stats['max_gpu_utilization_percent'] = max(all_utils)
        
        return stats
    
    def print_statistics(self):
        """Print performance statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("GPU PERFORMANCE STATISTICS")
        print("="*50)
        
        if 'avg_iteration_time' in stats:
            print(f"Average iteration time: {stats['avg_iteration_time']:.3f}s")
            print(f"Min iteration time: {stats['min_iteration_time']:.3f}s")
            print(f"Max iteration time: {stats['max_iteration_time']:.3f}s")
            print(f"Processing speed: {stats['samples_per_second']:.2f} samples/second")
        
        if 'memory' in stats:
            mem = stats['memory']
            print(f"\nGPU Memory Usage:")
            print(f"  Average allocated: {mem['avg_allocated_gb']:.2f} GB")
            print(f"  Average reserved: {mem['avg_reserved_gb']:.2f} GB")
            print(f"  Peak allocated: {mem['max_allocated_gb']:.2f} GB")
            print(f"  Peak reserved: {mem['max_reserved_gb']:.2f} GB")
        
        if 'avg_gpu_utilization_percent' in stats:
            print(f"\nGPU Utilization:")
            print(f"  Average utilization: {stats['avg_gpu_utilization_percent']:.1f}%")
            print(f"  Peak utilization: {stats['max_gpu_utilization_percent']:.1f}%")
        
        print("="*50)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_question_from_conversations(conversations: List[Dict]) -> str:
    """Extract question from conversations"""
    for conv in conversations:
        if conv.get('from') == 'human':
            # Remove <video> token from question 
            question = conv.get('value', '').replace('<video>\n', '').replace('<video>', '')
            return question
    return ""


def prepare_data_for_batch(samples: List[Dict[str, Any]], data_root: str) -> tuple:
    """Prepare data for batch processing"""
    input_paths = []
    questions = []
    sample_ids = []
    
    for sample in samples:
        sample_id = sample['id']
        video_path = os.path.join(data_root, sample['video'])
        question = extract_question_from_conversations(sample['conversations'])
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        # Add <video> token if not present and it's a video
        if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif')):
            if '<video>' not in question:
                question = '<video>\n' + question
        else:
            # For images, use <image> token
            if '<image>' not in question:
                question = '<image>\n' + question
        
        input_paths.append(video_path)
        questions.append(question)
        sample_ids.append(sample_id)
    
    return input_paths, questions, sample_ids


def process_batch_samples(model, tokenizer, samples: List[Dict[str, Any]], 
                         data_root: str, args, gpu_monitor=None) -> List[Dict[str, Any]]:
    """Process a batch of samples using batch inference"""
    batch_start_time = time.time()
    
    # Prepare data for batch processing
    input_paths, questions, sample_ids = prepare_data_for_batch(samples, data_root)
    
    if not input_paths:
        return []
    
    try:
        # Load media files
        media_batch = []
        for input_path in input_paths:
            if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif')):
                # Video processing
                imgs = load_video_frames(input_path,
                                       min_frames=args.num_frames,
                                       max_frames=args.num_frames,
                                       sampling='rand')
                if not imgs:
                    print(f"Warning: Failed to load video: {input_path}")
                    media_batch.append([])
                else:
                    media_batch.append(imgs)
            else:
                # Image processing
                try:
                    imgs = [load_image(input_path)]
                    media_batch.append(imgs)
                except Exception as e:
                    print(f"Warning: Failed to load image: {input_path}, {e}")
                    media_batch.append([])
        
        # Run batch inference
        results = batch_inference(model, tokenizer, media_batch, questions, 
                                args.max_tokens, batch_size=len(samples))
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            if i < len(sample_ids):
                formatted_results.append({
                    "idx": sample_ids[i],
                    "prediction": result['answer'].strip()
                })
        
        # Record metrics
        batch_end_time = time.time()
        if gpu_monitor:
            gpu_monitor.record_metrics(batch_start_time, batch_end_time)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        # Return empty results for failed samples
        return [{"idx": sample_id, "prediction": ""} for sample_id in sample_ids]


def main():
    parser = argparse.ArgumentParser(description='Batch inference for VSI-Bench dataset')
    parser.add_argument('--checkpoint', 
                        default='work_dirs/internvl_chat_dual_compressor/internvl_chat_dual_compressor_8b_mix_s3_2/checkpoint-6400',
                        help='Path to dual-encoder checkpoint')
    parser.add_argument('--dataset', 
                        default='/mnt/chengchangxu/data/VSI-Bench/vsi_bench_test.jsonl',
                        help='Path to VSI-Bench test dataset')
    parser.add_argument('--data-root',
                        default='/mnt/chengchangxu/data/VSI-Bench/',
                        help='Root directory for video/image files')
    parser.add_argument('--random-sample', type=int, default=-1, help="Random sample the dataset to a certain number (-1 for all)")
    parser.add_argument('--output', 
                        default='vsi_bench_compressor_8b_mix_s3_2_6400.json',
                        help='Output file for predictions')
    parser.add_argument('--num-frames', type=int, default=32,
                        help='Number of video frames to sample')
    parser.add_argument('--max-patches', type=int, default=1,
                        help='Dynamic patches per image')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Maximum generation length')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start index for processing (for resuming)')
    parser.add_argument('--end-idx', type=int, default=-1,
                        help='End index for processing (-1 for all)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--disable-gpu-monitor', action='store_true',
                        help='Disable GPU performance monitoring')
    args = parser.parse_args()

    # Initialize GPU monitor
    gpu_monitor = None if args.disable_gpu_monitor else GPUMonitor()
    
    # Load model
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    model = patch_model_batch_method(model)
    print("Model loaded successfully!")

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    print(f"save file: {args.output}")
    dataset = load_jsonl(args.dataset)
    print(f"Loaded {len(dataset)} samples")
    
    # Determine processing range
    start_idx = args.start_idx
    end_idx = len(dataset) if args.end_idx == -1 else min(args.end_idx, len(dataset))
    dataset_subset = dataset[start_idx:end_idx]
    if args.random_sample > 0:
        dataset_subset = random.sample(dataset_subset, min(args.random_sample, len(dataset_subset)))
    
    print(f"Processing samples {start_idx} to {end_idx-1} ({len(dataset_subset)} samples)")
    print(f"Using batch size: {args.batch_size}")

    # Process samples in batches
    results = []
    count = 0
    total_start_time = time.time()
    
    
    for i in tqdm(range(0, len(dataset_subset), args.batch_size), desc="Processing batches"):
        batch_end = min(i + args.batch_size, len(dataset_subset))
        batch_samples = dataset_subset[i:batch_end]
        
        # Filter samples by type if needed (uncomment to filter)
        # filtered_samples = []
        # for sample in batch_samples:
        #     if "room_size_estimation" in sample['type'] or "obj_appearance_order" in sample['type']:
        #         filtered_samples.append(sample)
        # batch_samples = filtered_samples
        
        if not batch_samples:
            continue
            
        count += len(batch_samples)
        # if count > 2650:
        #     continue
        # Process batch
        batch_results = process_batch_samples(model, tokenizer, batch_samples, 
                                            args.data_root, args, gpu_monitor)
        results.extend(batch_results)
        
        # Print progress
        if len(results) > 0 and len(results) % (args.batch_size * 10) == 0:
            elapsed_time = time.time() - total_start_time
            samples_per_sec = len(results) / elapsed_time
            print(f"Processed {len(results)} samples. Speed: {samples_per_sec:.2f} samples/sec")
    
    print(f"Processed {count} samples in {len(results)} results")
    
    # Print GPU performance statistics
    if gpu_monitor:
        gpu_monitor.print_statistics()
    
    # Save results
    print(f"Saving results to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - total_start_time
    print(f"Batch inference completed! Results saved to {args.output}")
    print(f"Processed {len(results)} samples in {total_time:.2f}s")
    print(f"Overall average speed: {len(results)/total_time:.2f} samples/second")


if __name__ == '__main__':
    main() 
