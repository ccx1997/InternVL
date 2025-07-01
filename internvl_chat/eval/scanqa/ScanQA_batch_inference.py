#!/usr/bin/env python3
"""
Batch inference script for ScanQA dataset using InternVL Chat Dual Encoder
"""
import sys
sys.path.append('/mnt/chensenda/codes/VLN/InternVL/internvl_chat')
import os
import argparse
import json
import sys
import time
import torch
from tqdm import tqdm
from typing import List, Dict, Any

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU utilization monitoring will be disabled.")

# Import functions from simple_inference_demo.py
from simple_inference_demo import (
    load_model_and_tokenizer,
    patch_model_chat_method,
    load_video_frames,
    load_image,
    preprocess_images,
    simple_chat
)

# Set CUDA device after imports to override any settings from imported modules
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return list of records"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_question_from_conversations(conversations: List[Dict]) -> str:
    """Extract question from ScanQA conversations"""
    for conv in conversations:
        if conv.get('from') == 'human':
            # Remove <video> token from question 
            question = conv.get('value', '').replace('<video>\n', '').replace('<video>', '')
            return question
    return ""


def uniform_sample_images(image_paths: List[str], num_frames: int = 16) -> List[str]:
    """Uniformly sample num_frames images from the image path list"""
    if len(image_paths) <= num_frames:
        return image_paths
    
    # Calculate step size for uniform sampling
    step = (len(image_paths) - 1) / (num_frames - 1)
    sampled_indices = [int(round(i * step)) for i in range(num_frames)]
    
    # Ensure the last index is included
    sampled_indices[-1] = len(image_paths) - 1
    
    return [image_paths[i] for i in sampled_indices]


def process_single_sample(model, tokenizer, sample: Dict[str, Any], 
                         data_root: str, args) -> Dict[str, Any]:
    """Process a single sample and return prediction with ScanQA format"""
    try:
        # Extract info from sample
        sample_id = sample['id']
        video_path = sample['video']  # Changed: now treat as single video file path
        question = extract_question_from_conversations(sample['conversations'])
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            # Return sample with empty internVL response
            result_sample = sample.copy()
            result_sample['conversations'].append({
                'from': 'internVL',
                'value': ""
            })
            return result_sample
        
        # Load video frames using the same method as simple_inference_demo.py
        imgs = load_video_frames(video_path,
                                min_frames=args.num_frames,
                                max_frames=args.num_frames,
                                sampling='rand')
        
        if not imgs:
            print(f"Warning: Failed to load video frames for sample {sample_id}")
            # Return sample with empty internVL response
            result_sample = sample.copy()
            result_sample['conversations'].append({
                'from': 'internVL',
                'value': ""
            })
            return result_sample
        
        # Create frame info for multiple video frames (same as simple_inference_demo.py)
        frame_info = '\n'.join([f'Frame-{i+1}: <image>' for i in range(len(imgs))])
        q = question.replace('<video>', frame_info) if '<video>' in question \
            else frame_info + '\n' + question
        
        # For video frames, use smaller max_patches per frame (same as simple_inference_demo.py)
        max_patches = 1  # Since we have multiple video frames, use 1 patch per frame
        
        # Preprocess
        pv1, pv2 = preprocess_images(imgs,
                                   image_size=model.config.force_image_size
                                   or model.config.vision_config.image_size,
                                   dynamic_size=True,
                                   use_thumbnail=model.config.use_thumbnail,
                                   max_patches=max_patches)
        
        # Inference
        answer = simple_chat(model, tokenizer, pv1, pv2,
                           question=q, max_tokens=args.max_tokens)
        
        # Prepare result in ScanQA format with conversations
        result_sample = sample.copy()
        
        # Add internVL response to conversations
        result_sample['conversations'].append({
            'from': 'internVL',
            'value': answer.strip()
        })
        
        return result_sample
        
    except Exception as e:
        print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
        result_sample = sample.copy()
        
        # Add empty internVL response on error
        result_sample['conversations'].append({
            'from': 'internVL',
            'value': ""
        })
        return result_sample


def main():
    parser = argparse.ArgumentParser(description='Batch inference for ScanQA dataset')
    parser.add_argument('--checkpoint', 
                        default='/mnt/models/InternVL3-8B',
                        help='Path to dual-encoder checkpoint')
    parser.add_argument('--dataset', 
                        default='/mnt/chensenda/codes/VLN/ScanQA/ScanQA_v1.0_val_reformat_std_video.json',
                        help='Path to ScanQA dataset')
    parser.add_argument('--data-root',
                        default='',  # Not needed since paths are absolute in the dataset
                        help='Root directory for video files (not used as paths are absolute)')
    parser.add_argument('--output', 
                        default='scanqa_predictions_internvl3_8b.json',
                        help='Output file for predictions')
    parser.add_argument('--num-frames', type=int, default=12,
                        help='Number of frames to sample from each video file')
    parser.add_argument('--max-patches', type=int, default=1,
                        help='Dynamic patches per video frame')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Maximum generation length')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start index for processing (for resuming)')
    parser.add_argument('--end-idx', type=int, default=-1,
                        help='End index for processing (-1 for all)')
    parser.add_argument('--disable-gpu-monitor', action='store_true',
                        help='Disable GPU performance monitoring')
    args = parser.parse_args()

    # Initialize GPU monitor
    gpu_monitor = None if args.disable_gpu_monitor else GPUMonitor()
    
    # Load model
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    model = patch_model_chat_method(model)
    print("Model loaded successfully!")

    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dataset = load_json(args.dataset)
    print(f"Loaded {len(dataset)} samples")
    
    # Determine processing range
    start_idx = args.start_idx
    end_idx = len(dataset) if args.end_idx == -1 else min(args.end_idx, len(dataset))
    dataset_subset = dataset[start_idx:end_idx]
    
    print(f"Processing samples {start_idx} to {end_idx-1} ({len(dataset_subset)} samples)")

    # Process samples
    results = []
    count = 0
    total_start_time = time.time()
    
    for sample in tqdm(dataset_subset, desc="Processing samples"):
        count += 1
        if count > 1000:
            break
        
        # Record start time for this iteration
        iter_start_time = time.time()
        
        result = process_single_sample(model, tokenizer, sample, args.data_root, args)
        results.append(result)
        
        # Record end time and GPU metrics
        iter_end_time = time.time()
        if gpu_monitor:
            gpu_monitor.record_metrics(iter_start_time, iter_end_time)
        
        # Print progress every 50 samples
        if len(results) % 50 == 0:
            elapsed_time = time.time() - total_start_time
            samples_per_sec = len(results) / elapsed_time
            # Get the internVL response for progress display
            internvl_response = ""
            if 'conversations' in result:
                for conv in result['conversations']:
                    if conv.get('from') == 'internVL':
                        internvl_response = conv.get('value', '')
                        break
            
            sample_id = result.get('id', 'unknown')
            print(f"Processed {len(results)} samples. Latest: id={sample_id}, prediction='{internvl_response[:50]}...', Speed: {samples_per_sec:.2f} samples/sec")
    
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