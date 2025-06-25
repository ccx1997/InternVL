"""
e.g.,  python shell/data/validate_vsi_meta.py --config shell/data/llava_video_178k_tmp.json --parallel 8 --skip-video-check --save
"""
import json
import os
import argparse
from PIL import Image
import concurrent.futures
import warnings
import cv2
import subprocess
import tempfile
from contextlib import contextmanager
import sys
from tqdm import tqdm
import itertools
from datetime import datetime


@contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr.
    """
    # Save the current stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


def check_image(image_path):
    """
    Checks if a single image is valid by attempting to open and convert it.
    
    :param image_path: The full path to the image file.
    """
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with Image.open(image_path).convert('RGB') as img:
                wid, height = img.size
                if wid > 20000 or height > 20000 or height / wid > 30 or wid / height > 30 or wid < 10 or height < 10:
                    print(f"Wrong image shape {wid}, {height}: {image_path}")
                    return False
                if len(w) > 0 and any("Corrupt EXIF" in str(warning.message) for warning in w):
                    print(f"Corrupt EXIF data: {image_path}")
                    for warning in w:
                        print(str(warning.message))
                    return False
    except Exception as e:
        print(f"Invalid image file {image_path}: {e}")
        return False
    return True


def check_video(video_path):
    """
    Checks if a single video is valid by using ffprobe.
    
    :param video_path: The full path to the video file.
    """
    try:
        # First try using ffprobe (more reliable)
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration',
            '-of', 'json',
            video_path
        ]
        
        try:
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            # If ffprobe is not available, fall back to OpenCV
            pass
        
        # Fall back to OpenCV if ffprobe fails or is not available
        with suppress_stdout_stderr():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open video file: {video_path}")
                return False
            
            # Try to read the first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"Cannot read frames from video: {video_path}")
                return False
                
            return True
            
    except Exception as e:
        print(f"Invalid video file {video_path}: {e}")
        return False


def find_media_file(media_file, dataset_config):
    """
    Find the actual path of a media file by trying different path combinations.
    Priority order:
    1. root + media_file
    2. absolute path (media_file as is)
    3. data_dir + media_file
    4. root + data_dir + media_file
    
    :param media_file: The media file path from annotation
    :param dataset_config: Dataset configuration
    :return: Full path if file exists, None otherwise
    """
    root_dir = dataset_config.get('root', '').strip()
    data_dir = dataset_config.get('data_dir', '').strip()
    
    # List of possible paths to try (in priority order)
    possible_paths = []
    
    # 1. If root is specified, try root + media_file first
    if root_dir and media_file:
        if not os.path.isabs(media_file):  # Only join if media_file is not absolute
            possible_paths.append(os.path.join(root_dir, media_file))
    
    # 2. Try the file path as is (might be absolute)
    if media_file:
        possible_paths.append(media_file)
    
    # 3. Try with data directory
    if data_dir and media_file:
        if not os.path.isabs(media_file):  # Only join if media_file is not absolute
            possible_paths.append(os.path.join(data_dir, media_file))
    
    # 4. Try combinations of root and data directory
    if root_dir and data_dir and media_file:
        if not os.path.isabs(media_file):  # Only join if media_file is not absolute
            possible_paths.append(os.path.join(root_dir, data_dir, media_file))
            possible_paths.append(os.path.join(data_dir, root_dir, media_file))
    
    # Remove any duplicates while preserving order
    possible_paths = list(dict.fromkeys(possible_paths))
    
    # Debug info
    if not possible_paths:
        print(f"Warning: No valid paths constructed for {media_file}")
        print(f"Root: {root_dir}")
        print(f"Data dir: {data_dir}")
        return None
    
    # Check each possible path
    for path in possible_paths:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
        
    # If no file found, print debug info
    print(f"Could not find file: {media_file}")
    print("Tried the following paths:")
    for path in possible_paths:
        print(f"- {path}")
    
    return None


def get_conversation_value(conv):
    """
    Get conversation value from different formats.
    
    :param conv: Conversation dictionary
    :return: tuple of (role/from, content/value)
    """
    # New format
    if 'from' in conv and 'value' in conv:
        return conv['from'], conv['value']
    # Old format
    elif 'role' in conv and 'content' in conv:
        role_mapping = {
            'user': 'human',
            'assistant': 'gpt',
            'human': 'human',
            'gpt': 'gpt'
        }
        return role_mapping.get(conv['role'], conv['role']), conv['content']
    else:
        raise ValueError("Unknown conversation format")


def validate_entry(entry, dataset_config, skip_video_check=False):
    """
    Validates a single entry from the annotation file.
    
    :param entry: The entry to validate
    :param dataset_config: Configuration for the dataset
    :param skip_video_check: If True, skips video file validation
    """
    entry_id = entry.get('id', entry.get('question_id', 'unknown'))
    
    # Check conversations structure
    if 'conversations' not in entry or not isinstance(entry['conversations'], list):
        print(f"Invalid conversations structure: {entry_id}")
        return False

    if not entry['conversations']:
        print(f"Empty conversations: {entry_id}")
        return False

    try:
        first_role, _ = get_conversation_value(entry['conversations'][0])
        if first_role not in ['human', 'user']:
            print(f"First conversation must be from human/user: {entry_id}")
            return False

        for i, conv in enumerate(entry['conversations']):
            try:
                role, content = get_conversation_value(conv)
            except ValueError as e:
                print(f"Invalid conversation format at index {i}: {entry_id}")
                return False

            if not isinstance(content, str):
                print(f"Conversation content must be string: {entry_id}")
                return False
            
            expected_role = 'human' if i % 2 == 0 else 'gpt'
            if role not in ['human', 'user'] and i % 2 == 0:
                print(f"Even-indexed conversations must be from human/user: {entry_id}")
                return False
            if role not in ['gpt', 'assistant'] and i % 2 == 1:
                print(f"Odd-indexed conversations must be from gpt/assistant: {entry_id}")
                return False
            
            if i % 2 == 1 and not content.strip():  # GPT responses should not be empty
                print(f"Empty GPT/assistant response at index {i}: {entry_id}")
                return False

    except Exception as e:
        print(f"Error validating conversations for entry {entry_id}: {e}")
        return False

    # Check media files (image or video)
    media_files = []
    media_type = None
    
    if 'image' in entry and entry['image']:
        media_type = 'image'
        if isinstance(entry['image'], list):
            media_files = entry['image']
        else:
            media_files = [entry['image']]
    elif 'src_image' in entry and entry['src_image']:
        media_type = 'image'
        if isinstance(entry['src_image'], list):
            media_files = entry['src_image']
        else:
            media_files = [entry['src_image']]
    elif 'video' in entry and entry['video']:
        media_type = 'video'
        data_source = entry.get('data_source', '').strip()
        if isinstance(entry['video'], str):
            entry['video'] = [entry['video']]
        media_files = [os.path.join(data_source, video) for video in entry['video']]
    
    # Check placeholder count
    try:
        # Count all possible placeholder formats
        placeholder_count = 0
        for conv in entry['conversations']:
            _, content = get_conversation_value(conv)
            placeholder_count += (
                content.count("<image>") +
                content.count("<image_placeholder>") +
                content.count("<video>")
            )
    except Exception as e:
        print(f"Error counting placeholders in conversations: {entry_id}")
        return False

    if media_files:
        # Validate file extensions
        if media_type == 'image':
            valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        else:  # video
            valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
        
        # There are some files without any extension, we don't check the extension
        # for media_file in media_files:
        #     if not media_file.lower().endswith(valid_extensions):
        #         print(f"Invalid {media_type} file extension: {media_file} for entry {entry_id}")
        #         return False
        
        # Check if placeholder count matches media file count
        if placeholder_count != len(media_files):
            print(f"Placeholder count ({placeholder_count}) doesn't match {media_type} count ({len(media_files)}): {entry_id}")
            return False
        
        # Check if files exist locally
        for media_file in media_files:
            actual_path = find_media_file(media_file, dataset_config)
            if not actual_path:
                print(f"Cannot find {media_type} file: {media_file} for entry {entry_id}")
                return False
            
            # Validate the file
            if media_type == 'image':
                if not check_image(actual_path):
                    print(f"Invalid {media_type} file: {actual_path} for entry {entry_id}")
                    return False
            else:  # video
                if not skip_video_check:
                    if not check_video(actual_path):
                        print(f"Invalid {media_type} file: {actual_path} for entry {entry_id}")
                        return False
    else:
        # No media files, check if there are placeholders (should not have placeholders without media)
        if placeholder_count > 0:
            print(f"Found {placeholder_count} placeholders but no media files: {entry_id}")
            return False

    return True


def load_annotation_file(annotation_path):
    """
    Load annotation file (JSON or JSONL format).
    
    :param annotation_path: Path to annotation file
    :return: List of entries
    """
    if not os.path.exists(annotation_path):
        print(f"Annotation file not found: {annotation_path}")
        return []
    
    entries = []
    try:
        if annotation_path.endswith('.jsonl'):
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error at line {line_num} in {annotation_path}: {e}")
        else:  # JSON format
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    entries = data
                else:
                    print(f"Expected list in JSON file: {annotation_path}")
                    return []
    except Exception as e:
        print(f"Error loading annotation file {annotation_path}: {e}")
        return []
    
    return entries


QUICK_CHECK_COUNT = 1000


def filter_valid_entries(entries, dataset_config, num_threads, quick_mode=False, skip_video_check=False):
    """
    Filter valid entries using parallel processing.

    :param entries: List of entries to validate
    :param dataset_config: Dataset configuration
    :param num_threads: Number of threads for parallel processing
    :param quick_mode: If True, only validate first 100 entries
    :param skip_video_check: If True, skips video file validation
    :return: List of valid entries, early_stop flag
    """
    entries_to_check = entries[:QUICK_CHECK_COUNT] if quick_mode else entries
    
    valid_entries = []
    has_invalid = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Use executor.map for better memory efficiency with large iterables
        results = executor.map(validate_entry, entries_to_check, itertools.repeat(dataset_config), itertools.repeat(skip_video_check))
        
        with tqdm(total=len(entries_to_check), desc="Validating entries", unit="entry") as pbar:
            for i, is_valid in enumerate(results):
                if is_valid:
                    valid_entries.append(entries_to_check[i])
                else:
                    has_invalid = True
                    if quick_mode:
                        print(f"\nFound invalid entry, stopping quick validation.")
                        return valid_entries, False  # Early exit in quick mode

                pbar.update(1)

    # In quick mode, if all checked entries are valid, return all entries
    if quick_mode and not has_invalid:
        print(f"\nQuick validation successful! All first {QUICK_CHECK_COUNT} entries are valid.")
        return entries, True
    
    return valid_entries, False


def process_dataset(dataset_name, dataset_config, num_threads, save_results, output_dir, quick_mode=False, skip_video_check=False):
    """
    Process a single dataset.
    
    :param dataset_name: Name of the dataset
    :param dataset_config: Configuration for the dataset
    :param num_threads: Number of threads for parallel processing
    :param save_results: Whether to save filtered results
    :param output_dir: Directory to save output files
    :param quick_mode: If True, only validate first 100 entries
    :param skip_video_check: If True, skips video file validation
    :return: True if validation was successful (or quick validation passed)
    """
    annotation_path = dataset_config.get('annotation')
    if not annotation_path:
        print(f"No annotation path specified for dataset: {dataset_name}")
        return False
    
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Annotation file: {annotation_path}")
    print(f"Root directory: {dataset_config.get('root', 'Not specified')}")
    print(f"Data directory: {dataset_config.get('data_dir', 'Not specified')}")
    
    # Load entries
    entries = load_annotation_file(annotation_path)
    if not entries:
        print(f"No entries loaded from {annotation_path}")
        return False
    
    total_entries = len(entries)
    if quick_mode:
        print(f"Quick mode: Will validate first {QUICK_CHECK_COUNT} entries out of {total_entries}")
    else:
        print(f"Found {total_entries} entries to validate")
    
    # Filter valid entries
    valid_entries, early_stop = filter_valid_entries(entries, dataset_config, num_threads, quick_mode, skip_video_check)
    
    # Report results
    if early_stop:
        tag = "✅"
        result_msg = f"{tag} Dataset {dataset_name}: Quick validation passed (first {QUICK_CHECK_COUNT}/{total_entries} entries valid)"
    else:
        tag = "✅" if len(valid_entries) == total_entries else "❌"
        result_msg = f"{tag} Dataset {dataset_name}: {len(valid_entries)} valid entries from {total_entries} total"
    
    print(f"\n{result_msg}")
    # cached to a txt file
    with open(os.path.join(output_dir, f"validation_log.txt"), 'a') as f:
        f.write(result_msg + '\n')
    
    # Save results if requested and there are invalid entries
    if save_results and len(valid_entries) != total_entries and not early_stop:
        fn0, fn1 = os.path.splitext(annotation_path)
        output_path = f"{fn0}_filtered{fn1}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(valid_entries, f, indent=2, ensure_ascii=False)
            sav_info = f"Filtered results saved to: {output_path}"
        except Exception as e:
            sav_info = f"Error saving filtered results: {e}"
        print(sav_info)
        with open(os.path.join(output_dir, f"validation_log.txt"), 'a') as f:
            f.write(sav_info + '\n')
    
    return early_stop or len(valid_entries) == total_entries


def main():
    parser = argparse.ArgumentParser(description="Filter VSI meta dataset entries.")
    parser.add_argument("--config", required=True, help="Path to vsi_meta_test.json config file")
    parser.add_argument("--datasets", nargs='*', help="Specific datasets to process (process all if not specified)")
    parser.add_argument("--parallel", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--save", action="store_true", help="Save filtered results")
    parser.add_argument("--output_dir", default="./filtered_results", help="Directory to save output files")
    parser.add_argument("--log", action="store_true", help="Enable logging to file")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode (only check first 1000 entries)")
    parser.add_argument("--skip-video-check", action="store_true", help="Skip the time-consuming video file validation.")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return
    
    # Setup logging
    if args.log:
        log_file = os.path.join(args.output_dir, "validation_log.txt")
        os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, f"validation_log.txt"), 'a') as f:
        f.write(f"\n\nValidation on {args.config}, started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Process datasets
    datasets_to_process = args.datasets if args.datasets else list(config.keys())
    total_datasets = len(datasets_to_process)
    
    print(f"Starting validation of {total_datasets} datasets...")
    if args.quick:
        print(f"Quick validation mode enabled (will check only first {QUICK_CHECK_COUNT} entries per dataset)")
    
    for i, dataset_name in enumerate(datasets_to_process, 1):
        if dataset_name not in config:
            print(f"\nDataset {dataset_name} not found in config")
            continue
        
        try:
            print(f"\nDataset {i}/{total_datasets}")
            success = process_dataset(
                dataset_name, 
                config[dataset_name], 
                args.parallel, 
                args.save, 
                args.output_dir,
                args.quick,
                args.skip_video_check
            )
            if not success:
                print(f"Dataset {dataset_name} validation failed")
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
        
        print("-" * 50)


if __name__ == "__main__":
    main()
