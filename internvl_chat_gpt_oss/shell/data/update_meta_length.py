import json
import os
from typing import Dict, Any

def count_data_lines(annotation_path: str) -> int:
    """
    Count the number of data entries in annotation file
    
    Args:
        annotation_path: Path to the annotation file (json or jsonl)
    
    Returns:
        Number of data entries
    """
    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file not found: {annotation_path}")
        return 0
    
    try:
        # Check file extension
        if annotation_path.endswith('.jsonl'):
            # For JSONL files, count lines
            with open(annotation_path, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if line.strip():  # Skip empty lines
                        count += 1
                return count
        else:
            # For JSON files, load and count array length
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return len(data)
                elif isinstance(data, dict):
                    # If it's a dict, it might contain multiple keys with arrays
                    # Return 1 for single object or sum of all arrays
                    return 1
                else:
                    return 1
    except Exception as e:
        print(f"Error reading {annotation_path}: {e}")
        return 0

def update_meta_file_length(meta_file_path: str):
    """
    Update length field for all datasets in a meta file
    
    Args:
        meta_file_path: Path to the meta file
    """
    print(f"Processing meta file: {meta_file_path}")
    
    if not os.path.exists(meta_file_path):
        print(f"Meta file not found: {meta_file_path}")
        return
    
    # Create backup
    backup_path = meta_file_path + ".backup"
    os.system(f"cp '{meta_file_path}' '{backup_path}'")
    print(f"Backup created: {backup_path}")
    
    try:
        # Load meta file
        with open(meta_file_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        total_updated = 0
        
        # Update each dataset
        for dataset_name, dataset_config in meta_data.items():
            if isinstance(dataset_config, dict) and 'annotation' in dataset_config:
                annotation_path = dataset_config['annotation']
                old_length = dataset_config.get('length', 0)
                
                # Count actual data entries
                new_length = count_data_lines(annotation_path)
                
                # Update length
                dataset_config['length'] = new_length
                total_updated += 1
                
                print(f"  {dataset_name}: {old_length} -> {new_length} (annotation: {annotation_path})")
        
        # Save updated meta file
        with open(meta_file_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        print(f"Updated {total_updated} datasets in {meta_file_path}")
        print()
        
    except Exception as e:
        print(f"Error processing {meta_file_path}: {e}")
        # Restore from backup
        if os.path.exists(backup_path):
            os.system(f"cp '{backup_path}' '{meta_file_path}'")
            print("Restored from backup due to error")

def main():
    """
    Main function to update all meta files
    """
    # List of meta files to update
    meta_files = [
        "./shell/data/vsi_train.json",
        "./shell/data/llava_video_178k.json", 
        "./shell/data/visual_grounding.json",
        "./shell/data/general_mm.json",
        "./shell/data/ocr.json"
    ]
    
    print("Starting to update meta file lengths...")
    print("=" * 50)
    
    for meta_file in meta_files:
        if os.path.exists(meta_file):
            update_meta_file_length(meta_file)
        else:
            print(f"Meta file not found: {meta_file}")
            print()
    
    print("=" * 50)
    print("All meta files processed!")

if __name__ == "__main__":
    main() 