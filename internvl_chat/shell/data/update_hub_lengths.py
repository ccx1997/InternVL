import json
import os
import argparse

def update_lengths(meta_file_path):
    """
    Opens a hub-type meta file, calculates the total length for each sub-meta file,
    and updates the 'length' field in the hub meta file.

    The total length is the sum of (length * repeat_time) for each item in the sub-meta file.
    """
    if not os.path.exists(meta_file_path):
        print(f"Error: File not found at {meta_file_path}")
        return

    # The paths in meta files are relative to 'internvl_chat' directory.
    base_dir = './'

    with open(meta_file_path, 'r') as f:
        hub_data = json.load(f)

    for item in hub_data.get('data', []):
        meta_path_str = item.get('meta')
        if not meta_path_str:
            continue

        # Construct the absolute path for the sub-meta file.
        # The paths inside the json seem to be relative to the json file itself.
        sub_meta_file_path = os.path.normpath(os.path.join(base_dir, meta_path_str))

        if not os.path.exists(sub_meta_file_path):
            print(f"Warning: Sub-meta file not found: {sub_meta_file_path}")
            # As a fallback, try resolving from the project root's 'internvl_chat' dir
            # This is based on observation of file structures
            alt_path = os.path.normpath(os.path.join('internvl_chat', meta_path_str.lstrip('./')))
            if os.path.exists(alt_path):
                sub_meta_file_path = alt_path
            else:
                print(f"Warning: Also not found at alternative path: {alt_path}")
                continue

        try:
            with open(sub_meta_file_path, 'r') as f:
                sub_meta_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {sub_meta_file_path}")
            continue
        
        total_length = 0
        if isinstance(sub_meta_data, list):
            for sub_item in sub_meta_data:
                length = sub_item.get('length', 0)
                repeat_time = sub_item.get('repeat_time', 1.0)
                total_length += length * repeat_time
        elif isinstance(sub_meta_data, dict):
            for key in sub_meta_data:
                sub_item = sub_meta_data[key]
                if isinstance(sub_item, dict):
                    length = sub_item.get('length', 0)
                    repeat_time = sub_item.get('repeat_time', 1.0)
                    total_length += length * repeat_time

        item['length'] = int(total_length)
    
    # calculate the raw ratio of each item
    total = sum(item['length'] for item in hub_data['data'])
    for item in hub_data['data']:
        raw_ratio = item['length'] / total
        print(item['meta'], raw_ratio)

    with open(meta_file_path, 'w') as f:
        json.dump(hub_data, f, indent=4)

    print(f"Successfully updated lengths in {meta_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update length fields in a hub meta JSON file.")
    parser.add_argument('meta_file', type=str, help='Path to the hub meta file.')
    args = parser.parse_args()
    
    update_lengths(args.meta_file)
