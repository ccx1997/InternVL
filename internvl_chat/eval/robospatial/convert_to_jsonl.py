import os
import json
from datasets import load_dataset
from PIL import Image

def convert_robospatial_to_jsonl():
    """Convert RoboSpatial-Home dataset to VSI-Bench format jsonl"""
    
    # Load the dataset
    print("Loading RoboSpatial-Home dataset...")
    ds = load_dataset("chanhee-luke/RoboSpatial-Home")
    
    # Output directory for images
    images_dir = "/mnt/chensenda/codes/VLN/InternVL/internvl_chat/eval/robospatial/images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Output jsonl file
    output_file = "/mnt/chensenda/codes/VLN/InternVL/internvl_chat/eval/robospatial/robospatial_test.jsonl"
    
    all_data = []
    current_id = 0
    
    # Process each split
    for split_name in ['context', 'compatibility', 'configuration']:
        print(f"Processing split: {split_name}")
        
        for item_idx, item in enumerate(ds[split_name]):
            # Save image
            image_filename = f"{split_name}_{item_idx:04d}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            # Save the main image
            if item['img'] is not None:
                # Convert PIL image to RGB if needed
                img = item['img']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(image_path)
                
                # Create jsonl entry
                entry = {
                    "id": current_id,
                    "image": f"images/{image_filename}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{item['question']}"
                        },
                        {
                            "from": "gpt", 
                            "value": item['answer']
                        }
                    ],
                    "type": item['category']
                }
                
                all_data.append(entry)
                current_id += 1
                
                # Optional: Also save depth image if available
                if item['depth_image'] is not None:
                    depth_filename = f"{split_name}_{item_idx:04d}_depth.jpg"
                    depth_path = os.path.join(images_dir, depth_filename)
                    
                    depth_img = item['depth_image']
                    if depth_img.mode != 'RGB':
                        depth_img = depth_img.convert('RGB')
                    depth_img.save(depth_path)
                
                # Optional: Also save mask if available
                if item['mask'] is not None:
                    mask_filename = f"{split_name}_{item_idx:04d}_mask.jpg"
                    mask_path = os.path.join(images_dir, mask_filename)
                    
                    mask_img = item['mask']
                    if mask_img.mode != 'RGB':
                        mask_img = mask_img.convert('RGB')
                    mask_img.save(mask_path)
            
            if (item_idx + 1) % 10 == 0:
                print(f"  Processed {item_idx + 1} items in {split_name}")
    
    # Write jsonl file
    print(f"Writing {len(all_data)} entries to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed!")
    print(f"- Total entries: {len(all_data)}")
    print(f"- Images saved to: {images_dir}")
    print(f"- JSONL file saved to: {output_file}")
    
    # Show some statistics
    categories = [entry['type'] for entry in all_data]
    unique_categories = list(set(categories))
    print(f"- Categories: {unique_categories}")
    for cat in unique_categories:
        count = categories.count(cat)
        print(f"  {cat}: {count} items")

if __name__ == "__main__":
    convert_robospatial_to_jsonl() 