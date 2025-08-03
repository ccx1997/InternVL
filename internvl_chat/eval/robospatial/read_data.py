from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("chanhee-luke/RoboSpatial-Home")

print(ds)

# Method 1: Access specific split and iterate through all items
print("\n=== Method 1: Iterate through all items in 'context' split ===")
for i, item in enumerate(ds['context']):
    print(f"Item {i}:")
    print(f"  Category: {item['category']}")
    print(f"  Question: {item['question']}")
    print(f"  Answer: {item['answer']}")
    print(f"  Image shape: {item['img'].size if item['img'] else 'None'}")
    print(f"  Depth image shape: {item['depth_image'].size if item['depth_image'] else 'None'}")
    print(f"  Mask shape: {item['mask'].size if item['mask'] else 'None'}")
    print("-" * 50)
    if i >= 2:  # Only show first 3 items for demonstration
        break

# Method 2: Access specific item by index
print("\n=== Method 2: Access specific item by index ===")
first_item = ds['context'][0]
print(f"First item in context split:")
print(f"  Category: {first_item['category']}")
print(f"  Question: {first_item['question']}")
print(f"  Answer: {first_item['answer']}")

# Method 3: Access items from different splits
print("\n=== Method 3: Access items from different splits ===")
for split_name in ['context', 'compatibility', 'configuration']:
    print(f"\nSplit: {split_name}")
    print(f"Number of items: {len(ds[split_name])}")
    
    # Show first item from each split
    if len(ds[split_name]) > 0:
        item = ds[split_name][0]
        print(f"  First item category: {item['category']}")
        print(f"  First item question: {item['question'][:100]}...")  # First 100 chars

# Method 4: Convert to pandas for easier manipulation (optional)
print("\n=== Method 4: Convert to pandas DataFrame ===")
try:
    import pandas as pd
    
    # Convert a specific split to pandas DataFrame
    df_context = ds['context'].to_pandas()
    print(f"Context DataFrame shape: {df_context.shape}")
    print(f"Columns: {df_context.columns.tolist()}")
    print(f"First few rows:")
    print(df_context[['category', 'question', 'answer']].head())
    
except ImportError:
    print("pandas not available, skipping DataFrame conversion")

# Method 5: Filter and select specific items
print("\n=== Method 5: Filter items by category ===")
for split_name in ['context', 'compatibility', 'configuration']:
    categories = [item['category'] for item in ds[split_name]]
    unique_categories = list(set(categories))
    print(f"Categories in {split_name}: {unique_categories}")
    
    # Filter items by specific category (example)
    if unique_categories:
        target_category = unique_categories[0]
        filtered_items = [item for item in ds[split_name] if item['category'] == target_category]
        print(f"  Items with category '{target_category}': {len(filtered_items)}")