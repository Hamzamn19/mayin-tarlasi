import os
import random
import shutil
from pathlib import Path

def split_data():
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "landmine_flat")
    
    # Ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    # TEST_RATIO will be the remainder (0.1)

    print(f"Analyzing files in {DATA_DIR}...")
    
    # Get all unique base names (files that have both .jpg and .xml)
    files = os.listdir(DATA_DIR)
    jpg_files = {f.replace(".jpg", "") for f in files if f.endswith(".jpg")}
    xml_files = {f.replace(".xml", "") for f in files if f.endswith(".xml")}
    
    # Only keep files that have both counterparts
    base_names = list(jpg_files.intersection(xml_files))
    print(f"Found {len(base_names)} valid image-xml pairs.")
    
    if not base_names:
        print("No valid pairs found! Checking for images only...")
        base_names = list(jpg_files)
        if not base_names:
            print("No images found. Exiting.")
            return

    # Shuffle
    random.seed(42)
    random.shuffle(base_names)
    
    # Calculate split indices
    total = len(base_names)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits = {
        "train": base_names[:train_end],
        "val": base_names[train_end:val_end],
        "test": base_names[val_end:]
    }
    
    # Create directories
    for split_name in splits:
        os.makedirs(os.path.join(DATA_DIR, split_name), exist_ok=True)
    
    # Move files
    print("Moving files...")
    for split_name, names in splits.items():
        print(f"  Moving {len(names)} pairs to {split_name}...")
        for name in names:
            # Move JPG
            src_jpg = os.path.join(DATA_DIR, f"{name}.jpg")
            dst_jpg = os.path.join(DATA_DIR, split_name, f"{name}.jpg")
            if os.path.exists(src_jpg):
                shutil.move(src_jpg, dst_jpg)
            
            # Move XML
            src_xml = os.path.join(DATA_DIR, f"{name}.xml")
            dst_xml = os.path.join(DATA_DIR, split_name, f"{name}.xml")
            if os.path.exists(src_xml):
                shutil.move(src_xml, dst_xml)
                
    print("\nData split complete!")
    print(f"Train: {len(splits['train'])}")
    print(f"Val:   {len(splits['val'])}")
    print(f"Test:  {len(splits['test'])}")

if __name__ == "__main__":
    split_data()
