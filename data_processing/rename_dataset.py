import os
import shutil
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.join(PROJECT_ROOT, "landmine_final - Copy")
output_dir = os.path.join(PROJECT_ROOT, "landmine_flat")

print(f"Creating flattened dataset in {output_dir}...")
os.makedirs(output_dir, exist_ok=True)

copied_count = 0

for xml_path in Path(base_dir).rglob("*.xml"):
    jpg_path = xml_path.with_suffix(".jpg")
    
    if not jpg_path.exists():
        continue
        
    # Create a unique name based on the relative path
    rel_path = xml_path.relative_to(base_dir)
    # Replace slashes with underscores to make it a flat filename
    unique_name_base = str(rel_path).replace(os.sep, "_").replace(" ", "_")
    
    unique_xml_name = unique_name_base
    unique_jpg_name = unique_name_base.replace(".xml", ".jpg")
    
    shutil.copy2(xml_path, os.path.join(output_dir, unique_xml_name))
    shutil.copy2(jpg_path, os.path.join(output_dir, unique_jpg_name))
    
    copied_count += 1
    if copied_count % 1000 == 0:
        print(f"  Copied {copied_count} files...")

print(f"Done! Successfully flattened and uniquely renamed {copied_count} image/xml pairs.")
