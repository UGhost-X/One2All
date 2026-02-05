import os
import shutil
from pathlib import Path

def setup_dataset():
    # Base path
    base_dir = Path("/home/software/One2All")
    src_dir = base_dir / "datasets/ele_type"
    dst_dir = base_dir / "datasets/ele_type_cls"
    
    classes = ["type-1", "type-2", "type-3"]
    splits = ["train", "val"]
    
    for split in splits:
        for cls in classes:
            path = dst_dir / split / cls
            path.mkdir(parents=True, exist_ok=True)
            
            src_file = src_dir / f"{cls}.jpg"
            if src_file.exists():
                shutil.copy(src_file, path / f"{cls}.jpg")
                print(f"Copied {src_file} to {path}")

if __name__ == "__main__":
    setup_dataset()
