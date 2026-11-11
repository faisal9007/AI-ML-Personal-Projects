import importlib
import subprocess
import sys

try:
	kagglehub = importlib.import_module("kagglehub")
except ModuleNotFoundError:
	subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
	kagglehub = importlib.import_module("kagglehub")

# Download latest version
path = kagglehub.dataset_download("arstrxshad/banglabengali-license-plate-with-number-detection")

print("Path to dataset files:", path)

import os

# See what's inside
for root, dirs, files in os.walk(path):
    level = root.replace(path, "").count(os.sep)
    print("  " * level + os.path.basename(root) + "/")
    if level >= 2:   # don't go too deep
        continue
    for f in files[:5]:
        print("  " * (level + 1) + f)

import shutil
from pathlib import Path

dest = Path(r"D:\Python\Personal Projects\Bangla\data\license_plate")

if not dest.exists():
    shutil.copytree(path, dest)
    print("Copied to:", dest)
else:
    print("Already exists:", dest)

DATA_DIR = dest

import os
from pathlib import Path

path = r"C:\Users\Faisal\.cache\kagglehub\datasets\arstrxshad\banglabengali-license-plate-with-number-detection\versions\1"

# Count files by extension
from collections import Counter
counts = Counter(p.suffix.lower() for p in Path(path).rglob("*") if p.is_file())
print(counts)

# Show the folder tree (2 levels)
for root, dirs, files in os.walk(path):
    depth = root.replace(path, "").count(os.sep)
    if depth > 2:
        continue
    print("  " * depth + os.path.basename(root) + "/", f"({len(files)} files)")
    for f in files[:3]:
        print("  " * (depth + 1) + f)

from pathlib import Path
import yaml

root = Path(r"D:\Python\Personal Projects\Bangla\data\license_plate")

for task in ["Vehicle_to_License_Plate", "License_Plate_to_Text"]:
    print("\n==", task)
    for split in ["train", "val", "test"]:
        imgs = list((root / task / split).rglob("*.jpg")) + \
               list((root / task / split).rglob("*.png"))
        lbls = list((root / task / split).rglob("*.txt"))
        print(f"{split:5} images={len(imgs):6}  labels={len(lbls):6}")

# Class names and paths
for y in [root / "Vehicle_to_License_Plate" / "data.yaml",
          root / "License_Plate_to_Text" / "data.yaml"]:
    print("\n--", y)
    print(yaml.safe_load(open(y, encoding="utf-8")))