"""Shared settings and paths for the Bangla license plate project."""
import json
import os
from pathlib import Path

import yaml

from textlogic import as_list

ROOT = Path(__file__).resolve().parent
CFG_DIR = ROOT / "configs"      # fixed YOLO yaml files (written by prepare_data.py)
RUNS = ROOT / "runs"            # YOLO training runs
MODELS = ROOT / "models"        # final plate_detector.pt, char_reader.pt, char_names.json
REPORTS = ROOT / "reports"      # evaluation output

KAGGLE_DATASET = "arstrxshad/banglabengali-license-plate-with-number-detection"

# ---------- Settings you can change ----------
PLATE_EPOCHS = 30
CHAR_EPOCHS = 80
PLATE_IMGSZ = 640
CHAR_IMGSZ = 416
BATCH = 16
PLATE_WEIGHTS = "yolov8n.pt"    # small and fast, plate detection is easy
CHAR_WEIGHTS = "yolov8s.pt"     # bigger, characters are small and there are 102 classes
WORKERS = 2                     # set to 0 if the dataloader crashes on Windows
SEED = 42


def _root_in(base: Path):
    """Return the folder that contains both task folders, looking one level deep."""
    if not base.is_dir():
        return None
    for d in [base] + sorted(c for c in base.iterdir() if c.is_dir()):
        if (d / "Vehicle_to_License_Plate").is_dir() and (d / "License_Plate_to_Text").is_dir():
            return d
    return None


def _pick_data_dir() -> Path:
    """Order: PLATE_DATA_DIR env var, ./data/license_plate, ../data/license_plate.
    The first one that already holds the dataset wins. Otherwise ./data/license_plate
    is where prepare_data.py will download it."""
    env = os.environ.get("PLATE_DATA_DIR")
    candidates = ([Path(env)] if env else []) + [
        ROOT / "data" / "license_plate",
        ROOT.parent / "data" / "license_plate",
    ]
    for c in candidates:
        if _root_in(c):
            return c
    return candidates[0]


DATA_DIR = _pick_data_dir()


def dataset_root() -> Path:
    r = _root_in(DATA_DIR)
    if r is None:
        raise FileNotFoundError(f"Dataset not found in {DATA_DIR}. Run: python prepare_data.py")
    return r


def plate_dir() -> Path:
    return dataset_root() / "Vehicle_to_License_Plate"


def char_dir() -> Path:
    return dataset_root() / "License_Plate_to_Text"


def get_device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def load_names():
    """Character class names: English list and optional Bengali list (same order)."""
    cd = char_dir()
    cfg = yaml.safe_load(open(cd / "data.yaml", encoding="utf-8"))
    en = as_list(cfg["names"])
    bn = None
    try:
        b = yaml.safe_load(open(cd / "data-bn.yaml", encoding="utf-8"))
        b = as_list(b["names"])
        if len(b) == len(en):
            bn = b
    except Exception:
        pass
    return en, bn


def load_meta() -> dict:
    p = MODELS / "char_names.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def save_meta(**updates):
    MODELS.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    meta.update(updates)
    with open(MODELS / "char_names.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
