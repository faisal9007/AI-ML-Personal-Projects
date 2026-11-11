"""Step 1: get the dataset, fix the YOLO config files, print a summary.

    python prepare_data.py
"""
import shutil
from pathlib import Path

import pandas as pd
import yaml

import config
from textlogic import list_images


def ensure_dataset() -> Path:
    try:
        return config.dataset_root()
    except FileNotFoundError:
        pass
    import kagglehub
    print("Downloading the dataset with kagglehub (about 830 MB)...")
    src = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    print("Copying into", config.DATA_DIR)
    shutil.copytree(src, config.DATA_DIR, dirs_exist_ok=True)
    return config.dataset_root()


def make_cfg(task_dir: Path, out_name: str):
    # The yaml files shipped with the dataset use '../train/images', which YOLO cannot
    # resolve. Write fixed copies with absolute paths.
    cfg = yaml.safe_load(open(task_dir / "data.yaml", encoding="utf-8"))
    cfg.update(path=str(task_dir), train="train/images", val="val/images", test="test/images")
    config.CFG_DIR.mkdir(exist_ok=True)
    out = config.CFG_DIR / out_name
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def main():
    root = ensure_dataset()
    print("Dataset root:", root)

    for c in root.rglob("labels.cache"):      # stale caches from the author's machine
        c.unlink()

    plate_cfg = make_cfg(config.plate_dir(), "plate.yaml")
    char_cfg = make_cfg(config.char_dir(), "char.yaml")
    print("Wrote", plate_cfg)
    print("Wrote", char_cfg)

    rows = []
    for task, d in [("plate", config.plate_dir()), ("char", config.char_dir())]:
        for split in ["train", "val", "test"]:
            imgs = list_images(d / split / "images")
            lbls = list((d / split / "labels").glob("*.txt"))
            stems = {p.stem for p in imgs}
            no_label = sum(1 for p in imgs if not (d / split / "labels" / (p.stem + ".txt")).exists())
            orphan = sum(1 for l in lbls if l.stem not in stems)
            rows.append(dict(task=task, split=split, images=len(imgs), labels=len(lbls),
                             images_without_label=no_label, labels_without_image=orphan))
    print()
    print(pd.DataFrame(rows).to_string(index=False))

    en, bn = config.load_names()
    print(f"\nCharacter classes: {len(en)} | Bengali names loaded: {bn is not None}")


if __name__ == "__main__":
    main()
