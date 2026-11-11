"""Step 3: train the two YOLO models.

    python train.py --model both          # plate detector, then character reader
    python train.py --model plate
    python train.py --model char
    python train.py --model char --resume # continue an interrupted run

Finished models are copied to models/plate_detector.pt and models/char_reader.pt.
"""
import argparse
import shutil

import config


def train_one(name, weights, cfg_path, resume, **kwargs):
    from ultralytics import YOLO
    last = config.RUNS / name / "weights" / "last.pt"
    if resume and last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(weights).train(
            data=str(cfg_path), project=str(config.RUNS), name=name, exist_ok=True,
            device=config.get_device(), workers=config.WORKERS, seed=config.SEED,
            plots=True, **kwargs)
    return config.RUNS / name / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["plate", "char", "both"], default="both")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plate-epochs", type=int, default=config.PLATE_EPOCHS)
    ap.add_argument("--char-epochs", type=int, default=config.CHAR_EPOCHS)
    args = ap.parse_args()

    for f in ("plate.yaml", "char.yaml"):
        if not (config.CFG_DIR / f).exists():
            raise SystemExit("Config files missing. Run: python prepare_data.py")

    if config.get_device() == "cpu":
        print("WARNING: no GPU found. Plate detector is slow on CPU and the character reader "
              "can take many hours. Train on Google Colab instead and copy the .pt files "
              "into the models/ folder (see README).")

    config.MODELS.mkdir(exist_ok=True)

    if args.model in ("plate", "both"):
        best = train_one("plate_detector", config.PLATE_WEIGHTS, config.CFG_DIR / "plate.yaml",
                         args.resume, epochs=args.plate_epochs, imgsz=config.PLATE_IMGSZ,
                         batch=config.BATCH, patience=15)
        shutil.copy(best, config.MODELS / "plate_detector.pt")
        print("Saved", config.MODELS / "plate_detector.pt")

    if args.model in ("char", "both"):
        # fliplr=0: a mirrored Bengali character is a different (wrong) character
        best = train_one("char_reader", config.CHAR_WEIGHTS, config.CFG_DIR / "char.yaml",
                         args.resume, epochs=args.char_epochs, imgsz=config.CHAR_IMGSZ,
                         batch=config.BATCH, patience=25, fliplr=0.0, flipud=0.0,
                         close_mosaic=10)
        shutil.copy(best, config.MODELS / "char_reader.pt")
        en, bn = config.load_names()
        config.save_meta(en=en, bn=bn)
        print("Saved", config.MODELS / "char_reader.pt")


if __name__ == "__main__":      # this guard is required on Windows
    main()
