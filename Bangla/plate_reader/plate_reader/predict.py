"""Read plates from one image or a whole folder.

    python predict.py path/to/car.jpg
    python predict.py path/to/folder --plate-conf 0.3

Annotated images and plates.csv go to outputs/predictions/.
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

import config
from plate_reader import PlateReader
from textlogic import list_images


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--plate-conf", type=float, default=0.4)
    ap.add_argument("--char-conf", type=float, default=None)
    ap.add_argument("--out", default=str(config.ROOT / "outputs" / "predictions"))
    args = ap.parse_args()

    src = Path(args.source)
    paths = [src] if src.is_file() else list_images(src)
    if not paths:
        raise SystemExit(f"No images found in {src}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reader = PlateReader()
    rows = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print("Could not read", p)
            continue
        res = reader.read(img, args.plate_conf, args.char_conf)
        cv2.imwrite(str(out / p.name), reader.annotate(img, res))
        if not res:
            print(f"{p.name}: no plate found")
        for r in res:
            print(f"{p.name}: {r['text_bn'] or r['text']}   ({r['text']})")
            rows.append(dict(image=p.name, plate=r["text"], plate_bn=r["text_bn"],
                             plate_conf=round(r["plate_conf"], 3),
                             char_conf=round(r["char_conf"], 3)))
    pd.DataFrame(rows).to_csv(out / "plates.csv", index=False, encoding="utf-8-sig")
    print("Saved to", out)


if __name__ == "__main__":
    main()
