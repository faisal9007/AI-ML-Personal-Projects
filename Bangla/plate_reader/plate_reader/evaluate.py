"""Step 4: measure both models. Writes everything into reports/.

    python evaluate.py              # uses the test split
    python evaluate.py --split val

Plate detector: mAP, precision, recall.
Character reader: mAP, weakest classes, exact-match accuracy, token error rate,
best confidence threshold, and a picture of the wrong readings.
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import pandas as pd

import config
from plate_reader import PlateReader
from textlogic import (chars_to_text, chars_to_tokens, levenshtein, list_images,
                       read_yolo_labels)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    reader = PlateReader()
    dev = reader.device
    out = config.REPORTS
    out.mkdir(exist_ok=True)
    metrics = {}

    # ---- 1. plate detector ----
    m1 = reader.plate_model.val(
        data=str(config.CFG_DIR / "plate.yaml"), split=args.split, imgsz=config.PLATE_IMGSZ,
        device=dev, project=str(config.RUNS / "eval"), name=f"plate_{args.split}", exist_ok=True)
    metrics["plate_detector"] = dict(map50=float(m1.box.map50), map50_95=float(m1.box.map),
                                     precision=float(m1.box.mp), recall=float(m1.box.mr))

    # ---- 2. character reader (YOLO metrics) ----
    m2 = reader.char_model.val(
        data=str(config.CFG_DIR / "char.yaml"), split=args.split, imgsz=config.CHAR_IMGSZ,
        device=dev, plots=True, project=str(config.RUNS / "eval"),
        name=f"char_{args.split}", exist_ok=True)
    metrics["char_reader"] = dict(map50=float(m2.box.map50), map50_95=float(m2.box.map),
                                  precision=float(m2.box.mp), recall=float(m2.box.mr))
    names = reader.char_model.names
    per_class = pd.DataFrame({"class": [names[i] for i in m2.box.ap_class_index],
                              "AP50": m2.box.ap50, "AP50-95": m2.box.ap}).sort_values("AP50")
    per_class.to_csv(out / "char_per_class.csv", index=False, encoding="utf-8-sig")
    print("\nWeakest 15 character classes:")
    print(per_class.head(15).to_string(index=False))

    # ---- 3. plate-text accuracy on the crops ----
    ldir = config.char_dir() / args.split / "labels"
    paths = [p for p in list_images(config.char_dir() / args.split / "images")
             if (ldir / (p.stem + ".txt")).exists()]
    gt = {p: [(xc, yc, h, c) for c, xc, yc, w, h in read_yolo_labels(ldir / (p.stem + ".txt"))]
          for p in paths}

    raw = {}   # predict once at a low threshold, filter afterwards
    stream = reader.char_model.predict([str(p) for p in paths], conf=0.05,
                                       imgsz=config.CHAR_IMGSZ, device=dev,
                                       stream=True, verbose=False, batch=32)
    for p, r in zip(paths, stream):
        raw[p] = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1], int(c), float(cf))
                  for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                      r.boxes.conf.cpu().numpy())]

    def evaluate_at(conf):
        exact = dist = total = 0
        for p in paths:
            g = chars_to_tokens(gt[p], names)
            pr = chars_to_tokens([t for t in raw[p] if t[4] >= conf], names)
            exact += (g == pr)
            dist += levenshtein(g, pr)
            total += len(g)
        return exact / len(paths), dist / max(1, total)

    sweep = []
    for conf in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        em, ter = evaluate_at(conf)
        sweep.append(dict(char_conf=conf, exact_match=round(em, 4), token_error_rate=round(ter, 4)))
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out / "char_conf_sweep.csv", index=False)
    print("\nPlate-text accuracy by character confidence threshold:")
    print(sweep_df.to_string(index=False))

    best = float(sweep_df.sort_values("exact_match", ascending=False).iloc[0]["char_conf"])
    metrics["plate_text"] = dict(split=args.split, plates=len(paths), best_char_conf=best,
                                 exact_match=float(sweep_df["exact_match"].max()))
    config.save_meta(best_char_conf=best)
    print("Best char_conf:", best, "(saved, apps use it as the default)")

    # ---- 4. mistakes ----
    errors = []
    for p in paths:
        g = chars_to_text(gt[p], names)
        pr = chars_to_text([t for t in raw[p] if t[4] >= best], names)
        if g != pr:
            errors.append(dict(image=p.name, ground_truth=g, predicted=pr, path=str(p)))
    pd.DataFrame(errors).drop(columns="path", errors="ignore").to_csv(
        out / "char_errors.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(errors)} wrong out of {len(paths)} plates")

    if errors:
        shown = errors[:12]
        plt.figure(figsize=(15, 3.2 * ((len(shown) + 2) // 3)))
        for i, e in enumerate(shown):
            plt.subplot((len(shown) + 2) // 3, 3, i + 1)
            plt.imshow(cv2.cvtColor(cv2.imread(e["path"]), cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"GT:   {e['ground_truth']}\nPRED: {e['predicted']}", fontsize=8)
        plt.tight_layout()
        plt.savefig(out / "char_errors.png", dpi=110)
        plt.close()

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\nReports saved in", out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
