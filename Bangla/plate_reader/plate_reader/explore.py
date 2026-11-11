"""Step 2 (optional): look at the data. Saves pictures into reports/.

    python explore.py
"""
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import pandas as pd

import config
from textlogic import list_images, read_yolo_labels

random.seed(config.SEED)


def draw_gt(img_path, label_dir):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    lbl = label_dir / (img_path.stem + ".txt")
    if lbl.exists():
        for c, xc, yc, bw, bh in read_yolo_labels(lbl):
            x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
            x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(1, w // 250))
    return img


def save_grid(images, path, cols=4, size=4):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * size, rows * size * 0.75))
    for i, im in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    config.REPORTS.mkdir(exist_ok=True)
    for name, d in [("plate_samples", config.plate_dir()), ("char_samples", config.char_dir())]:
        imgs = random.sample(list_images(d / "train" / "images"), 8)
        save_grid([draw_gt(p, d / "train" / "labels") for p in imgs], config.REPORTS / f"{name}.png")

    # class balance of the character reader
    en, _ = config.load_names()
    cnt = Counter()
    for t in (config.char_dir() / "train" / "labels").glob("*.txt"):
        for c, *_ in read_yolo_labels(t):
            cnt[c] += 1
    df = pd.DataFrame({"class": en, "train_instances": [cnt.get(i, 0) for i in range(len(en))]})
    df = df.sort_values("train_instances", ascending=False).reset_index(drop=True)
    df.to_csv(config.REPORTS / "char_class_counts.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(16, 4))
    plt.bar(df["class"][:50], df["train_instances"][:50])
    plt.xticks(rotation=90)
    plt.title("Top 50 classes by training instances")
    plt.tight_layout()
    plt.savefig(config.REPORTS / "char_class_balance.png", dpi=110)
    plt.close()

    rare = df[df["train_instances"] < 10]
    print(f"{len(rare)} classes have fewer than 10 training examples:")
    print(", ".join(f"{r['class']}({r['train_instances']})" for _, r in rare.iterrows()))
    print("Pictures saved in", config.REPORTS)


if __name__ == "__main__":
    main()
