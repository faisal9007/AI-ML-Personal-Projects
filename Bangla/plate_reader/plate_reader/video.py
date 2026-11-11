"""Run the reader on a video and log every plate with its timestamp.

    python video.py path/to/clip.mp4 --every 3
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

import config
from plate_reader import PlateReader


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=3, help="run the models on every Nth frame")
    ap.add_argument("--plate-conf", type=float, default=0.4)
    ap.add_argument("--char-conf", type=float, default=None)
    args = ap.parse_args()

    reader = PlateReader()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = config.ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    stem = Path(args.video).stem
    writer = cv2.VideoWriter(str(out_dir / f"{stem}_plates.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    log, i, last = [], 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.every == 0:
            last = reader.read(frame, args.plate_conf, args.char_conf)
            for r in last:
                if r["text"]:
                    log.append(dict(frame=i, time_s=round(i / fps, 2), plate=r["text"],
                                    plate_bn=r["text_bn"], confidence=round(r["char_conf"], 3)))
        writer.write(reader.annotate(frame, last))
        i += 1
    cap.release()
    writer.release()

    df = pd.DataFrame(log)
    df.to_csv(out_dir / f"{stem}_log.csv", index=False, encoding="utf-8-sig")
    print(f"{i} frames processed. Video and log saved in {out_dir}")
    if not df.empty:
        print(df.drop_duplicates("plate").to_string(index=False))


if __name__ == "__main__":
    main()
