"""The two-stage reader: vehicle photo -> plate box -> character boxes -> text."""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config
from textlogic import chars_to_text


class PlateReader:
    def __init__(self, plate_weights=None, char_weights=None, device=None):
        plate_weights = Path(plate_weights or config.MODELS / "plate_detector.pt")
        char_weights = Path(char_weights or config.MODELS / "char_reader.pt")
        for p in (plate_weights, char_weights):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Train with 'python train.py' or copy the .pt files "
                    f"from Colab into {config.MODELS}")
        self.device = config.get_device() if device is None else device
        self.plate_model = YOLO(str(plate_weights))
        self.char_model = YOLO(str(char_weights))

        meta = config.load_meta()
        self.names_bn = meta.get("bn")
        if self.names_bn is None:
            try:
                self.names_bn = config.load_names()[1]
            except Exception:
                pass
        self.char_conf_default = float(meta.get("best_char_conf", 0.25))

    def read(self, img_bgr, plate_conf=0.4, char_conf=None, pad=0.05):
        """Run on one BGR image. Returns one dict per detected plate."""
        char_conf = self.char_conf_default if char_conf is None else char_conf
        H, W = img_bgr.shape[:2]
        en = self.char_model.names
        out = []
        res = self.plate_model.predict(img_bgr, conf=plate_conf, imgsz=config.PLATE_IMGSZ,
                                       device=self.device, verbose=False)[0]
        for box, pc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            cx2, cy2 = int(min(W, x2 + px)), int(min(H, y2 + py))
            crop = img_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            cres = self.char_model.predict(crop, conf=char_conf, imgsz=config.CHAR_IMGSZ,
                                           device=self.device, verbose=False)[0]
            items = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1], int(c), float(cf))
                     for b, c, cf in zip(cres.boxes.xyxy.cpu().numpy(),
                                         cres.boxes.cls.cpu().numpy(),
                                         cres.boxes.conf.cpu().numpy())]
            out.append(dict(
                box=(int(x1), int(y1), int(x2), int(y2)),
                plate_conf=float(pc),
                text=chars_to_text(items, en),
                text_bn=chars_to_text(items, self.names_bn) if self.names_bn else None,
                char_conf=float(np.mean([t[4] for t in items])) if items else 0.0,
                n_chars=len(items),
                crop=crop))
        return out

    @staticmethod
    def annotate(img_bgr, results):
        """OpenCV cannot draw Bengali glyphs, so the picture shows English transliteration.
        Bengali text is returned in result['text_bn'] for the app to display."""
        out = img_bgr.copy()
        W = out.shape[1]
        th = max(2, W // 400)
        scale = max(0.6, W / 1200)
        for r in results:
            x1, y1, x2, y2 = r["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), th)
            label = r["text"] or "?"
            pos = (x1, max(20, y1 - 8))
            cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), th, cv2.LINE_AA)
        return out
