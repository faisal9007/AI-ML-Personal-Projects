"""Pure helper functions (no YOLO needed): labels, character ordering, edit distance."""
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


def as_list(names):
    return list(names) if isinstance(names, list) else [names[i] for i in sorted(names)]


def read_yolo_labels(txt):
    """Returns (class, xc, yc, w, h) tuples, coordinates normalized 0-1."""
    out = []
    for line in open(txt, encoding="utf-8"):
        p = line.split()
        if len(p) >= 5:
            out.append((int(p[0]), *map(float, p[1:5])))
    return out


def order_characters(items, row_gap=0.6):
    """items: (xc, yc, height, class_id, ...). Returns rows, each sorted left to right.
    Two-line plates are split where the vertical gap between neighbours is large."""
    if not items:
        return []
    items = sorted(items, key=lambda t: t[1])
    mean_h = sum(t[2] for t in items) / len(items)
    rows = [[items[0]]]
    for prev, cur in zip(items, items[1:]):
        if cur[1] - prev[1] > row_gap * mean_h:
            rows.append([])
        rows[-1].append(cur)
    return [sorted(r, key=lambda t: t[0]) for r in rows]


def chars_to_text(items, names, sep=" | "):
    rows = order_characters(items)
    return sep.join(" ".join(names[t[3]] for t in r) for r in rows)


def chars_to_tokens(items, names):
    return [names[t[3]] for r in order_characters(items) for t in r]


def levenshtein(a, b):
    """Edit distance between two token lists."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
