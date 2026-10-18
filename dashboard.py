"""
Bottle inspection dashboard.

Usage:
    python dashboard.py --weights best.pt --source video.mp4
    python dashboard.py --weights best.pt --source video.mp4 --show --note "AI-generated demo footage"

Output: inspection_output.mp4 (video with a stats panel on the right)

Logic:
  - the model tracks whole bottles (class "bottle") with ByteTrack
  - cap and fill boxes whose centre lies inside a bottle box belong to that bottle
  - every bottle collects confidence-weighted votes over its whole track:
        cap  group: Capped / No Cap
        fill group: Normal / Overfilled / Underfilled
  - a bottle is DEFECTIVE only if a defect class WINS its group vote, so a few
    wrong frames do not flip a bottle that was normal for most of its track
  - defect classes need a higher confidence (--defect-conf) to be allowed to vote
  - bottles clipped by the left/right frame edge are not judged (grey box)
"""
import argparse
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

# BGR colours
GREEN = (90, 200, 70)
RED = (40, 40, 235)
AMBER = (0, 175, 255)
WHITE = (255, 255, 255)
GRAY = (165, 165, 165)
BG = (30, 28, 26)
CARD = (48, 45, 42)

CAP = {"capped", "no cap"}
FILL = {"normal", "overfilled", "underfilled"}
BAD_PARTS = {"no cap", "overfilled", "underfilled"}
SHOW_NAME = {"no cap": "No Cap", "overfilled": "Overfilled", "underfilled": "Underfilled"}

MIN_FRAMES = 4     # judged frames before a bottle is counted / can be called defective
MIN_SCORE = 2.0    # summed confidence a defect needs while winning its vote (~3 confident frames)
EDGE_MARGIN = 20   # px: bottles touching the left/right edge are not judged


class State:
    def __init__(self):
        self.seen = Counter()
        self.cap = defaultdict(Counter)
        self.fill = defaultdict(Counter)

    def verdict(self, tid):
        """Confirmed defect type for a tracked bottle, or None."""
        if tid is None or self.seen[tid] < MIN_FRAMES:
            return None
        for votes in (self.cap[tid], self.fill[tid]):
            if votes:
                name, score = votes.most_common(1)[0]
                if name in BAD_PARTS and score >= MIN_SCORE:
                    return name
        return None


def text(img, s, org, scale, color=WHITE, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def center_in(box, outer):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def compute_stats(state):
    counted = [t for t, n in state.seen.items() if n >= MIN_FRAMES]
    types = Counter()
    for t in counted:
        d = state.verdict(t)
        if d:
            types[d] += 1
    defective = sum(types.values())
    total = len(counted)
    return {
        "total": total,
        "ok": total - defective,
        "defective": defective,
        "rate": 100 * defective / total if total else 0.0,
        "types": types,
    }


def draw_ok(img, box, k):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, max(2, int(2 * k)))
    text(img, "OK", (x1 + 6, y1 + int(24 * k)), 0.7 * k, GREEN, 2)


def draw_neutral(img, box, k):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, max(1, int(2 * k)))


def draw_defect(img, box, label, k):
    x1, y1, x2, y2 = [int(v) for v in box]
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), RED, -1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), RED, max(4, int(6 * k)))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, 2)
    y0 = max(0, y1 - th - int(18 * k))
    cv2.rectangle(img, (x1, y0), (x1 + tw + 14, y0 + th + int(18 * k)), RED, -1)
    text(img, label, (x1 + 7, y0 + th + int(6 * k)), 0.8 * k, WHITE, 2)


def process(frame, bottles, parts, state, k, defect_conf):
    """bottles: [(track_id, box)], parts: [(name, box, conf)].
    Draws on frame, updates votes, returns True if a defective bottle is visible now."""
    W = frame.shape[1]
    defect_now = False
    for tid, bb in bottles:
        if bb[0] <= EDGE_MARGIN or bb[2] >= W - EDGE_MARGIN:
            draw_neutral(frame, bb, k)  # clipped by the frame edge: not judged
            continue
        inside = [(n, pb, c) for n, pb, c in parts if center_in(pb, bb)]
        if tid is not None:
            state.seen[tid] += 1
            for group, store in ((CAP, state.cap[tid]), (FILL, state.fill[tid])):
                cand = [(c, n) for n, pb, c in inside
                        if n in group and (n not in BAD_PARTS or c >= defect_conf)]
                if cand:  # best box of this group in this frame
                    c, n = max(cand)
                    store[n] += c
        d = state.verdict(tid)
        if d:
            defect_now = True
            draw_defect(frame, bb, "DEFECT: " + SHOW_NAME[d], k)
            for n, pb, c in inside:
                if n == d:
                    x1, y1, x2, y2 = [int(v) for v in pb]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), AMBER, max(2, int(3 * k)))
                    text(frame, SHOW_NAME[n], (x1 + 4, y2 - 8), 0.55 * k, AMBER, 2)
        else:
            draw_ok(frame, bb, k)
    return defect_now


def card(p, x, y, w, h, title, value, color, k):
    cv2.rectangle(p, (x, y), (x + w, y + h), CARD, -1)
    text(p, title, (x + int(12 * k), y + int(26 * k)), 0.5 * k, GRAY, 1)
    text(p, str(value), (x + int(12 * k), y + h - int(16 * k)), 1.2 * k, color, 2)


def build_panel(H, k, stats, defect_now, fps, frame_i, total, note):
    PW = int(360 * k)
    p = np.full((H, PW, 3), BG, np.uint8)
    pad = int(16 * k)
    text(p, "BOTTLE INSPECTION", (pad, int(38 * k)), 0.8 * k, WHITE, 2)

    # status banner
    y = int(60 * k)
    h = int(84 * k)
    cv2.rectangle(p, (pad, y), (PW - pad, y + h), RED if defect_now else GREEN, -1)
    msg = "DEFECT DETECTED" if defect_now else "ALL OK"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * k, 2)
    text(p, msg, ((PW - tw) // 2, y + (h + th) // 2), 0.9 * k, WHITE, 2)

    # stat cards
    cw = (PW - 3 * pad) // 2
    ch = int(88 * k)
    y1 = y + h + pad
    y2 = y1 + ch + pad
    card(p, pad, y1, cw, ch, "BOTTLES", stats["total"], WHITE, k)
    card(p, 2 * pad + cw, y1, cw, ch, "OK", stats["ok"], GREEN, k)
    card(p, pad, y2, cw, ch, "DEFECTIVE", stats["defective"], RED, k)
    card(p, 2 * pad + cw, y2, cw, ch, "DEFECT RATE", f"{stats['rate']:.0f}%", AMBER, k)

    # footer
    yf = H - int(14 * k)
    if note:
        text(p, note, (pad, yf - int(26 * k)), 0.45 * k, GRAY, 1)
    text(p, f"FPS {fps:4.1f}   frame {frame_i}/{total}", (pad, yf), 0.5 * k, GRAY, 1)
    return p


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--source", default="video.mp4")
    ap.add_argument("--out", default="inspection_output.mp4")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35, help="general detection threshold")
    ap.add_argument("--defect-conf", type=float, default=0.55,
                    help="a defect class must reach this confidence to vote")
    ap.add_argument("--show", action="store_true", help="show a live window (q to quit)")
    ap.add_argument("--note", default="", help="footer text, e.g. 'AI-generated demo footage'")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = {int(i): n.lower() for i, n in model.names.items()}
    if "bottle" not in names.values():
        raise SystemExit(f"Model has no 'bottle' class. Classes: {list(names.values())}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    k = H / 720
    PW = int(360 * k)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W + PW, H))

    state = State()
    fps, t_prev, i = 0.0, time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        bottles, parts = [], []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else [None] * len(cls))
            for b, c, cf, t in zip(xyxy, cls, conf, ids):
                n = names[int(c)]
                if n == "bottle":
                    bottles.append((t, b))
                else:
                    parts.append((n, b, float(cf)))

        defect_now = process(frame, bottles, parts, state, k, args.defect_conf)

        now = time.time()
        dt = max(now - t_prev, 1e-6)
        t_prev = now
        fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

        panel = build_panel(H, k, compute_stats(state), defect_now, fps, i, total, args.note)
        if defect_now:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), RED, max(6, int(10 * k)))
        out = np.hstack([frame, panel])
        writer.write(out)

        if args.show:
            cv2.imshow("Bottle inspection", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    s = compute_stats(state)
    print(f"Saved {args.out} | bottles {s['total']} | ok {s['ok']} | "
          f"defective {s['defective']} {dict(s['types'])}")
    print("Votes per defective bottle (track id: cap votes | fill votes | judged frames):")
    for t in sorted(state.seen):
        if state.verdict(t):
            print(f"  id {t}: {dict(state.cap[t])} | {dict(state.fill[t])} | {state.seen[t]}")


if __name__ == "__main__":
    main()
