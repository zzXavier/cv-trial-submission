"""Create targets.json by drawing a box around the target player on frame 0.

For each clip in clips/ (or just the ones you name), frame 0 opens in a window:
drag a box tightly around the target player's whole body and press ENTER
(press c to cancel and skip that clip). Existing entries are kept unless you
re-draw that clip.

Usage:
    python make_targets.py                     # every clips/*.mp4
    python make_targets.py clips/clip_a.mp4    # just one clip
"""
import json
import sys
from pathlib import Path

import cv2

OUT = Path("targets.json")


def main():
    clips = [Path(p) for p in sys.argv[1:]] or sorted(Path("clips").glob("*.mp4"))
    if not clips:
        raise SystemExit("no clips found -- put the .mp4 files in clips/ first")
    targets = json.loads(OUT.read_text()) if OUT.exists() else {}

    for clip in clips:
        ok, frame = cv2.VideoCapture(str(clip)).read()
        if not ok:
            raise SystemExit(f"cannot read frame 0 of {clip}")
        x, y, w, h = cv2.selectROI(f"{clip.stem}: box the target, then ENTER", frame)
        cv2.destroyAllWindows()
        if w and h:
            targets[clip.stem] = {"frame_index": 0, "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            print(f"{clip.stem}: {targets[clip.stem]}")
        else:
            print(f"{clip.stem}: skipped")

    OUT.write_text(json.dumps(targets, indent=2) + "\n")
    print(f"wrote {OUT} ({len(targets)} clip(s))")


if __name__ == "__main__":
    main()
