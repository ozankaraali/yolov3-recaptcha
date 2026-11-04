"""Offline demo: mark the tiles to click on a saved screenshot of a 3x3 reCAPTCHA challenge.

    uv run image_demo.py                       # the two bundled screenshots in docs/
    uv run image_demo.py shot.png "traffic lights"

Results are written to out/.
"""

import sys
from pathlib import Path

import cv2

from detector import TileFinder, coco_class, draw, find_grid

EXAMPLES = [("docs/find.png", "a fire hydrant"), ("docs/find2.png", "a bus")]
OUT = Path("out")


def main() -> None:
    if len(sys.argv) == 3:
        jobs = [(sys.argv[1], sys.argv[2])]
    elif len(sys.argv) == 1:
        jobs = EXAMPLES
    else:
        sys.exit(__doc__)

    finder = TileFinder()
    OUT.mkdir(exist_ok=True)
    for path, prompt in jobs:
        target = coco_class(prompt)
        if target is None:
            print(f"{path}: '{prompt}' is not a COCO class, skipping")
            continue
        image = cv2.imread(path)
        if image is None:
            sys.exit(f"{path}: cannot read image")
        rects = find_grid(image)
        if len(rects) != 9:
            sys.exit(f"{path}: expected a 3x3 grid, found {len(rects)} tiles")
        picked, boxes = finder.pick(image, rects, target, separate_images=True)
        out = OUT / Path(path).name
        cv2.imwrite(str(out), draw(image, rects, picked, boxes, target))
        print(f"{path}: {target} in tiles {picked} (row by row, from 0) -> {out}")


if __name__ == "__main__":
    main()
