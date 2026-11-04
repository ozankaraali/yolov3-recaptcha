import cv2
import pytest

from detector import TileFinder, coco_class, coverage, find_grid


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("a fire hydrant", "fire hydrant"),
        ("Fire Hydrants", "fire hydrant"),
        ("buses", "bus"),
        ("a bus", "bus"),
        ("traffic lights", "traffic light"),
        ("taxis", "car"),
        ("crosswalks", None),
        ("mountains or hills", None),
    ],
)
def test_coco_class(prompt, expected):
    assert coco_class(prompt) == expected


def test_coverage():
    square = (0, 0, 100, 100)
    assert coverage(square, (0, 0, 100, 100, 0.9)) == 1.0
    assert coverage(square, (50, 50, 200, 200, 0.9)) == 0.25
    assert coverage(square, (100, 0, 200, 100, 0.9)) == 0.0


@pytest.fixture(scope="module")
def finder():
    return TileFinder()


@pytest.mark.parametrize(
    ("path", "target", "tiles"),
    [("docs/find.png", "fire hydrant", [3, 4, 5]), ("docs/find2.png", "bus", [0, 1, 5])],
)
def test_bundled_screenshots(finder, path, target, tiles):
    image = cv2.imread(path)
    rects = find_grid(image)
    assert len(rects) == 9
    picked, boxes = finder.pick(image, rects, target, separate_images=True)
    assert picked == tiles
    assert len(boxes) >= len(tiles)


def test_one_photo_in_squares(finder):
    # Treat the red-bus tile (index 5) as one photo cut into a 4x4 grid. The bus spans the middle
    # two rows and the right three columns: squares 5, 6, 7 and 9, 10, 11.
    image = cv2.imread("docs/find2.png")
    x0, y0, x1, y1 = find_grid(image)[5]
    photo = image[y0:y1, x0:x1]
    side = photo.shape[0] // 4
    squares = [(c * side, r * side, (c + 1) * side, (r + 1) * side) for r in range(4) for c in range(4)]
    picked, boxes = finder.pick(photo, squares, "bus", separate_images=False)
    assert boxes
    assert picked == [5, 6, 7, 9, 10, 11]
