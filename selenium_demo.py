"""Live demo: solve the image challenge on Google's reCAPTCHA demo page with YOLO26 and Selenium.

    uv run selenium_demo.py

Needs Google Chrome; Selenium Manager downloads the matching ChromeDriver. Each screenshot the
solver acts on is saved to out/ with the detections and the clicked tiles marked. See the README
for why this rarely gets through in 2026.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from detector import Rect, TileFinder, coco_class, draw

DEMO_URL = "https://www.google.com/recaptcha/api2/demo"
ANCHOR_FRAME = (By.CSS_SELECTOR, "iframe[src*='/recaptcha/api2/anchor']")
CHALLENGE_FRAME = (By.CSS_SELECTOR, "iframe[src*='/recaptcha/api2/bframe']")
GRID = (By.CSS_SELECTOR, "#rc-imageselect-target table")
MAX_ROUNDS = 10
MAX_REFILLS = 10  # replacement waves in one "click verify once there are none left" round
OUT = Path("out")

# Errors that leave the same challenge on screen; reload instead of retrying it unchanged.
RETRY_ERRORS = (
    ".rc-imageselect-error-select-more",
    ".rc-imageselect-error-dynamic-more",
    ".rc-imageselect-error-select-something",
)


class Solver:
    def __init__(self, driver: webdriver.Chrome, finder: TileFinder):
        self.driver = driver
        self.finder = finder
        self.wait = WebDriverWait(driver, 15)

    def enter(self, frame: tuple[str, str]) -> None:
        self.driver.switch_to.default_content()
        self.wait.until(EC.frame_to_be_available_and_switch_to_it(frame))

    def visible(self, css: str) -> bool:
        return any(e.is_displayed() for e in self.driver.find_elements(By.CSS_SELECTOR, css))

    def solved(self) -> bool:
        self.enter(ANCHOR_FRAME)
        return self.driver.find_element(By.ID, "recaptcha-anchor").get_attribute("aria-checked") == "true"

    def run(self) -> bool:
        self.driver.get(DEMO_URL)
        self.enter(ANCHOR_FRAME)
        self.wait.until(EC.element_to_be_clickable((By.ID, "recaptcha-anchor"))).click()
        time.sleep(2)

        for round_no in range(1, MAX_ROUNDS + 1):
            if self.solved():
                print("The checkbox is ticked.")
                return self.submit()
            self.enter(CHALLENGE_FRAME)
            if self.visible(".rc-doscaptcha-header"):
                print("Blocked: reCAPTCHA says the computer or network may be sending automated queries.")
                return False
            try:
                self.solve_round(round_no)
            except TimeoutException:
                if self.solved():
                    continue
                print("No image challenge appeared (audio-only or blocked?).")
                return False
            time.sleep(3)
            self.enter(CHALLENGE_FRAME)
            for css in RETRY_ERRORS:
                if self.visible(css):
                    print(f"  reCAPTCHA: {self.driver.find_element(By.CSS_SELECTOR, css).text!r}; reloading")
                    self.driver.find_element(By.ID, "recaptcha-reload-button").click()
                    time.sleep(2)
                    break

        print(f"Gave up after {MAX_ROUNDS} rounds.")
        return False

    def tiles(self) -> tuple[WebElement, list[WebElement]]:
        grid = self.wait.until(EC.visibility_of_element_located(GRID))
        return grid, grid.find_elements(By.CSS_SELECTOR, "td.rc-imageselect-tile")

    def snapshot(self) -> tuple[np.ndarray, list[Rect], list[WebElement]]:
        """Screenshot the grid; return it with each tile's rectangle in screenshot pixels."""
        grid, tiles = self.tiles()
        canvas = cv2.imdecode(np.frombuffer(grid.screenshot_as_png, np.uint8), cv2.IMREAD_COLOR)
        g = grid.rect
        scale = canvas.shape[1] / g["width"]  # CSS pixels -> screenshot pixels (device pixel ratio)
        rects = []
        for tile in tiles:
            t = tile.rect
            x0, y0 = t["x"] - g["x"], t["y"] - g["y"]
            rects.append(tuple(round(v * scale) for v in (x0, y0, x0 + t["width"], y0 + t["height"])))
        return canvas, rects, tiles

    def solve_round(self, round_no: int) -> None:
        grid, _ = self.tiles()
        time.sleep(1)  # tiles fade in
        description = self.driver.find_element(By.CSS_SELECTOR, ".rc-imageselect-desc-wrapper")
        prompt = description.find_element(By.TAG_NAME, "strong").text
        refills = "none left" in description.text.lower()
        # 3x3 grids show nine separate photos; 4x4 grids cut one photo into squares.
        separate = "rc-imageselect-table-44" not in grid.get_attribute("class")
        target = coco_class(prompt)
        print(f"Round {round_no}: {prompt!r} ({'separate photos' if separate else 'one photo in squares'})")
        if target is None:
            print("  not a COCO class; asking for another challenge")
            self.driver.find_element(By.ID, "recaptcha-reload-button").click()
            return

        canvas, rects, tiles = self.snapshot()
        picked, boxes = self.finder.pick(canvas, rects, target, separate)
        self.save(f"round{round_no:02d}", canvas, rects, picked, boxes, target)
        for wave in range(1, MAX_REFILLS + 1):
            for i in picked:
                tiles[i].click()
                time.sleep(0.3)
            if not (refills and picked):
                break
            # Clicked tiles fade out and get a new photo; check only those.
            self.wait_for_refill(picked)
            canvas, rects, tiles = self.snapshot()
            picked, boxes = self.finder.pick(canvas, rects, target, True, only=picked)
            self.save(f"round{round_no:02d}-refill{wave}", canvas, rects, picked, boxes, target)

        verify = self.driver.find_element(By.ID, "recaptcha-verify-button")
        print(f"  pressing {verify.text!r}")
        verify.click()

    def wait_for_refill(self, indices: list[int]) -> None:
        def refilled(_) -> bool:
            _, tiles = self.tiles()
            return not any("rc-imageselect-dynamic-selected" in tiles[i].get_attribute("class") for i in indices)

        try:
            WebDriverWait(self.driver, 20, ignored_exceptions=[StaleElementReferenceException]).until(refilled)
        except TimeoutException:
            pass
        time.sleep(1)  # new photos fade in

    def save(self, name: str, canvas: np.ndarray, rects: list[Rect], picked: list[int], boxes, target: str) -> None:
        OUT.mkdir(exist_ok=True)
        path = OUT / f"{name}.png"
        cv2.imwrite(str(path), draw(canvas, rects, picked, boxes, target))
        print(f"  {target} in tiles {picked} -> {path}")

    def submit(self) -> bool:
        self.driver.switch_to.default_content()
        self.driver.find_element(By.ID, "recaptcha-demo-submit").click()
        try:
            print(self.wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "recaptcha-success"))).text)
            return True
        except TimeoutException:
            print("Submitted, but the demo page did not confirm success.")
            return False


def main() -> None:
    finder = TileFinder()
    driver = webdriver.Chrome()
    try:
        ok = Solver(driver, finder).run()
        if sys.stdin.isatty():
            input("Press Enter to close the browser...")
    finally:
        driver.quit()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
