import cv2
import numpy as np
import threading

class TemplateMatcher:
    def __init__(self, template, scale_range, scale_step):
        self.template = template
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.best_val = -1
        self.best_match = None
        self.best_scale = None
        self.best_loc = None
        self.lock = threading.Lock()

    def match_template(self, gray_frame, scale):
        resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        with self.lock:
            if max_val > self.best_val:
                self.best_val = max_val
                self.best_match = resized_template
                self.best_scale = scale
                self.best_loc = max_loc

    def run(self, gray_frame):
        threads = []
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            thread = threading.Thread(target=self.match_template, args=(gray_frame, scale))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return self.best_val, self.best_match, self.best_scale, self.best_loc

# 사용 예시
if __name__ == "__main__":
    template = cv2.imread('screen/screenshot.jpg', 0)
    gray_frame = cv2.imread('target/b1.jpg', 0)
    scale_range = (0.5, 1.5)
    scale_step = 0.1

    matcher = TemplateMatcher(template, scale_range, scale_step)
    best_val, best_match, best_scale, best_loc = matcher.run(gray_frame)

    print(f'Best value: {best_val}')
    print(f'Best scale: {best_scale}')
    print(f'Best location: {best_loc}')
