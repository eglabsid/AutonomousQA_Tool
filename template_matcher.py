import cv2
import numpy as np
import threading

class TemplateMatcher:
    def __init__(self, template, scale_range, scale_step, threshold=0.8):
        self.template = template
        self.scale_range = scale_range
        self.scale_step = scale_step
        # template 하나에 대해서만 추출할 경우
        self.best_val = -1
        self.best_match = None
        self.best_scale = None
        self.best_loc = None
        # template 과 같은 여러 개체 추출할 경우
        self.threshold = threshold
        self.matches = []
        self.lock = threading.Lock()

    def match_templates(self, gray_frame, scale):
        resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= self.threshold)

        with self.lock:
            for loc in zip(*locations[::-1]):
                self.matches.append((loc, scale, result[loc[1], loc[0]]))

    def match_a_template(self, gray_frame, scale):
        resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        with self.lock:
            if max_val > self.best_val:
                self.best_val = max_val
                self.best_match = resized_template
                self.best_scale = scale
                self.best_loc = max_loc
                
    def get_matches(self, gray_frame):
        threads = []
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            thread = threading.Thread(target=self.match_templates, args=(gray_frame, scale))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return self.matches

    def get_a_match(self, gray_frame):
        threads = []
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            thread = threading.Thread(target=self.match_template, args=(gray_frame, scale))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return self.best_val, self.best_match, self.best_scale, self.best_loc
    
    def draw_matches(self, image):
        for (loc, scale, score) in self.matches:
            top_left = loc
            bottom_right = (top_left[0] + int(self.template.shape[1] * scale), top_left[1] + int(self.template.shape[0] * scale))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image


def main():
    # 사용 예제
    template = cv2.imread('target/b1.jpg', 0)
    image = cv2.imread('screen/screenshot.jpg', 0)
    matcher = TemplateMatcher(template, scale_range=(0.5, 1.2), scale_step=0.1, threshold=0.8)
    matches = matcher.get_matches(image)

    # 매치 결과를 이미지에 그리기
    result_image = matcher.draw_matches(cv2.imread('screen/screenshot.jpg'))
    cv2.imwrite('result_matchs.png', result_image)

if __name__ == "__main__":
    main()