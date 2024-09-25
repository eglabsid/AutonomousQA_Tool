import cv2
import numpy as np
import threading

class TemplateMatcher:
    def __init__(self, template, scale_range, scale_step, angle_range, angle_step, threshold=0.8):
        self.template = template
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.threshold = threshold
        self.matches = []
        self.lock = threading.Lock()

    def match_templates(self, gray_frame, scale, angle):
        # 템플릿 회전
        (h, w) = self.template.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_template = cv2.warpAffine(self.template, M, (w, h))

        # 템플릿 매칭
        result = cv2.matchTemplate(gray_frame, rotated_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= self.threshold)

        with self.lock:
            for loc in zip(*locations[::-1]):
                self.matches.append((loc, scale, angle, result[loc[1], loc[0]]))

    def get_matches(self, gray_frame):
        threads = []
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            for angle in np.arange(self.angle_range[0], self.angle_range[1], self.angle_step):
                thread = threading.Thread(target=self.match_templates, args=(gray_frame, scale, angle))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        return self.matches

    def draw_matches(self, image):
        for (loc, scale, angle, score) in self.matches:
            top_left = loc
            (h, w) = self.template.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated_template = cv2.warpAffine(self.template, M, (w, h))
            bottom_right = (top_left[0] + rotated_template.shape[1], top_left[1] + rotated_template.shape[0])
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image

# 사용 예제
template = cv2.imread('template.png', 0)
image = cv2.imread('image.png', 0)
matcher = TemplateMatcher(template, scale_range=(0.5, 1.5), scale_step=0.1, angle_range=(-180, 180), angle_step=10, threshold=0.8)
matches = matcher.get_matches(image)

# 매치 결과를 이미지에 그리기
result_image = matcher.draw_matches(cv2.imread('image.png'))
cv2.imwrite('result.png', result_image)
# cv2.imshow('Matches', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
