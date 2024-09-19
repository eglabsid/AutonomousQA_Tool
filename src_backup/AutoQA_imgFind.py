import cv2
import numpy as np
import pyautogui


def find_image_center(image_path, similarity_threshold):
    # 스크린샷을 찍어 numpy 배열로 변환
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)

    # 찾을 이미지 로드
    template = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 템플릿 매칭을 수행 (TM_CCOEFF_NORMED는 유사도 측정 방식)
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

    # 유사도 임계값 설정 (threshold 이상의 유사도만 고려)
    locations = np.where(result >= similarity_threshold)

    # 매칭된 좌표 중 첫 번째 좌표를 기준으로 중심 좌표를 계산
    for point in zip(*locations[::-1]):
        x, y = point
        h, w, _ = template.shape  # 템플릿 이미지의 높이와 너비
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y)  # 첫 번째 매칭된 이미지의 중심 좌표 반환

    # 일치하는 이미지가 없는 경우
    return None
