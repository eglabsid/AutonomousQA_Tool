import cv2
import numpy as np
import pyautogui


def return_center_pos_of_found_img(image_path, similarity_threshold):
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


def detect_image(source_path, target_path):
    # 소스 이미지와 타겟 이미지 읽기
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    
    # 이미지를 그레이스케일로 변환
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    # 템플릿 매칭 수행
    result = cv2.matchTemplate(source_gray, target_gray, cv2.TM_CCOEFF_NORMED)
    
    
    # 임계값 설정 (이 값을 조정하여 매칭의 정확도를 조절할 수 있습니다)
    threshold = 0.8
    
    # 임계값을 넘는 위치 찾기
    locations = np.where(result >= threshold)
    
    # cv2.imshow('locations',result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    locations = list(zip(*locations[::-1]))
    
    detected = []
    
    # 중복 검출 제거
    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), target.shape[1], target.shape[0]]
        detected.append(rect)
        detected = sorted(detected, key=lambda r: r[2]*r[3], reverse=True)
        detected = [detected[0]] + [d for d in detected[1:] if not is_inside(detected[0], d)]
    
    return detected

def is_inside(rect1, rect2):
    return (rect1[0] < rect2[0] + rect2[2] and
            rect1[0] + rect1[2] > rect2[0] and
            rect1[1] < rect2[1] + rect2[3] and
            rect1[1] + rect1[3] > rect2[1])

def main():
    target_path = 'src_backup/test.PNG'
    # source_path = 'src_backup/sadCat.png'
    source_path = 'src_backup/sadCat.png'
    
    detected = detect_image(source_path, target_path)
    
    if detected:
        print(f"검출된 이미지 수: {len(detected)}")
        for i, rect in enumerate(detected):
            print(f"검출 {i+1}:")
            print(f"  위치: ({rect[0]}, {rect[1]})")
            print(f"  크기: {rect[2]}x{rect[3]} 픽셀")
            print(f"  중앙 위치: {(rect[2]+rect[0])*0.5}x{(rect[3]+rect[1])*0.5} 픽셀")
    else:
        print("타겟 이미지를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()