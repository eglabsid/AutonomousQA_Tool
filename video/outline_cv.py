import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_and_get_contours(image_path, threshold=127):
    # 1. 이미지 로드 및 그레이스케일 변환
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 임계값 처리로 이진화
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 3. 윤곽선 추출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 윤곽선 그리기 (시각화)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    return binary, contours, contour_image

def display_results(original, binary, contour_image):
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("Binary Segmentation")
    plt.imshow(binary, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Contours on Image")
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # 이미지 경로
    image_path = "./video/test.jpg"
    
    # 세그멘테이션 및 윤곽선 추출
    binary, contours, contour_image = segment_and_get_contours(image_path, threshold=127)
    
    # 원본 이미지 로드
    original_image = cv2.imread(image_path)
    
    # 결과 시각화
    display_results(original_image, binary, contour_image)
    
    # 윤곽선 좌표 출력
    for i, contour in enumerate(contours):
        print(f"Contour {i}: {contour.reshape(-1, 2)}")  # 좌표 정보를 2D 배열로 출력