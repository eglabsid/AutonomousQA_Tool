# import easyocr
# import cv2

# # EasyOCR 리더 생성
# reader = easyocr.Reader(['en', 'ko'],gpu=True)  # 필요한 언어를 추가하세요

# # 이미지 로드
# image_path = 'screen/screenshot.jpg'
# image = cv2.imread(image_path)

# # 텍스트 인식
# results = reader.readtext(image)

# # 결과 출력
# for (bbox, text, prob) in results:
#     # bbox는 텍스트의 위치를 나타내는 네 개의 좌표입니다
#     print(f"Text: {text}, Probability: {prob}")
#     # print(f"Bounding Box: {bbox}")

#     # 텍스트 위치에 사각형 그리기
#     (top_left, top_right, bottom_right, bottom_left) = bbox
#     top_left = tuple(map(int, top_left))
#     bottom_right = tuple(map(int, bottom_right))
#     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# # 결과 이미지 저장
# cv2.imwrite('result_1.png', image)

import keras_ocr
import cv2
import threading

# Keras-OCR 파이프라인 생성
pipeline = keras_ocr.pipeline.Pipeline()

# 이미지 로드
image_path = 'screen/ui_test.jpg'
image = keras_ocr.tools.read(image_path)

# 텍스트 인식 함수
def recognize_text(image, results, lock):
    prediction_groups = pipeline.recognize([image])
    with lock:
        results.extend(prediction_groups[0])

# 멀티쓰레드 설정
results = []
lock = threading.Lock()
threads = []

# 이미지 분할 (예: 4개의 영역으로 분할)
height, width = image.shape[:2]
regions = [
    image[0:height//2, 0:width//2],
    image[0:height//2, width//2:width],
    image[height//2:height, 0:width//2],
    image[height//2:height, width//2:width]
]

# 각 영역에 대해 스레드 생성 및 시작
for region in regions:
    thread = threading.Thread(target=recognize_text, args=(region, results, lock))
    threads.append(thread)
    thread.start()

# 모든 스레드가 완료될 때까지 대기
for thread in threads:
    thread.join()

# 결과 출력 및 바운딩 박스 그리기
for (text, box) in results:
    print(f"Text: {text}, Bounding Box: {box}")
    
    # 바운딩 박스 좌표 추출
    (top_left, top_right, bottom_right, bottom_left) = box
    top_left = tuple(map(int, top_left))
    top_right = tuple(map(int, top_right))
    bottom_right = tuple(map(int, bottom_right))
    bottom_left = tuple(map(int, bottom_left))
    
    # 바운딩 박스 그리기
    cv2.line(image, top_left, top_right, (0, 255, 0), 2)
    cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
    cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
    cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)
    
    # 텍스트 표시
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 결과 이미지 저장
cv2.imwrite('result.png', image)
