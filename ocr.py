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

# pip install tensorflow==2.13.0 keras==2.13.1 keras-ocr==0.9.3

# Keras-OCR 파이프라인 생성
pipeline = keras_ocr.pipeline.Pipeline()

# 이미지 로드
image_path = 'screen/screenshot.jpg'
image = keras_ocr.tools.read(image_path)

# 텍스트 인식
prediction_groups = pipeline.recognize([image])

# 결과 출력 및 바운딩 박스 그리기
for predictions in prediction_groups:
    for (text, box) in predictions:
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
