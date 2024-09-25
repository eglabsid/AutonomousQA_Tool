import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab


# Tesseract-OCR 경로 설정 (Windows의 경우)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 스크린샷 캡처
# screenshot = ImageGrab.grab()
# screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# 이미지를 그레이스케일로 변환
gray_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

# OCR 수행
data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)

# 결과 출력
for i in range(len(data['text'])):
    if int(data['conf'][i]) > 60:  # 신뢰도 60 이상인 경우만 출력
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        text = data['text'][i]
        print(f'Text: {text}, Coordinates: ({x}, {y}, {w}, {h})')
        # 검출된 텍스트와 좌표를 이미지에 표시
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(screenshot, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 결과 이미지 저장
cv2.imwrite('result.png', screenshot)
