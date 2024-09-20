import torch
import cv2
import numpy as np
from PIL import Image

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    img = Image.open(image_path)
    
    # 객체 탐지 수행
    results = model(img)
    
    # 결과 이미지 생성
    img_with_boxes = np.squeeze(results.render())
    
    return img_with_boxes

def main():
    # 이미지 경로 설정 (자신의 이미지 경로로 변경하세요)
    image_path = 'test3.png'
    
    # 객체 탐지 수행
    result_image = detect_objects(image_path)
    
    # 결과 표시
    imS = cv2.resize(result_image, (960, 540))  
    cv2.imshow('Object Detection Result', imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()