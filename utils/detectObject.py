import torch
import numpy as np

from capture import *

# pip install torch torchvision pyautogui opencv-python pillow pywin32
#
# git clone https://github.com/ultralytics/yolov5
# cd yolov5
# pip install -r requirements.txt


class DetectObject():
    
    def __init__(self):
        self.model = self.load_yolo_model()
        self.screen = None
        self.detections = {}
        self.targets = []

    
    def load_yolo_model(self):
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def detect_class_in_gui(self, target_class):
        target_detections = self.detections[self.detections['name'] == target_class]
        
        if target_detections.empty:
            print(f"'{target_class}' 클래스의 객체를 찾을 수 없습니다.")
            return None
        
        best_detection = target_detections.iloc[0]
        x1, y1, x2, y2 = best_detection[['xmin', 'ymin', 'xmax', 'ymax']]
        
        result ={}
        result['x'] = int(x1)
        result['y'] = int(y1)
        result['width'] = int(x2) - int(x1)
        result['height'] = int(y2) - int(y1)
        
        if result:
            print(f"검출된 객체: {target_class}")
            print(f"위치: ({result['x']}, {result['y']})")
            print(f"크기: {result['width']}x{result['height']}")
        else:
            print("객체를 찾을 수 없습니다.")
        
        return result

    def detect_images_in_gui(self, process_name):
        hwnd = get_process_window(process_name)
        
        if not hwnd:
            print(f"프로세스 '{process_name}'을 찾을 수 없습니다.")
            return None

        self.screenshot = capture_process_window(hwnd)
        if self.screenshot is None:
            print("스크린샷을 캡처할 수 없습니다.")
            return None

        results = self.model(self.screenshot)
        
        # 결과 출력    
        self.detections = results.pandas().xyxy[0]
        for _, row in self.detections.iterrows():
            print(f"객체: {row['name']}")
            print(f"신뢰도: {row['confidence']:.2f}")
            print(f"바운딩 박스: ({row['xmin']:.0f}, {row['ymin']:.0f}, {row['xmax']:.0f}, {row['ymax']:.0f})")
            print("---")
            
        return results
        

def main():
    process_name = "Geometry Dash"  # 예: "Notepad", "Chrome" 등
    target_class = "clock"  # 예: "person", "car", "laptop" 등
    
    
    observer = DetectObject()
    # model = load_yolo_model()
    results = observer.detect_images_in_gui(process_name)
    result = observer.detect_class_in_gui(target_class)
    
    results.show()        
    

if __name__ == "__main__":
    main()