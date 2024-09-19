import torch
import pyautogui
import cv2
import numpy as np
import win32gui
import win32ui
import win32con
from PIL import Image

# pip install torch torchvision pyautogui opencv-python pillow pywin32
#
# git clone https://github.com/ultralytics/yolov5
# cd yolov5
# pip install -r requirements.txt
#
def get_process_window(process_name):
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            if process_name.lower() in win32gui.GetWindowText(hwnd).lower():
                hwnds.append(hwnd)
        return True
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds[0] if hwnds else None

def capture_process_window(hwnd):
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    else:
        return None

def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_image_in_gui(model, process_name, target_class):
    hwnd = get_process_window(process_name)
    if not hwnd:
        print(f"프로세스 '{process_name}'을 찾을 수 없습니다.")
        return None

    screenshot = capture_process_window(hwnd)
    if screenshot is None:
        print("스크린샷을 캡처할 수 없습니다.")
        return None

    results = model(screenshot)
    
    detections = results.pandas().xyxy[0]
    target_detections = detections[detections['name'] == target_class]
    
    if target_detections.empty:
        print(f"'{target_class}' 클래스의 객체를 찾을 수 없습니다.")
        return None
    
    best_detection = target_detections.iloc[0]
    x1, y1, x2, y2 = best_detection[['xmin', 'ymin', 'xmax', 'ymax']]
    
    return {
        'x': int(x1),
        'y': int(y1),
        'width': int(x2 - x1),
        'height': int(y2 - y1)
    }

def main():
    process_name = "대상 프로세스 이름"  # 예: "Notepad", "Chrome" 등
    target_class = "찾고자 하는 객체 클래스"  # 예: "person", "car", "laptop" 등
    
    model = load_yolo_model()
    result = detect_image_in_gui(model, process_name, target_class)
    
    if result:
        print(f"검출된 객체: {target_class}")
        print(f"위치: ({result['x']}, {result['y']})")
        print(f"크기: {result['width']}x{result['height']}")
    else:
        print("객체를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()