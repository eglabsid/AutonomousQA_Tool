import cv2
import pyautogui
import psutil
import numpy as np
import time
from Quartz import CoreGraphics as CG

# Visual Studio Code 프로세스 ID 찾기
def find_vscode_pid():
    for proc in psutil.process_iter(['pid', 'name']):
        print(proc.info['name'])
        if proc.info['name'] == 'Qt Creator':  # macOS의 경우
            return proc.info['pid']
    return None

# 프로세스 ID를 사용하여 창을 활성화하는 함수
def activate_window(pid):
    windows = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionOnScreenOnly, CG.kCGNullWindowID)
    for window in windows:
        if window.get('kCGWindowOwnerPID') == pid:
            window_id = window['kCGWindowNumber']
            CG.CGWindowListCreateDescriptionFromArray([window_id])
            CG.CGWindowListCreateImage(CG.CGRectNull, CG.kCGWindowListOptionIncludingWindow, window_id, CG.kCGWindowImageDefault)
            break

# Visual Studio Code 창을 활성화시키기 위해 프로세스 ID 사용
vscode_pid = find_vscode_pid()

if vscode_pid:
    activate_window(vscode_pid)
    time.sleep(1)  # 창이 활성화될 시간을 기다림

    # 추적할 이미지 파일 경로
    image_path = 'target/q.png'

    # 이미지 찾기
    location = pyautogui.locateOnScreen(image_path, confidence=0.7)

    if location:
        print(f"이미지를 찾았습니다! 위치: {location}")

        # 이미지의 중심 좌표
        center = pyautogui.center(location)
        print(f"중심 좌표: {center}")

        # 스크린샷 찍기
        screenshot = pyautogui.screenshot()

        # 스크린샷을 OpenCV 이미지로 변환
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # 바운딩 박스 그리기
        top_left = (location.left, location.top)
        bottom_right = (location.left + location.width, location.top + location.height)
        cv2.rectangle(screenshot, top_left, bottom_right, (0, 0, 255), 3)

        # 바운딩 박스가 그려진 이미지 저장
        cv2.imwrite('screenshot_with_bounding_box.png', screenshot)
        print("바운딩 박스가 그려진 스크린샷을 저장했습니다: screenshot_with_bounding_box.png")

        # 이미지 크롭
        cropped_image = screenshot[location.top:location.top + location.height, location.left:location.left + location.width]
        cv2.imwrite('cropped_image.png', cropped_image)
        print("크롭된 이미지를 저장했습니다: cropped_image.png")
    else:
        print("이미지를 찾을 수 없습니다.")
else:
    print("Visual Studio Code 프로세스를 찾을 수 없습니다.")
