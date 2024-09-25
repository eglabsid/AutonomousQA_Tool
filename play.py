import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

# opencv
import cv2


import numpy as np
import mss
import mss.tools

# import gc
from memory_profiler import profile

import pygetwindow as gw

# pyautogui
import pyautogui
import subprocess, psutil

import threading

window_ui = 'play.ui'

crop_size = {}
crop_size['width'] = 300
crop_size['height'] = 300

class TemplateMatcher:
    def __init__(self, template, scale_range, scale_step):
        self.template = template
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.best_val = -1
        self.best_match = None
        self.best_scale = None
        self.best_loc = None
        self.lock = threading.Lock()

    def match_template(self, gray_frame, scale):
        resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        with self.lock:
            if max_val > self.best_val:
                self.best_val = max_val
                self.best_match = resized_template
                self.best_scale = scale
                self.best_loc = max_loc

    def run(self, gray_frame):
        threads = []
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            thread = threading.Thread(target=self.match_template, args=(gray_frame, scale))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return self.best_val, self.best_match, self.best_scale, self.best_loc


class ScreenCaptureThread(QThread):
    frame_captured = pyqtSignal(np.ndarray, tuple, np.ndarray)
    

    def __init__(self, target_img, fps=30):
        super().__init__()
        self.fps = fps
        self.running = True
        
        # 템플릿 이미지 로드
        self.template = cv2.imread(target_img, 0)
        self.template_w, self.template_h = self.template.shape[::-1]

    def detect_template_with_pyautogui(self,frame,location):
        
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
            print("크롭된 이미지를 저장했습니다: cropped_image.png")
        else:
            print("이미지를 찾을 수 없습니다.")
            
    def detect_template_in_frame(self,frame,template, threshold = 0.8):
        
        template_w, template_h = template.shape[::-1]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use template matching to find the object in the frame
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Define the bounding box for the detected object
        top_left = max_loc
        bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

        # Check if the match is good enough
        if max_val > threshold:  # You can adjust this threshold
            return True
        else:
            return False

    # 초기 검색을 위해 사용
    def detect_template_using_cv(self,frame, scale_range=(0.45, 0.85), scale_step=0.15):
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matcher = TemplateMatcher(self.template, scale_range, scale_step)
        best_val, best_match, best_scale, best_loc = matcher.run(gray_frame)
        
        top_left = best_loc
        # h, w, _ = best_match.shape
        h, w = best_match.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bounding_box = [top_left,bottom_right]
        
        # Match 된 Bounding Box
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        
        center_middle = (top_left[0] + w*0.5, top_left[1] + h*0.5)
        
        # 크롭할 영역 정의 (x, y, width, height)
        w = crop_size['width']
        h = crop_size['height']
        crop_top_left = (top_left[0] - int(w*0.2), top_left[1] - int(h*0.2))
        crop_bottom_right = (crop_top_left[0] + int(w*2), crop_top_left[1] + int(h*1))
        # crop_w, crop_h = 250, 250
        crop_sx, crop_sy = crop_top_left[0],crop_top_left[1] 
        crop_ex, crop_ey = crop_bottom_right[0],crop_bottom_right[1] 
        cropped_img = frame[crop_sy:crop_ey, crop_sx:crop_ex]

        # 결과를 이미지에 그리기
        cv2.rectangle(frame, crop_top_left, crop_bottom_right, (255, 0, 0), 4)
        
        return center_middle, cropped_img, bounding_box
        
    def run(self):
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while self.running:
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                center_middle, crop_img, _ = self.detect_template_using_cv(img)

                # self.frame_captured.emit(img,center_middle,cropped_image)
                self.frame_captured.emit(img,center_middle,crop_img)
                self.msleep(int(1000 / self.fps))
                

    def stop(self):
        self.running = False
        self.wait()


class playWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(playWindow, self).__init__()

        # UI 파일 로드
        uic.loadUi(window_ui, self)
        
        self.centralWidget.setLayout(self.main_layout)
        
        self.setWindowTitle("Image based Playing Tool (ver.Dev)")
        self.setWindowIcon(QIcon('images/icon/eglab.ico'))
        self.setGeometry(300, 300, 800, 900)
        
        self.log.setReadOnly(True)
        # buttons = "self.pushButton_"
        # for n in range(4):
        #     name = buttons+str(n)
        #     button = eval(name)
        #     button.setFixedHeight(80)
        #     if n == 0:
        #         button.clicked.connect(self.capture_screen)

        # fps 설정
        self.capture_thread = ScreenCaptureThread(target_img='target/b1.jpg',fps=60)
        self.capture_thread.frame_captured.connect(self.update_frame)
        self.capture_thread.start()


    def get_widgets_img(self,frame, re_width, re_height):
        # PyQt5 프레임에 표시
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        frame_bytes = frame.tobytes()
        q_img = QImage(frame_bytes, width, height, bytes_per_line, QImage.Format_RGB888)

        # QImage를 리사이즈
        q_img_resized = q_img.scaled(re_width,re_height)
        return QPixmap.fromImage(q_img_resized)

    
    # @profile
    def update_frame(self, frame, pos, crop_frame):

        pix_map = self.get_widgets_img(frame, 500, 500)
        self.screen.setPixmap(pix_map)
        
        pix_map = self.get_widgets_img(crop_frame, crop_size['width'], crop_size['height'])
        self.monitor.setPixmap(pix_map)
                              

        self.log.append("추적하는 객체 위치 : {0},{1}".format(pos[0],pos[1]))
        

    def connect_application_by_process_name(self, process_name):
        # 메모장 실행
        process = subprocess.Popen([process_name])
        # psutil을 사용하여 프로세스 정보 가져오기
        proc = psutil.Process(process.pid)

        while not proc.is_running():
            self.check_process_load(proc)

        # 윈도우 활성화
        window = self.find_window_by_pid(proc.pid)
        if window:
            window.activate()
            print("Window activated successfully!")
        else:
            print("Window not found.")        

        
    def mouseclick(self,button:str,coords:tuple):
        # 마우스 이동
        pyautogui.moveTo(coords[0],coords[1], duration=1)
        # 마우스 클릭
        pyautogui.click()
    
    def sendkey(self,key:str):        
        pyautogui.press(key)

    # 프로세스 ID를 기반으로 윈도우 찾기
    def find_window_by_pid(self, pid):
        for window in gw.getWindowsWithTitle(''):
            if window._hWnd == pid:
                return window
        return None
    
    # 프로세스 로드 상태 체크 함수
    def check_process_load(self, proc):
        try:
            if proc.is_running():
                cpu_usage = proc.cpu_percent(interval=1)
                memory_info = proc.memory_info()
                memory_usage = memory_info.rss / (1024 * 1024)  # 메모리 사용량 (MB)
                print(f"CPU Usage: {cpu_usage}%")
                print(f"Memory Usage: {memory_usage} MB")
            else:
                print("Process is not running.")
        except psutil.NoSuchProcess:
            print("Process no longer exists.")

    def closeEvent(self, event):
        self.capture_thread.stop()
        event.accept()
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = playWindow()
    window.show()
    sys.exit(app.exec_())
