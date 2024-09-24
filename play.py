import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

import cv2
import numpy as np
import mss
import mss.tools

# import gc
from memory_profiler import profile

import pygetwindow as gw
import pyautogui
import subprocess, psutil

window_ui = 'play.ui'


class ScreenCaptureThread(QThread):
    frame_captured = pyqtSignal(np.ndarray, tuple, np.ndarray)
    

    def __init__(self, target_img, fps=30):
        super().__init__()
        self.fps = fps
        self.running = True
        
        # 템플릿 이미지 로드
        self.template = cv2.imread(target_img, 0)
        self.template_w, self.template_h = self.template.shape[::-1]
    
    
    def run(self):
        scale_range=(0.5, 1.5)
        scale_step=0.15
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while self.running:
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # 그레이스케일로 변환
                gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                best_match = None
                best_val = -1
                best_scale = 1.0
                best_loc = (0, 0)

                try:
                    for scale in np.arange(scale_range[0], scale_range[1], scale_step):
                        # 크롭된 이미지의 크기 조정
                        resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
                        result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                        if max_val > best_val:
                            best_val = max_val
                            best_match = resized_template
                            best_scale = scale
                            best_loc = max_loc
                except:
                    pass

                top_left = best_loc
                # h, w, _ = best_match.shape
                h, w = best_match.shape
                bottom_right = (top_left[0] + w, top_left[1] + h)

                center_middle = (top_left[0] + w*0.5, top_left[1] + h*0.5)
                
                # 크롭할 영역 정의 (x, y, width, height)
                crop_w, crop_h = 250, 250
                crop_x, crop_y = top_left[0],top_left[1] 
                cropped_image = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

                # 결과를 이미지에 그리기
                cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 4)

                self.frame_captured.emit(img,center_middle,cropped_image)
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
        self.setGeometry(300, 300, 600, 800)
        
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

    
    @profile
    def update_frame(self, frame, pos, crop_frame):

        pix_map = self.get_widgets_img(frame, 500, 500)
        self.screen.setPixmap(pix_map)
        
        pix_map = self.get_widgets_img(crop_frame, 250, 250)
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
