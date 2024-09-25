import cv2
import numpy as np
import os
import glob
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QStatusBar
from PyQt5.QtCore import pyqtSignal, QThread

def get_subfolders(root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    return subfolders

def get_all_images(root_folder):
    # 이미지 파일 확장자 목록
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    # 모든 이미지 파일 경로를 저장할 리스트
    all_images = []
    
    # 하위 폴더를 포함한 모든 이미지 파일 검색
    for extension in image_extensions:
        all_images.extend(glob.glob(os.path.join(root_folder, '**', extension), recursive=True))

    return all_images

class TemplateMatcher(QThread):
    update_progress = pyqtSignal(int, int)  # 현재 진행 상황과 총 작업 수를 전달하는 시그널
    finished = pyqtSignal(np.ndarray)  # 작업 완료 시 결과 이미지를 전달하는 시그널

    def __init__(self, templates, scale_range, scale_step, threshold=0.8, gray_frame=None):
        super().__init__()
        self.templates = templates
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.threshold = threshold
        self.matches = []
        self.lock = threading.Lock()
        self.gray_frame = gray_frame

    def match_templates(self, template, scale):
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        result = cv2.matchTemplate(self.gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= self.threshold)

        with self.lock:
            for loc in zip(*locations[::-1]):
                self.matches.append((loc, scale, result[loc[1], loc[0]], template))

    def run(self):
        total_tasks = len(self.templates) * len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_step))
        current_task = 0
        for template in self.templates:
            for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
                self.match_templates(template, scale)
                current_task += 1
                self.update_progress.emit(current_task, total_tasks)

        result_image = self.draw_matches(cv2.imread('screen/ui_test.jpg'))
        self.finished.emit(result_image)

    def draw_matches(self, image):
        for (loc, scale, score, template) in self.matches:
            top_left = loc
            bottom_right = (top_left[0] + int(template.shape[1] * scale), top_left[1] + int(template.shape[0] * scale))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Template Matcher')
        self.setGeometry(100, 100, 800, 600)
        self.show()

    def update_status_bar(self, current, total):
        self.status_bar.showMessage(f"Progress: {current}/{total}")

    def on_finished(self, result_image):
        cv2.imwrite('result_matchs.jpg', result_image)
        self.status_bar.showMessage("Matching completed!")

    def run_template_matching(self):
        folder_dir = 'screen/UI'
        image_files = get_all_images(folder_dir)
        sub_folders = get_subfolders(folder_dir)
        # 사용 예제
        templates = [cv2.imread(file, 0) for file in image_files]
        image = cv2.imread('screen/ui_test.jpg')
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.matcher = TemplateMatcher(templates, scale_range=(0.9, 1.2), scale_step=0.1, threshold=0.7, gray_frame=gray_frame)
        self.matcher.update_progress.connect(self.update_status_bar)  # 시그널 연결
        self.matcher.finished.connect(self.on_finished)  # 작업 완료 시그널 연결
        self.matcher.start()  # QThread 시작

def main():
    app = QApplication([])
    main_window = MainWindow()
    main_window.run_template_matching()
    app.exec_()

if __name__ == "__main__":
    main()
