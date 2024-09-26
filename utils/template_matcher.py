import cv2
import numpy as np
import threading

import os
import glob
from tqdm import tqdm

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
        # all_images.extend(glob.glob(os.path.join(root_folder, '**', extension), recursive=False))
        all_images.extend(glob.glob(os.path.join(root_folder, extension), recursive=False))

    return all_images

class UITemplateMatcher(QThread):
    update_progress = pyqtSignal(int, int)  # 현재 진행 상황과 총 작업 수를 전달하는 시그널
    finished = pyqtSignal(np.ndarray)  # 작업 완료 시 결과 이미지를 전달하는 시그널

    __slot__ = ['frame','templates','scale_range','scale_step','threshold','lock']
    
    def __init__(self, frame, templates, scale_range, scale_step, threshold=0.8):
        super().__init__()
        self.frame = frame
        self.templates = templates
        self.scale_range = scale_range
        self.scale_step = scale_step
        self.threshold = threshold
        
        self.matches = []
        self.lock = threading.Lock()
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    def match_templates(self, template, pbar):
        # Dict 처리
        template_tuple = [ (k,v) for k,v in template.items()][0]
        # name = template_tuple[0]
        img = template_tuple[1]
        
        best_match = None
        best_val = -1
        best_scale = 1.0
        best_loc = (0, 0)
                    
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            resized_template = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            result = cv2.matchTemplate(self.gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            # locations = np.where(result >= self.threshold)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            pbar.update(1)  # 스레드 완료 시 진행 상황 업데이트
            self.current_task += 1
            self.update_progress.emit(self.current_task,self.total_tasks)
            with self.lock:
                if max_val > best_val and max_val >= self.threshold:
                    best_val = max_val
                    best_match = resized_template
                    best_scale = scale
                    best_loc = max_loc
        
        with self.lock:
            self.matches.append((best_loc, best_scale, result[best_loc[1], best_loc[0]], template_tuple))
                
    def run(self):
        threads = []
        self.total_tasks = len(self.templates) * len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_step))
        self.current_task = 0
        with tqdm(total=self.total_tasks, desc="Matching templates") as pbar:
            while len(self.templates) > 0:
                template = self.templates.pop(0)
                thread = threading.Thread(target=self.match_templates, args=(template, pbar))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
                    
            result_image = self.draw_matches(self.frame)
            self.finished.emit(result_image)
    
    def stop(self):
        # self.running = False
        self.stop()
        
    def draw_matches(self, image):
        for i,(loc, scale, score, template) in enumerate(self.matches):
            top_left = loc
            bottom_right = (top_left[0] + int(template[1].shape[1] * scale), top_left[1] + int(template[1].shape[0] * scale))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 4)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(image, f'{template[0]}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imwrite(f"test_result/{i}.jpg",image)
        return image
        
class TemplateMatcher:
    def __init__(self, template, scale_range, scale_step, threshold=0.8):
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.scale_range = scale_range
        self.scale_step = scale_step
        # template 하나에 대해서만 추출할 경우
        self.best_val = -1
        self.best_match = None
        self.best_scale = None
        self.best_loc = None
        # template 과 같은 여러 개체 추출할 경우
        self.threshold = threshold
        self.matches = []
        self.lock = threading.Lock()
        # current progress state
        self.current_task = 0
        self.total_task = 0
        self.templates = []

    def match_templates(self, gray_frame, pbar):
        
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
            result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.threshold)

            pbar.update(1)  # 스레드 완료 시 진행 상황 업데이트
            self.current_task += 1
            with self.lock:
                for loc in zip(*locations[::-1]):
                    self.matches.append((loc, scale, result[loc[1], loc[0]]))

    def match_a_template(self, gray_frame, pbar):
        
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            
            resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
            result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            pbar.update(1)  # 스레드 완료 시 진행 상황 업데이트
            self.current_task += 1
            with self.lock:
                if max_val > self.best_val:
                    self.best_val = max_val
                    self.best_match = resized_template
                    self.best_scale = scale
                    self.best_loc = max_loc
    
    def match_mixed_templates(self, gray_frame, template_tuple, pbar):
        
        template = template_tuple[1]
        
        best_match = None
        best_val = -1
        best_scale = 1.0
        best_loc = (0, 0)
        
        for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
            resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            result = cv2.matchTemplate(gray_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            # locations = np.where(result >= self.threshold)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            pbar.update(1)  # 스레드 완료 시 진행 상황 업데이트
            self.current_task += 1
            
            with self.lock:
                if max_val > best_val and max_val >= self.threshold:
                    best_val = max_val
                    best_match = resized_template
                    best_scale = scale
                    best_loc = max_loc
        
        with self.lock:
            self.matches.append((best_loc, best_scale, result[best_loc[1], best_loc[0]], template_tuple))
    
    # templates 는 dictionary를 갖고 있는 list 타입
    def get_mixed_match(self, image, templates):
        
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        threads = []
        self.total_task = len(templates) * len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_step))
        self.current_task = 0
        with tqdm(total=self.total_task, desc="Matching templates") as pbar:
            # for template in templates:
            while len(templates) > 0:
                template = templates.pop(0)
                # Dict 처리
                template_tuple = [ (k,v) for k,v in template.items()][0]
                
                # for scale in np.arange(self.scale_range[0], self.scale_range[1], self.scale_step):
                thread = threading.Thread(target=self.match_mixed_templates, args=(gray_frame, template_tuple, pbar))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
                
    def get_matches(self, image):
        threads = []
        total_tasks = len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_step))
        with tqdm(total=total_tasks, desc="Matching templates") as pbar:
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thread = threading.Thread(target=self.match_templates, args=(gray_frame, pbar))
            threads.append(thread)
            thread.start()

            for thread in threads:
                thread.join()

        return self.matches

    def get_a_match(self, image):
        threads = []
        total_tasks = len(np.arange(self.scale_range[0], self.scale_range[1], self.scale_step))
        with tqdm(total=total_tasks, desc="Matching templates") as pbar:
            gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thread = threading.Thread(target=self.match_a_template, args=(gray_frame, pbar))
            threads.append(thread)
            thread.start()

            for thread in threads:
                thread.join()

        return self.best_val, self.best_match, self.best_scale, self.best_loc
    
    def draw_matches(self, image):
        for (loc, scale, score) in self.matches:
            top_left = loc
            bottom_right = (top_left[0] + int(self.template.shape[1] * scale), top_left[1] + int(self.template.shape[0] * scale))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image
    
    def draw_mixed_matches(self, image):
        for (loc, scale, score, template) in self.matches:
            top_left = loc
            bottom_right = (top_left[0] + int(template[1].shape[1] * scale), top_left[1] + int(template[1].shape[0] * scale))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            # cv2.putText(image, f'{score:.2f}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(image, f'{template[0]}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        return image


def main():
    # 사용 예제
    template = cv2.imread('target/b1.jpg', 0)
    image = cv2.imread('screen/screenshot.jpg', 0)
    matcher = TemplateMatcher(template, scale_range=(0.5, 1.2), scale_step=0.1, threshold=0.8)
    matches = matcher.get_matches(image)

    # 매치 결과를 이미지에 그리기
    result_image = matcher.draw_matches(cv2.imread('screen/screenshot.jpg'))
    cv2.imwrite('result_matchs.png', result_image)

if __name__ == "__main__":
    main()