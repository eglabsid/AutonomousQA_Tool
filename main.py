import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from src.action_dialog import ActionDialog
from src.image_dialog import ImageDialog
from src.interval_dialog import IntervalDialog

from utils.routine import Routine
from utils.template_matcher import UITemplateMatcher, TemplateMatcher, get_all_images, get_subfolders

# opencv
import cv2

import numpy as np
import mss
import mss.tools

window_ui = 'main_window.ui'

class mainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.initUI()
        
    def initUI(self):
        # UI 파일 로드
        uic.loadUi(window_ui, self)
        self.centralWidget.setLayout(self.main_layout)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_label = QtWidgets.QLabel("Progress:")
        self.statusbar.addWidget(self.progress_label)
        self.statusbar.addWidget(self.progress_bar)
        self.setStatusBar(self.statusbar)
        
        self.setWindowTitle("AutoQA Tool (ver.Dev)")
        self.setWindowIcon(QIcon('images/icon/eglab.ico'))
        self.setGeometry(300, 300, 1650, 960)
        
        buttons = "self.pushButton_"
        for n in range(8):
            name = buttons+str(n)
            button = eval(name)
            button.setFixedHeight(80)
            if n == 0:
                button.clicked.connect(self.add_actions)
            elif n == 1:
                button.clicked.connect(self.add_image)
            elif n == 2:
                button.clicked.connect(self.delete_cur_action)
            elif n == 4:
                button.clicked.connect(self.set_interval)
            elif n == 6:
                button.clicked.connect(self.start_routine)
            elif n == 7:
                button.clicked.connect(self.stop_routine)
            
        
        self.gui_search.clicked.connect(self.match_gui_templates) 
        
        # self.gui_img_files = []
        # self.gui_sub_folders = []
        folder_dir = 'screen/UI'
        self.gui_folder.setText(folder_dir)
        self.gui_img_files = get_all_images(folder_dir)
        self.gui_sub_folders = get_subfolders(folder_dir)
            
        self.gui_browser.clicked.connect(self.open_browser) 
        
        self.process_name.setText('Geometry Dash')
        self.process_name.setEnabled(False)
        self.pcheck.stateChanged.connect(self.toggle_process_name)
        
        self.preset_combo.addItems([f"프리셋 {i}" for i in range(0, 10)])  # 예시 프리셋 추가
        self.preset_combo.currentIndexChanged.connect(self.update_preset)                
        self.log_text.setReadOnly(True)

        self.worker = None
        self.gui_pixmap = QPixmap()
        
    def resizeEvent(self, event):
        scaled_pixmap = self.gui_pixmap.scaled(self.gui_result.size(), Qt.KeepAspectRatio)
        self.gui_result.setPixmap(scaled_pixmap)

        
    def toggle_process_name(self, state):
        if state == 2:  # QCheckBox가 체크된 상태
            self.process_name.setEnabled(False)
        else:  # QCheckBox가 체크 해제된 상태
            self.process_name.setEnabled(True)
    
    def start_routine(self):
        if self.worker is None or not self.worker.isRunning():
            actions = [self.list_widget.item(i) for i in range(self.list_widget.count())]
            self.worker = Routine(actions)
            self.log_text.append("루틴이 시작되었습니다.")
            self.worker.start()
            
            # 윈도우 창 최소화            
            self.showMinimized()

    def stop_routine(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.log_text.append("루틴이 정지되었습니다.")
            self.worker.wait()
            
    def set_interval(self):
        dialog = IntervalDialog(self)
        result = dialog.exec_()

        if result == QtWidgets.QDialog.Accepted:
            item = QtWidgets.QListWidgetItem(f"{dialog.interval_line.text()} 초 대기")
            item.setData(Qt.UserRole, [4, [dialog.interval_line.text()]])

            self.log_text.append(f"대기Action 추가 : {item.data(Qt.UserRole)}")
            self.list_widget.addItem(item)
            
    def delete_cur_action(self):
        selectedRow = self.list_widget.currentRow()
        if selectedRow != -1:
            selectedItem = self.list_widget.item(selectedRow)
            self.log_text.append(f"제거 : {selectedItem.text()}")
            self.list_widget.takeItem(selectedRow)
          
    def update_preset(self, idx):
        self.list_widget.clear()
        self.preset_index = idx
        c_list = self.action_list[idx]
        
        for action in c_list:
            if action[0] == 0:
                self.list_widget.addItem(f"클릭 (x,y : {action[1][0]}, {action[1][1]})")
    
    def add_actions(self):
        dialog = ActionDialog(self)  # ActionDialog 생성
        result = dialog.exec_()  # 다이얼로그 실행
        item = None

        if result == QtWidgets.QDialog.Accepted:
            if dialog.input_toggle == 0:
                item = QtWidgets.QListWidgetItem(f"좌표 클릭 ({dialog.mousePos[0]}, {dialog.mousePos[1]})")
                item.setData(Qt.UserRole, [0, dialog.mousePos])
                self.log_text.append(f"클릭Action 추가 : ({item.data(Qt.UserRole)})")
            elif dialog.input_toggle == 1:
                item = QtWidgets.QListWidgetItem(f"키 입력 ({dialog.input_key.text()})")
                item.setData(Qt.UserRole, [1, [dialog.input_key.text()]])
                self.log_text.append(f"키Action 추가 : ({item.data(Qt.UserRole)})")
            self.list_widget.addItem(item)
        
    
    def add_image(self):
        dialog = ImageDialog(self)
        result = dialog.exec_()
        item = None

        if result == QtWidgets.QDialog.Accepted:
            item = QtWidgets.QListWidgetItem(f"이미지 클릭 {os.path.basename(dialog._imgPath)}")
            item.setData(Qt.UserRole, [2, [dialog._imgPath, dialog.confidence.value()]])

            self.log_text.append(f"이미지클릭Action 추가{os.path.basename(dialog._imgPath)}, 유사도:{dialog.confidence.value()}")
            self.list_widget.addItem(item)

    def update_status_bar(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat("{0}%".format(int(current/total*100)))
        self.gui_search.setEnabled(False)

    def on_finished(self, result_image):
        self.progress_bar.setFormat(" Completed!")
        self.gui_search.setEnabled(True)
        
        self.gui_pixmap = self.view_resized_img_on_widget(result_image,self.gui_result.width(),self.gui_result.height())
        self.gui_result.setPixmap(self.gui_pixmap)
        

    def open_browser(self):
        folder_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder_dir:
            self.gui_folder.setText(folder_dir)
            self.gui_img_files = get_all_images(folder_dir)
            self.gui_sub_folders = get_subfolders(folder_dir)


    def match_gui_templates(self): 
        # folder_dir = 'screen/UI'
        if len(self.gui_img_files) < 1:
            self.progress_bar.setFormat(f"경로 '{self.gui_folder.text()}' 상에 폴더가 비어있습니다.")
            return
        
        # 윈도우 창 최소화            
        self.showMinimized()
        QThread.msleep(int(100))
        # 사용 예제
        # templates = [cv2.imread(file, 0) for file in self.gui_img_files]
        templates = []
        for file in self.gui_img_files:
            name = file.split('/')[-1]
            name = name.split('.')[0]
            template = {}
            template[name] = cv2.imread(file,0)
            templates.append(template)
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor) # screenshot 
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
            self.matcher = UITemplateMatcher(image,templates, scale_range=(0.9, 1.2), scale_step=0.1, threshold=0.7)
            self.matcher.update_progress.connect(self.update_status_bar)  # 시그널 연결
            self.matcher.finished.connect(self.on_finished)  # 작업 완료 시그널 연결
            self.matcher.start()  # QThread 시작

    
    def view_resized_img_on_widget(self,frame, width, height):
        # PyQt5 프레임에 표시
        h, w, _ = frame.shape # _ 는 channel
        bytes_per_line = 3 * w

        frame_bytes = frame.tobytes()
        q_img = QImage(frame_bytes, w, h, bytes_per_line, QImage.Format_RGB888)

        # QImage를 리사이즈
        resized_img = q_img.scaled(width,height)
        return QPixmap.fromImage(resized_img)
    
    def closeEvent(self, event):
        event.accept()
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
