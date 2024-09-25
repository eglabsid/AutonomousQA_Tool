import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt

from src.action_dialog import ActionDialog
from src.image_dialog import ImageDialog
from src.interval_dialog import IntervalDialog

from utils.routine import Routine
from utils.template_matcher import UITemplateMatcher, get_all_images, get_subfolders

# opencv
import cv2

window_ui = 'main_window.ui'

class mainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()

        # UI 파일 로드
        uic.loadUi(window_ui, self)
        
        self.centralWidget.setLayout(self.main_layout)
        self.progress_bar = QtWidgets.QProgressBar()
        self.statusbar.addWidget(self.progress_bar)
        self.setStatusBar(self.statusbar)
        
        self.setWindowTitle("AutoQA Tool (ver.Dev)")
        self.setWindowIcon(QIcon('images/icon/eglab.ico'))
        self.setGeometry(300, 300, 1650, 960)
        
        buttons = "self.pushButton_"
        for n in range(9):
            name = buttons+str(n)
            button = eval(name)
            button.setFixedHeight(80)
            if n == 0:
                button.clicked.connect(self.add_actions)
            elif n == 1:
                button.clicked.connect(self.add_image)
            elif n == 2:
                button.clicked.connect(self.delete_curAction)
            elif n == 4:
                button.clicked.connect(self.set_interval)
            elif n == 6:
                button.clicked.connect(self.start_routine)
            elif n == 7:
                button.clicked.connect(self.stop_routine)
            elif n == 8:
                button.clicked.connect(self.run_template_matching) # <------ 여기서부터 진행 2024.09.26 작업
        
        self.preset_combo.addItems([f"프리셋 {i}" for i in range(0, 10)])  # 예시 프리셋 추가
        self.preset_combo.currentIndexChanged.connect(self.update_preset)                
        self.log_text.setReadOnly(True)

        self.worker = None
        
    def start_routine(self):
        if self.worker is None or not self.worker.isRunning():
            actions = [self.list_widget.item(i) for i in range(self.list_widget.count())]
            self.worker = Routine(actions)
            self.log_text.append("루틴이 시작되었습니다.")
            self.worker.start()

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
            
    def delete_curAction(self):
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
        # self.statusbar.showMessage(f"UI Matching Progress: {current/total*100}%")

    def on_finished(self, result_image):
        # cv2.imwrite('result_matchs.jpg', result_image)
        self.statusbar.showMessage("Matching completed!")

    def run_template_matching(self): # <------ 여기서부터 진행 2024.09.26 작업
        folder_dir = 'screen/UI'
        image_files = get_all_images(folder_dir)
        sub_folders = get_subfolders(folder_dir)
        
        # 사용 예제
        templates = [cv2.imread(file, 0) for file in image_files]
        image = cv2.imread('screen/ui_test.jpg')
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.matcher = UITemplateMatcher(image,templates, scale_range=(0.9, 1.2), scale_step=0.1, threshold=0.7, gray_frame=gray_frame)
        self.matcher.update_progress.connect(self.update_status_bar)  # 시그널 연결
        self.matcher.finished.connect(self.on_finished)  # 작업 완료 시그널 연결
        self.matcher.start()  # QThread 시작
    
    def get_widgets_img(self,frame, re_width, re_height):
        # PyQt5 프레임에 표시
        height, width, channel = frame.shape
        bytes_per_line = 3 * width

        frame_bytes = frame.tobytes()
        q_img = QImage(frame_bytes, width, height, bytes_per_line, QImage.Format_RGB888)

        # QImage를 리사이즈
        q_img_resized = q_img.scaled(re_width,re_height)
        return QPixmap.fromImage(q_img_resized)
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
