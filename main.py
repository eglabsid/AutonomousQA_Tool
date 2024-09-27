import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from src.action_dialog import ActionDialog
from src.image_dialog import ImageDialog
from src.interval_dialog import IntervalDialog

from utils.process_handler import WindowProcessHandler
from utils.repeat_pattern import RepeatPattern, PatternType
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
        self.progress_label = QtWidgets.QLabel("Detecting GUI Images :")
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
            
        # gui 및 process 확인 기능
        self.confirm_process.clicked.connect(self.confirm_running_process)
        self.gui_search.clicked.connect(self.match_gui_templates) 
        
        folder_dir = 'screen/UI'
        self.gui_folder.setText(folder_dir)
        self.gui_img_files = get_all_images(folder_dir)
        self.gui_sub_folders = get_subfolders(folder_dir)
        self.gui_info = []
        self.gui_resource_path = ""
        self.gui_browser.clicked.connect(self.open_browser) 
        
        self.gui_pixmap = QPixmap()
        
        # 프로세스 handler
        # process_name = 'Geometry Dash'
        # self.process_name.setText(process_name)
        # self.process_name.setEnabled(False)
        # self.pcheck.stateChanged.connect(self.toggle_process_name)
        process_name = 'GeometryDash.exe'
        self.handler = WindowProcessHandler()
        self.handler.process_name = process_name
        
        # List-up Running Process
        proc_lst = self.handler.get_running_process_list()
        for proc in proc_lst:
            self.process_list.addItem(f"{proc['name']}")
        self.process_list.currentIndexChanged.connect(self.update_process_list)                
        
        self.preset_combo.addItems([f"프리셋 {i}" for i in range(0, 10)])  # 예시 프리셋 추가
        self.preset_combo.currentIndexChanged.connect(self.update_preset)                
        self.log_text.setReadOnly(True)

        self.repeater = None

    # Event Section
    def resizeEvent(self, event):
        scaled_pixmap = self.gui_pixmap.scaled(self.gui_result.size(), Qt.KeepAspectRatio)
        self.gui_result.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        event.accept()
    
    # Function Section
    def start_routine(self):
        if self.repeater is None or not self.repeater.isRunning():
            actions = [self.action_sequence.item(i) for i in range(self.action_sequence.count())]
            self.repeater = RepeatPattern(actions, self.handler)
            self.log_text.append("루틴이 시작되었습니다.")
            self.repeater.start()
            
            # 윈도우 창 최소화            
            self.showMinimized()
               
    def stop_routine(self):
        if self.repeater is not None and self.repeater.isRunning():
            self.repeater.stop()
            self.log_text.append("루틴이 정지되었습니다.")
            self.repeater.wait()
            
            self.showNormal()
    
    def on_finished(self, result_image):
        self.progress_bar.setFormat(" Complete ")
        self.gui_search.setEnabled(True)
        
        # gui 관련 파라미터 업데이트
        self.update_gui_parameters()
        
        self.gui_pixmap = self.view_resized_img_on_widget(result_image,self.gui_result.width(),self.gui_result.height())
        self.gui_result.setPixmap(self.gui_pixmap)
        self.showNormal()
        
        self.update_action_Sequence()

    def open_browser(self):
        # options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.ReadOnly  # 파일을 읽기 전용으로 열기
        # file_filter = "Images (*.png *.jpg *.jpeg *.bmp)"  # 이미지 파일 필터 설정
        # folder_dir, _ = QtWidgets.QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", file_filter, options=options)
        folder_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory')
        if folder_dir:
            self.gui_folder.setText(folder_dir)
            self.gui_img_files = get_all_images(folder_dir)
            self.gui_sub_folders = get_subfolders(folder_dir)


    def match_gui_templates(self):
        # process 접근
        # msg = self.handler.connect_application_by_handler()
        msg = self.handler.connect_application_by_process_name(self.process_list.currentText())
        self.log_text.append(msg)
        # folder_dir = 'screen/UI'
        if len(self.gui_img_files) < 1:
            self.progress_bar.setFormat(f"경로 :'{self.gui_folder.text()}' 내에 이미지 파일이 없습니다.")
            return
        
        # 윈도우 창 최소화            
        # self.showMinimized()
        QThread.msleep(int(200))
        # 사용 예제
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

            # threshhold = 0.9
            self.matcher = UITemplateMatcher(image,templates, scale_range=(0.7, 1.2), scale_step=0.1)#,threshold=threshhold)
            self.matcher.update_progress.connect(self.update_status_bar)  # 시그널 연결
            self.matcher.finished.connect(self.on_finished)  # 작업 완료 시그널 연결
            self.matcher.start()  # QThread 시작

    
    def view_resized_img_on_widget(self,frame, width, height):
        # Show on PyQt5 layout
        h, w, _ = frame.shape # _ 는 channel
        bytes_per_line = 3 * w

        frame_bytes = frame.tobytes()
        q_img = QImage(frame_bytes, w, h, bytes_per_line, QImage.Format_RGB888)

        # Re-scale QImage to fit layout
        resized_img = q_img.scaled(width,height)
        return QPixmap.fromImage(resized_img)
    
    def confirm_running_process(self): ##
        cur_proc = self.process_list.currentText()
        # msg = self.handler.connect_application_by_handler()
        msg = self.handler.connect_application_by_process_name(cur_proc)
        self.log_text.append(msg)
            
    def set_interval(self):
        dialog = IntervalDialog(self)
        result = dialog.exec_()

        if result == QtWidgets.QDialog.Accepted:
            item = QtWidgets.QListWidgetItem(f"{dialog.interval_line.text()} 초 대기")
            item.setData(Qt.UserRole, [PatternType.DELAY, [dialog.interval_line.text()]])

            self.log_text.append(f"대기Action 추가 : {item.data(Qt.UserRole)}")
            self.action_sequence.addItem(item)
            
    def delete_cur_action(self):
        selectedRow = self.action_sequence.currentRow()
        if selectedRow != -1:
            selectedItem = self.action_sequence.item(selectedRow)
            self.log_text.append(f"제거 : {selectedItem.text()}")
            self.action_sequence.takeItem(selectedRow)
    
    def add_actions(self):
        dialog = ActionDialog(self)  # ActionDialog 생성
        result = dialog.exec_()  # 다이얼로그 실행
        item = None

        if result == QtWidgets.QDialog.Accepted:
            if dialog.input_toggle == 0:
                item = QtWidgets.QListWidgetItem(f"좌표 클릭 ({dialog.mousePos[0]}, {dialog.mousePos[1]})")
                item.setData(Qt.UserRole, [PatternType.CLICK, dialog.mousePos])
                self.log_text.append(f"클릭Action 추가 : ({item.data(Qt.UserRole)})")
            elif dialog.input_toggle == 1:
                item = QtWidgets.QListWidgetItem(f"키 입력 ({dialog.input_key.text()})")
                item.setData(Qt.UserRole, [PatternType.TYPING, [dialog.input_key.text()]])
                self.log_text.append(f"키Action 추가 : ({item.data(Qt.UserRole)})")
            self.action_sequence.addItem(item)
        
    def add_image(self):
        dialog = ImageDialog(self)
        result = dialog.exec_()
        item = None

        if result == QtWidgets.QDialog.Accepted:
            item = QtWidgets.QListWidgetItem(f"이미지 클릭 {os.path.basename(dialog._imgPath)}")
            item.setData(Qt.UserRole, [PatternType.MATCH, [dialog._imgPath, dialog.confidence.value()]])

            self.log_text.append(f"이미지클릭Action 추가{os.path.basename(dialog._imgPath)}, 유사도:{dialog.confidence.value()}")
            self.action_sequence.addItem(item)

    # Update Section
    def update_preset(self, idx):
        self.action_sequence.clear()
        self.preset_index = idx
        c_list = self.action_list[idx]
        
        for action in c_list:
            if action[0] == 0:
                self.action_sequence.addItem(f"클릭 (x,y : {action[1][0]}, {action[1][1]})")
                
    def update_process_list(self):
        msg = ""
        selected_proc = self.process_list.currentText()
        msg = f"Selected Process: {selected_proc}"
        self.log_text.append(msg)
        
        # self.handler.process_name = selected_proc
        # msg = self.handler.connect_application_by_process_name(selected_proc)
        # self.log_text.append(msg)
        
    def update_status_bar(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat("{0}%".format(int(current/total*100)))
        self.gui_search.setEnabled(False)

    def update_gui_parameters(self):
        # gui 관련 파라미터 초기화
        self.gui_resource_path = ""
        self.gui_info.clear()
        self.log_text.clear()
        
        # gui 관련 파라미터 설정
        for loc, scale, score, template_tuple in self.matcher.matches:
            if self.gui_resource_path == "":
                self.gui_resource_path = template_tuple[0].split('\\')[0]
            path = template_tuple[0].split('\\')[-1]
            name = path.split('.')[0]
            h,w = template_tuple[1].shape # 중심좌표
            gui_dic = {} # gui 유사도, 좌표, ui name 탐색
            mc_loc = [loc[0] + int(w * scale * 0.5), loc[1] + int(h * scale * 0.5)]
            gui_dic[name] = ( score , mc_loc )
            self.gui_info.append(gui_dic)
      
            self.log_text.append(f"파일명 : {template_tuple[0]}, 좌표 : ( {mc_loc[0]} , {mc_loc[1]} )")
    
    def update_action_Sequence(self):
        self.action_sequence.clear()
        self.log_text.clear()
        for info in self.gui_info:
            name, (score, mc_loc) = [[k,v] for k,v in info.items()][0]
            msg = f"Img:{name}, Coord:{mc_loc}, Act:{PatternType.CLICK.name}, Conf:{100-int(score*100)}"
            item = QtWidgets.QListWidgetItem(msg)
            item.setData(Qt.UserRole, [PatternType.CLICK, name, mc_loc])
            self.action_sequence.addItem(item)
            msg = f"<< Add : [{name}, {PatternType.CLICK.name}]"
            self.log_text.append(msg)
            
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
