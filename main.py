import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

from src.action_dialog import ActionDialog
from src.image_dialog import ImageDialog
from src.interval_dialog import IntervalDialog

from utils.process_handler import WindowProcessHandler
from utils.repeat_pattern import RepeatPattern, ItemType, SendKey
from utils.template_matcher import UITemplateMatcher, TemplateMatcher, get_all_images, get_subfolders

# opencv
import cv2

import re

window_ui = 'main_window.ui'

class mainWindow(QtWidgets.QMainWindow):
    
    rematch = pyqtSignal(UITemplateMatcher)

    def __init__(self):
        super(mainWindow, self).__init__()
        self.initUI()
        
    def initUI(self):
        # UI 파일 로드
        uic.loadUi(window_ui, self)
        self.centralWidget.setLayout(self.main_layout)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_label = QtWidgets.QLabel("Recognizes the GUI")
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
                button.clicked.connect(self.set_delay)
            elif n == 6:
                button.clicked.connect(self.start_routine)
            elif n == 7:
                button.clicked.connect(self.stop_routine)
            
        # gui 및 process 확인 기능
        self.confirm_process.clicked.connect(self.confirm_running_process)
        self.gui_search.clicked.connect(self.match_gui_templates) 
        
        self.gui_resource_root_dir = f"{os.getcwd()}/screen/UI" # 단일 path
        self.gui_resource_root_dir = self.gui_resource_root_dir.replace("\\","/")
        
        self.gui_img_files, self.gui_subfolders=self.update_files_in_directory(self.gui_resource_root_dir)
        self.gui_matched_subfolders = []
        self.gui_matchinfo = []
        
        self.gui_browser.clicked.connect(self.open_browser) 
        
        self.gui_pixmap = QPixmap()
        
        self.auto_play.clicked.connect(self.play_auto)
        
        # 프로세스 handler
        # process_name = 'Geometry Dash'
        # self.process_name.setText(process_name)
        # self.process_name.setEnabled(False)
        # self.pcheck.stateChanged.connect(self.toggle_process_name)
        # process_name = 'GeometryDash.exe'
        self.handler = WindowProcessHandler()
        
        # List-up Running Process
        proc_lst = self.handler.get_running_process_list()
        for proc in proc_lst:
            self.process_list.addItem(f"{proc['name']}")
        self.process_list.currentIndexChanged.connect(self.update_process_list)                
        self.handler.process_name = self.process_list.currentText()
        
        self.preset_combo.addItems([f"프리셋 {i}" for i in range(0, 10)])  # 예시 프리셋 추가
        self.preset_combo.currentIndexChanged.connect(self.update_preset)                
        self.log_text.setReadOnly(True)

        self.repeater = RepeatPattern()
        self.repeater.receive_handler(self.handler)
        self.repeater.subfolder.connect(self.update_decision)
        self.repeater.finished.connect(self.clear)
        
        self.rematch.connect(self.repeater.receive_matcher)
        
        self.is_rematch = False
        self.is_auto = False

    # Event Section
    def resizeEvent(self, event):
        scaled_pixmap = self.gui_pixmap.scaled(self.gui_result.size(), Qt.KeepAspectRatio)
        self.gui_result.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        event.accept()
    
    def clear(self,finished):
        self.action_sequence.clear()
        self.log_text.clear()
        self.log_text.setText(finished)
        self.is_auto = False
        self.showNormal()
    
    def play_auto(self):
        self.confirm_running_process()
        self.match_gui_templates()
        self.is_auto = True
        
    
    # Function Section
    def start_routine(self):
        if self.repeater is None or not self.repeater.isRunning():
            # actions = [self.action_sequence.item(i) for i in range(self.action_sequence.count())]
            items = [item.data(Qt.UserRole) for item in self.action_sequence.findItems("", Qt.MatchContains)]
            self.repeater.receive_items(items)
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


    def open_browser(self):
        self.gui_resource_root_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Directory')
        if self.gui_resource_root_dir:
            self.gui_img_files, self.gui_subfolders = self.update_files_in_directory(self.gui_resource_root_dir)

    def make_gui_template(self,img_files):
         # 사용 예제
        templates = []
        for file in img_files:
            # name = file.split('/')[-1]
            name = file.split('.')[0]
            print(name)
            template = {}
            template[name] = cv2.imread(file,0)
            templates.append(template)
        return templates
            
    def match_gui_templates(self):
        # process 접근
        if self.handler.window_process == None:
            # msg = self.handler.connect_application_by_handler() #pywinauto 이용방법
            msg = self.handler.connect_application_by_process_name(self.process_list.currentText())
            self.log_text.append(msg)
        else:
            self.handler.window_process.activate()
            
        # folder_dir = 'screen/UI'
        if len(self.gui_img_files) < 1:
            self.progress_bar.setFormat(f"경로 :'{self.gui_folder.text()}' 내에 이미지 파일이 없습니다.")
            return
        
        # 윈도우 화면 전체 캡쳐
        QThread.msleep(int(300))
        image = self.handler.caputer_monitor_to_cv_img()
        
        # GUI 폴더 경로상의 이미지 
        templates = self.make_gui_template(self.gui_img_files)

        # threshhold = 0.9
        self.matcher = UITemplateMatcher(image,templates, scale_range=(0.7, 1.0), scale_step=0.1)#,threshold=threshhold)
        self.matcher.update_progress.connect(self.update_status_bar)  # 시그널 연결
        self.matcher.finished.connect(self.on_finished)  # 작업 완료 시그널 연결
        self.matcher.start()  # QThread 시작
        self.rematch.emit(self.matcher) 

    def on_finished(self, result_image):
        
        # GUI 상의 이미지 검출이 되지 않았을 때, 재시도구간
        if len(self.matcher.matches) == 0:
            
            # 1회 재시도 - 스크린 미 캡쳐되었을경우 
            if self.matcher.iter < 1:
                self.match_gui_templates()
                self.matcher.iter += 1
                self.repeater.stop()
                pass
            
            # 모든 폴더에 대해 재 시도
            if len(self.gui_subfolders)>0:
                subfodler = self.gui_subfolders.pop()
                search = self.gui_resource_root_dir+f"/{subfodler}"
                # GUI 폴더 경로상의 이미지 
                self.gui_img_files = get_all_images(search)
                self.match_gui_templates()
                # self.repeater.stop()  
                pass
                
            # 루트 한번
            if self.matcher.iter == 1:
                self.gui_img_files = get_all_images(self.gui_resource_root_dir)
                self.match_gui_templates()
                self.matcher.iter += 1
                pass
        
        self.matcher.iter = 0
        
        self.progress_bar.setFormat(" I got it.! ")
        self.gui_search.setEnabled(True)
        
        
        # gui 관련 파라미터 업데이트
        self.update_gui_parameters(self.matcher)
        
        self.gui_pixmap = self.view_resized_img_on_widget(result_image,self.gui_result.width(),self.gui_result.height())
        self.gui_result.setPixmap(self.gui_pixmap)
        
        # template matching 결과 Item 업데이트
        self.update_action_sequence(self.gui_matchinfo,self.gui_subfolders)
        
        if self.is_rematch:
            self.is_rematch = False
            
            items = [item.data(Qt.UserRole) for item in self.action_sequence.findItems("", Qt.MatchContains)]
            self.repeater.receive_items(items)
            self.repeater.start()
        
        if self.is_auto:
            self.start_routine()    
    
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
        # List-up Running Process
        proc_lst = self.handler.get_running_process_list()
        for proc in proc_lst:
            self.process_list.addItem(f"{proc['name']}")
        
        msg = f"실행중인 프로세스를 다시 확인합니다."
        self.log_text.append(msg)    
        # cur_proc = self.process_list.currentText()
        # # msg = self.handler.connect_application_by_handler()
        # msg = self.handler.connect_application_by_process_name(cur_proc)
        # self.log_text.append(msg)
            
    def set_delay(self):
        dialog = IntervalDialog(self)
        result = dialog.exec_()

        if result == QtWidgets.QDialog.Accepted:
            item = QtWidgets.QListWidgetItem(f"{dialog.interval_line.text()} 초 대기")
            item.setData(Qt.UserRole, [ItemType.DELAY, [dialog.interval_line.text()]])

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
                item.setData(Qt.UserRole, [ItemType.CLICK, dialog.mousePos])
                self.log_text.append(f"클릭Action 추가 : ({item.data(Qt.UserRole)})")
            elif dialog.input_toggle == 1:
                item = QtWidgets.QListWidgetItem(f"키 입력 ({dialog.input_key.text()})")
                item.setData(Qt.UserRole, [ItemType.TYPING, [dialog.input_key.text()]])
                self.log_text.append(f"키Action 추가 : ({item.data(Qt.UserRole)})")
            self.action_sequence.addItem(item)
        
    def add_image(self):
        dialog = ImageDialog(self)
        result = dialog.exec_()
        item = None

        if result == QtWidgets.QDialog.Accepted:
            item = QtWidgets.QListWidgetItem(f"이미지 클릭 {os.path.basename(dialog._imgPath)}")
            item.setData(Qt.UserRole, [ItemType.REMATCH, [dialog._imgPath, dialog.confidence.value()]])

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
        
        if current % 2 == 0:
            self.progress_bar.setFormat("....")
        else:
            self.progress_bar.setFormat("...")
        
        self.gui_search.setEnabled(False)

    def update_gui_parameters(self,matcher):
        # gui 관련 파라미터 초기화
        self.gui_matchinfo.clear()
        self.log_text.clear()
        
        # gui 관련 파라미터 설정
        for loc, scale, score, template_tuple in matcher.matches:    
            path = template_tuple[0].split('\\')[-1]
            name = path.split('.')[0]
            h,w = template_tuple[1].shape # 중심좌표
            gui_dic = {} # gui 유사도, 좌표, ui name 탐색
            mc_loc = [loc[0] + int(w * scale * 0.5), loc[1] + int(h * scale * 0.5)]
            gui_dic[name] = ( score , mc_loc )
            self.gui_matchinfo.append(gui_dic)
      
            msg = f"파일명 : {template_tuple[0]}, 좌표 : ( {mc_loc[0]} , {mc_loc[1]} )"
            # print(msg)
            self.log_text.append(msg)
    
    def update_action_sequence(self,match_info,sub_folders):
        self.action_sequence.clear()
        self.log_text.clear()
        for info in match_info:
            name, (score, mc_loc) = [[k,v] for k,v in info.items()][0]
            
            # 정규 표현식 패턴
            # 정규 표현식을 사용하여 문자열 검색
            is_match = False
            for sub_folder in sub_folders:
                pattern = re.escape(name) # 특정 문자열을 정규 표현식 패턴으로 변환
                search = re.search(pattern,sub_folder)
                if search:
                    is_match = True
                    break
            
            msg = f"{name}, [{ItemType.CLICK.name}, Coord:{mc_loc}, Conf:{100-int(score*100)}]"
            if is_match:
                msg = f"{name}, [{ItemType.REMATCH.name}, Coord:{mc_loc}, Conf:{100-int(score*100)}]"
            
            item = QtWidgets.QListWidgetItem(msg)
            data = [ItemType.CLICK, [name, mc_loc]]
            if is_match:
                data = [ItemType.REMATCH, [name, mc_loc]]
            
            item.setData(Qt.UserRole, data)
            self.action_sequence.addItem(item)
            msg = f"<< Add : [{name}, {ItemType.CLICK.name}]"
            if is_match:
                msg = f"<< Add : [{name}, {ItemType.REMATCH.name}]"
                
            self.log_text.append(msg)
    
    def update_files_in_directory(self,root_path, search_path = ""):
        
        img_files_in_dir = get_all_images(root_path)
        subfolders = get_subfolders(root_path) # update subfolder list
        if len(subfolders) > 0:
            if search_path != "":
                root_path = root_path.replace(f"/{search_path}","")
            subfolders = [folder.replace(f"{root_path}","") for folder in subfolders]
            
        self.gui_folder.setText(root_path)
        return img_files_in_dir, subfolders

    def update_subfolders(self,subfolder):
        search = f"\\{subfolder}"
        if search in self.gui_subfolders:
            self.gui_subfolders.remove(search)
            self.gui_matched_subfolders.append(search)
            
        gui_resource_search_path = f"{self.gui_resource_root_dir}/{subfolder}"
        self.gui_img_files, subfolders = self.update_files_in_directory(gui_resource_search_path,subfolder)
        self.gui_subfolders.extend(subfolders)
        
    def update_decision(self,subfolder):
        
        self.is_rematch = True
        
        self.update_subfolders(subfolder)
        
        # 사용 예제
        templates = self.make_gui_template(self.gui_img_files)
        self.handler.window_process.activate()
        QThread.msleep(int(500))
        # 윈도우 화면 전체 캡쳐
        image = self.handler.caputer_monitor_to_cv_img()
        
        self.matcher = UITemplateMatcher(image,templates, scale_range=(0.7, 1.0), scale_step=0.1)#,threshold=threshhold)
        self.matcher.update_progress.connect(self.update_status_bar)  # 시그널 연결
        self.matcher.finished.connect(self.on_finished)  # 작업 완료 시그널 연결
        self.matcher.start()  # QThread 시작
        # QThread 쪽으로 호출
        self.rematch.emit(self.matcher)
        
    
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
