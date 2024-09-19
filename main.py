import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from src.action_dialog import ActionDialog
from src.image_dialog import ImageDialog
from src.interval_dialog import IntervalDialog

from src.routine import Routine

from pynput import mouse

window_ui = 'main_window.ui'

class mainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()

        # UI 파일 로드
        uic.loadUi(window_ui, self)
        
        self.centralWidget.setLayout(self.main_layout)
        
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
                button.clicked.connect(self.delete_curAction)
            elif n == 4:
                button.clicked.connect(self.wait_action)
            elif n == 6:
                button.clicked.connect(self.start_routine)
            elif n == 7:
                button.clicked.connect(self.stop_routine)
        
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
            
    def wait_action(self):
        dialog = IntervalDialog(self)
        result = dialog.exec_()

        if result == QtWidgets.QDialog.Accepted:
            item = QtWidgets.QListWidgetItem(f"{dialog.wait_Line.text()} 초 대기")
            item.setData(Qt.UserRole, [4, [dialog.wait_Line.text()]])

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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
