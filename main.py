import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from src.action_dialog import ActionDialog
from src.image_dialog import ImageDialog


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
        
        self.preset_combo.addItems([f"프리셋 {i}" for i in range(0, 10)])  # 예시 프리셋 추가
        self.preset_combo.currentIndexChanged.connect(self.update_preset)                
        self.log_text.setReadOnly(True)
    
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
            item = QtWidgets.QListWidgetItem(f"이미지 클릭 {os.path.basename(dialog.image_Path)}")
            item.setData(Qt.UserRole, [2, [dialog.image_Path, dialog.confidence.value()]])

            self.log_text.append(f"이미지클릭Action 추가{os.path.basename(dialog.image_Path)}, 유사도:{dialog.confidence.value()}")
            self.list_widget.addItem(item)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
