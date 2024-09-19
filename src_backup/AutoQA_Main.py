import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QListWidget, \
    QScrollArea, QTextEdit, QGridLayout, QLineEdit, QDialog, QListWidgetItem
from PyQt5.QtGui import QIcon
from AutoQA_Action import ActionDialog
from AutoQA_Image import ImageDialog
from AutoQA_Wait import WaitDialog
from AutoQA_Routine import StartRoutine
from PyQt5.QtCore import Qt


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.preset_index = 0

    def initUI(self):
        # 메인 레이아웃 설정
        main_layout = QHBoxLayout()

        # 좌측 레이아웃 - 리스트와 스크롤
        left_layout = QVBoxLayout()

        # 프리셋과 이름 변경 버튼, 저장 버튼, 불러오기 버튼 레이아웃
        top_layout = QHBoxLayout()
        self.preset_combo = QComboBox(self)
        self.preset_combo.addItems([f"프리셋 {i}" for i in range(0, 10)])  # 예시 프리셋 추가
        self.preset_combo.currentIndexChanged.connect(self.update_preset_index)

        self.line_edit = QLineEdit(self)  # LineEdit 추가
        self.name_button = QPushButton("이름변경", self)

        self.save_button = QPushButton("저장", self)  # 저장 버튼 추가
        self.load_button = QPushButton("불러오기", self)  # 불러오기 버튼 추가

        top_layout.addWidget(self.preset_combo)
        top_layout.addWidget(self.line_edit)
        top_layout.addWidget(self.name_button)
        top_layout.addWidget(self.save_button)  # 저장 버튼 레이아웃에 추가
        top_layout.addWidget(self.load_button)  # 불러오기 버튼 레이아웃에 추가

        left_layout.addLayout(top_layout)

        # 동작 리스트
        self.list_widget = QListWidget(self)
        left_layout.addWidget(self.list_widget)

        # 우측 레이아웃
        right_layout = QVBoxLayout()

        # 버튼들 그리드 레이아웃
        grid_layout = QGridLayout()

        button_names = ["동작추가", "이미지 클릭 추가", "선택 삭제", "전투 추가", "대기 시간 추가", "동작 수정", "루틴 시작", "루틴 정지",]
        for i in range(len(button_names)):
            button = QPushButton(button_names[i], self)
            button.setFixedHeight(80)
            grid_layout.addWidget(button, (i // 4) + 1, i % 4)
            if i == 0:
                button.clicked.connect(self.adAct)
            elif i == 1:
                button.clicked.connect(self.adImg)
            elif i == 2:
                button.clicked.connect(self.current_del)
            elif i == 4:
                button.clicked.connect(self.adWait)
            elif i == 6:
                button.clicked.connect(self.start_routine)
            elif i == 7:
                button.clicked.connect(self.stop_Routine)


        right_layout.addLayout(grid_layout)

        # 로그 창
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)

        # 메인 레이아웃에 좌우측 레이아웃 추가
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 윈도우 설정
        self.setLayout(main_layout)
        self.setWindowTitle('EGLAB_AutoQA_Tool')
        self.setWindowIcon(QIcon('eglab.ico'))
        self.setGeometry(300, 300, 1200, 900)
        self.show()

        self.worker = None

        # 테스트 필드

        # 함수들
    def update_preset_index(self, index):
        self.list_widget.clear()
        self.preset_index = index
        c_list = self.action_list[index]
        print(c_list)
        for action in c_list:
            if action[0] == 0:
                self.list_widget.addItem(f"클릭 (x,y : {action[1][0]}, {action[1][1]})")

    def adAct(self):
        dialog = ActionDialog(self)  # ActionDialog 생성
        result = dialog.exec_()  # 다이얼로그 실행
        item = None

        if result == QDialog.Accepted:
            if dialog.input_toggle == 0:
                item = QListWidgetItem(f"좌표 클릭 ({dialog.input_x.text()}, {dialog.input_y.text()})")
                item.setData(Qt.UserRole, [0, [dialog.input_x.text(), dialog.input_y.text()]])
                self.log_text.append(f"클릭Action 추가 : ({item.data(Qt.UserRole)})")
            elif dialog.input_toggle == 1:
                item = QListWidgetItem(f"키 입력 ({dialog.input_key.text()})")
                item.setData(Qt.UserRole, [1, [dialog.input_key.text()]])
                self.log_text.append(f"키Action 추가 : ({item.data(Qt.UserRole)})")
            self.list_widget.addItem(item)

    def adImg(self):
        dialog = ImageDialog(self)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            item = QListWidgetItem(f"이미지 클릭 ({dialog.image_Path}), 유사도:{dialog.confidence.value() / 100}")
            item.setData(Qt.UserRole, [2, [dialog.image_Path, dialog.confidence.value() / 100]])

            self.log_text.append(f"이미지클릭Action 추가 : {item.data(Qt.UserRole)}")
            self.list_widget.addItem(item)

    def current_del(self):
        selectedRow = self.list_widget.currentRow()
        if selectedRow != -1:
            selectedItem = self.list_widget.item(selectedRow)
            self.log_text.append(f"제거 : {selectedItem.text()}")
            self.list_widget.takeItem(selectedRow)

    def adWait(self):
        dialog = WaitDialog(self)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            item = QListWidgetItem(f"{dialog.wait_Line.text()} 초 대기")
            item.setData(Qt.UserRole, [4, [dialog.wait_Line.text()]])

            self.log_text.append(f"대기Action 추가 : {item.data(Qt.UserRole)}")
            self.list_widget.addItem(item)

    def start_routine(self):
        if self.worker is None or not self.worker.isRunning():

            items = [self.list_widget.item(i) for i in range(self.list_widget.count())]
            self.worker = StartRoutine(items)
            self.log_text.append("루틴이 시작되었습니다.")
            self.worker.start()

    def stop_Routine(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.log_text.append("루틴이 정지되었습니다.")
            self.worker.wait()






if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
