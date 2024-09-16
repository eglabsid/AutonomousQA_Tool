from PyQt5.QtWidgets import QVBoxLayout, QDialog, QRadioButton, QLineEdit, QLabel, QPushButton, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QCursor, QIntValidator
from PyQt5.QtCore import QTimer

class ActionDialog(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.initUI()
        self.input_toggle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.track_mouse)
        self.timer.start(100)

    def initUI(self):
        self.setWindowTitle("동작 추가")
        self.resize(400, 200)

        layout = QVBoxLayout()
        layoutH1 = QHBoxLayout()
        layoutH2 = QHBoxLayout()
        layoutH3 = QHBoxLayout()

        self.confirm_button = QPushButton("확인", self)
        self.confirm_button.clicked.connect(self.acceptbtn)
        self.cancel_button = QPushButton("취소", self)
        self.cancel_button.clicked.connect(self.reject)

        self.radio_pos = QRadioButton("마우스 클릭", self)
        self.radio_key = QRadioButton("키 입력", self)

        self.radio_pos.toggled.connect(self.update_input_field)
        self.radio_key.toggled.connect(self.update_input_field)

        self.input_x_label = QLabel("x좌표 : ", self)
        self.input_x = QLineEdit(self)
        self.input_x.setPlaceholderText("X좌표")
        self.input_x.setValidator(QIntValidator(-10000, 10000, self))
        self.input_y_label = QLabel("y좌표 : ", self)
        self.input_y = QLineEdit(self)
        self.input_y.setPlaceholderText("Y좌표")
        self.input_y.setValidator(QIntValidator(-10000,10000, self))

        self.input_key_label = QLabel("키 입력 : ", self)
        self.input_key = QLineEdit(self)
        self.input_key.setMaxLength(1)
        self.input_key.setPlaceholderText("특정 Key")

        self.mouse_pos = QLabel("마우스 좌표",self)

        layoutH1.addWidget(self.radio_pos)
        layoutH1.addWidget(self.input_x_label)
        layoutH1.addWidget(self.input_x)
        layoutH1.addWidget(self.input_y_label)
        layoutH1.addWidget(self.input_y)
        layout.addLayout(layoutH1)

        layoutH2.addWidget(self.radio_key)
        layoutH2.addWidget(self.input_key_label)
        layoutH2.addWidget(self.input_key)
        layout.addLayout(layoutH2)

        layout.addWidget(self.mouse_pos)

        layoutH3.addWidget(self.confirm_button)
        layoutH3.addWidget(self.cancel_button)
        layout.addLayout(layoutH3)

        self.radio_pos.setChecked(True)
        self.setLayout(layout)

    def update_input_field(self):
        if self.radio_pos.isChecked():
            self.input_toggle = 0
            self.input_key.setDisabled(True)
            self.input_x.setDisabled(False)
            self.input_y.setDisabled(False)
        if self.radio_key.isChecked():
            self.input_toggle = 1
            self.input_key.setDisabled(False)
            self.input_x.setDisabled(True)
            self.input_y.setDisabled(True)

    def track_mouse(self):
        mouse_pos = QCursor.pos()
        self.mouse_pos.setText(f"현재 마우스 좌표 : ({mouse_pos.x()}, {mouse_pos.y()})")

    def acceptbtn(self):
        if self.input_toggle == 0:
            if not self.input_x.text() or not self.input_y.text():
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("좌표값이 비어있습니다.")
                msg.setWindowTitle("Error")
                msg.resize(600,200)
                msg.exec_()
            else:
                self.accept()

        if self.input_toggle == 1:
            if not self.input_key.text():
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("키 값이 비어있습니다.")
                msg.setWindowTitle("Error")
                msg.resize(600, 200)
                msg.exec_()
            else:
                self.accept()
