from PyQt5.QtWidgets import QVBoxLayout, QDialog, QMessageBox, QLineEdit, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtGui import QDoubleValidator

class WaitDialog(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("대기시간 추가")
        self.resize(400, 200)

        layout = QVBoxLayout()
        layoutH1 = QHBoxLayout()
        layoutH2 = QHBoxLayout()

        self.wait_label = QLabel("대기시간(초) : ", self)
        self.wait_Line = QLineEdit(self)
        self.wait_Line.setValidator(QDoubleValidator(0.0, 10000.0, 3, self))
        self.wait_Line.setMaxLength(5)

        self.confirm_button = QPushButton("확인", self)
        self.confirm_button.clicked.connect(self.acceptbtn)
        self.confirm_button.setDefault(True)
        self.cancel_button = QPushButton("취소", self)
        self.cancel_button.clicked.connect(self.reject)

        layoutH1.addWidget(self.wait_label)
        layoutH1.addWidget(self.wait_Line)
        layout.addLayout(layoutH1)

        layoutH2.addWidget(self.confirm_button)
        layoutH2.addWidget(self.cancel_button)
        layout.addLayout(layoutH2)

        self.setLayout(layout)

    def acceptbtn(self):
        if not self.wait_Line:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("입력값이 비어있습니다.")
            msg.setWindowTitle("Error")
            msg.resize(600, 200)
            msg.exec_()
        else:
            self.accept()
