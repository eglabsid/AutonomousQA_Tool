import os.path

from PyQt5.QtWidgets import QVBoxLayout, QDialog, QMessageBox, QLineEdit, QLabel, QPushButton, QHBoxLayout, QFileDialog, QSpinBox

class ImageDialog(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.initUI()
        self.image_Path = None

    def initUI(self):
        self.setWindowTitle("이미지 클릭 추가")
        self.resize(400, 200)

        layout = QVBoxLayout()
        layoutH1 = QHBoxLayout()
        layoutH2 = QHBoxLayout()
        layoutH3 = QHBoxLayout()

        self.img_label = QLabel("이미지 : ", self)
        self.img_path = QLineEdit(self)

        self.img_path.setReadOnly(True)
        self.img_path.setPlaceholderText("이미지 경로")

        self.browsebtn = QPushButton("찾아보기", self)
        self.browsebtn.clicked.connect(self.select_image)
        self.browsebtn.setDefault(False)

        self.confidence_label = QLabel("유사도 : ")
        self.confidence = QSpinBox(self)
        self.confidence.setRange(50,100)
        self.confidence.setValue(80)

        self.confirm_button = QPushButton("확인", self)
        self.confirm_button.clicked.connect(self.acceptbtn)
        self.confirm_button.setDefault(True)
        self.cancel_button = QPushButton("취소", self)
        self.cancel_button.clicked.connect(self.reject)

        layoutH1.addWidget(self.img_label)
        layoutH1.addWidget(self.img_path)
        layoutH1.addWidget(self.browsebtn)
        layout.addLayout(layoutH1)

        layoutH2.addWidget(self.confidence_label)
        layoutH2.addWidget(self.confidence)
        layoutH2.addStretch(2)
        layout.addLayout(layoutH2)

        layoutH3.addWidget(self.confirm_button)
        layoutH3.addWidget(self.cancel_button)
        layout.addLayout(layoutH3)

        self.setLayout(layout)



    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # 파일을 읽기 전용으로 열기
        file_filter = "Images (*.png *.jpg *.jpeg *.bmp)"  # 이미지 파일 필터 설정
        image_path, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", file_filter, options=options)

        if image_path:
            project_dir = os.path.dirname(os.path.abspath(__file__))

            relative_image_path = os.path.relpath(image_path, project_dir)
            image_path = relative_image_path
            self.img_path.setText(image_path)
            self.image_Path = image_path
        else:
            return None

    def acceptbtn(self):
        if not self.image_Path:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("선택한 이미지가 없습니다.")
            msg.setWindowTitle("Error")
            msg.resize(600, 200)
            msg.exec_()
        else:
            self.accept()









