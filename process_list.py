# import psutil

# # 모든 프로세스 목록 가져오기
# def get_process_list():
#     process_list = []
#     for proc in psutil.process_iter(['pid', 'name', 'username']):
#         process_info = proc.info
#         process_list.append(process_info)
#     return process_list

# # 프로세스 목록 출력
# processes = get_process_list()
# for process in processes:
#     print(process)


import sys
import psutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QVBoxLayout, QWidget, QPushButton, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Process List")
        self.setGeometry(100, 100, 300, 200)

        # QComboBox 생성
        self.combo_box = QComboBox()

        # 프로세스 목록 가져오기 및 QComboBox에 추가
        self.populate_process_list()

        # 선택된 프로세스 정보를 표시할 QLabel 생성
        self.label = QLabel("Selected Process Info")

        # 버튼 생성 및 클릭 시 선택된 프로세스 정보 출력
        self.button = QPushButton("Show Selected Process")
        self.button.clicked.connect(self.show_selected_process)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.combo_box)
        layout.addWidget(self.button)
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def populate_process_list(self):
        process_list = self.get_process_list()
        for process in process_list:
            self.combo_box.addItem(f"{process['pid']} - {process['name']}")

    def get_process_list(self):
        process_list = []
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            process_info = proc.info
            process_list.append(process_info)
        return process_list

    def show_selected_process(self):
        selected_process = self.combo_box.currentText()
        self.label.setText(f"Selected Process: {selected_process}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
