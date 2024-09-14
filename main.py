import sys
from PyQt5 import uic, QtWidgets

window_ui = 'main_window.ui'

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        # UI 파일 로드
        try:
            uic.loadUi(window_ui, self)
        except Exception as e:
            print(f"UI 파일을 로드하는 중 오류가 발생했습니다: {e}")
            sys.exit(1)
        
        # 버튼을 불러와서 클릭 이벤트 연결
        try:
            print('do something')
            # self.my_button = self.findChild(QtWidgets.QPushButton, 'my_button')  # 'my_button'은 UI에서 설정한 버튼의 objectName
            # self.my_button.clicked.connect(self.on_button_click)
        except AttributeError:
            print("버튼 객체를 찾을 수 없습니다.")
            sys.exit(1)

    def on_button_click(self):
        try:
            print("버튼이 클릭되었습니다!")
            # 여기서 원하는 동작을 추가하면 됩니다
        except Exception as e:
            print(f"버튼 클릭 처리 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
