import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon

window_ui = 'main_window.ui'

class mainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()

        # UI 파일 로드
        try:
            uic.loadUi(window_ui, self)
            
            self.centralWidget.setLayout(self.main_layout)
            
            self.setWindowTitle("AutoQA Tool (ver.Dev)")
            self.setWindowIcon(QIcon('images/icon/eglab.ico'))
            self.setGeometry(300, 300, 1650, 960)
            
        except Exception as e:
            print(f"UI 파일을 로드하는 중 오류가 발생했습니다: {e}")
            sys.exit(1)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec_())
