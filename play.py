import sys,os
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

window_ui = 'play.ui'

class playWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(playWindow, self).__init__()

        # UI 파일 로드
        uic.loadUi(window_ui, self)
        
        self.centralWidget.setLayout(self.main_layout)
        
        self.setWindowTitle("Image based Playing Tool (ver.Dev)")
        self.setWindowIcon(QIcon('images/icon/eglab.ico'))
        self.setGeometry(300, 300, 600, 800)
        
        buttons = "self.pushButton_"
        for n in range(4):
            name = buttons+str(n)
            button = eval(name)
            button.setFixedHeight(80)
            # if n == 0:
            #     button.clicked.connect(self.add_actions)
        
        
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = playWindow()
    window.show()
    sys.exit(app.exec_())
