
from PyQt5.QtCore import QThread, Qt, pyqtSignal
import numpy as np

from enum import Enum
class PatternType(Enum):
    CLICK = 0
    TYPING = 1
    MATCH = 2
    DELAY = 3
    


class RepeatPattern(QThread):

    def __init__(self, items, handler):
        super().__init__()
        self.running = True
        self.items = items
        self.handler = handler
        
        self.message = pyqtSignal(str)
        # self.detected_objects = pyqtSignal(str, list) 

    def run(self): # ctrl+esc 로 종료 메시지
        while self.running:
            for item in self.items:
                if not self.running:
                    break
                data = item.data(Qt.UserRole)

                if data:
                    a = data[0]
                    b = data[1]
                    if a == 0:
                        # pyautogui.click(int(b[0]), int(b[1]))    # 클릭
                        pos_xy = list(map(int,b))
                        # inputManager.move_mouse(pos_xy[0],pos_xy[1])
                        # inputManager.click_mouse()
                    elif a == 1:
                        # pyautogui.press(b)   # 키 입력
                        # inputManager.release_key(b)
                        pass
                    elif a == 2:
                        
                        self.finished.emit("이미지 탐색중")
                        # image_path = os.path.abspath(b[0])
                        image_path = b[0]
                        print(f"이미지 절대 경로: {image_path}")

                        
                    print(f"실행 {data}")
                    
                # self.msleep(int(1000*delay))

    def stop(self):
        self.running = False
