
from PyQt5.QtCore import QThread, Qt, pyqtSignal
import numpy as np

from enum import Enum
class PatternType(Enum):
    CLICK = 0
    TYPING = 1
    MATCH = 2
    DELAY = 3

class StrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
    
class SendKey(StrEnum):
    ENTER = '{ENTER}'
    TAB = '{TAB}'
    ESC = '{ESC}'
    SPACE = '{SPACE}'
    BACKSPACE = '{BACKSPACE}'
    DELETE = '{DELETE}'
    UP = '{DOWN}'
    LEFT = '{RIGHT}'
    Shift = '+'
    Ctrl = '^'
    Alt = '%'


class RepeatPattern(QThread):

    message = pyqtSignal(str)
    
    def __init__(self, items, handler):
        super().__init__()
        self.running = True
        self.items = items
        self.handler = handler
        
        
        # self.redirect = pyqtSignal(str) 

    def run(self): # ctrl+esc 로 종료 메시지
        while self.running:
            
            if len(self.items) < 1:
                self.running = False
                break
                    
            # for item in self.items:
            item = self.items.pop(0)
                
            if not self.running:
                break
            
            data = item.data(Qt.UserRole)
            delay = 2.5
            if data:
                ptype = data[0]
                dinfo = data[1]
                if ptype == PatternType.CLICK:
                    img = dinfo[0]
                    coord = dinfo[1]
                    self.handler.mouseclick('left',coord)
                    self.message.emit(img)
                elif ptype == PatternType.TYPING:
                    # pyautogui.press(b)   # 키 입력
                    # inputManager.release_key(b)
                    pass
                elif ptype == PatternType.MATCH:
                    
                    
                    # image_path = os.path.abspath(b[0])
                    image_path = dinfo[0]
                    print(f"이미지 절대 경로: {image_path}")
                elif ptype == PatternType.DELAY:
                    delay = int(dinfo)
                print(f"실행 {data}")
                
            self.msleep(int(1000*delay))

    def stop(self):
        self.running = False
