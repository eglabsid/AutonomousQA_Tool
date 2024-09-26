import numpy as np
import cv2
import os

import threading
from PIL import Image

import platform
from functools import wraps

current_os = platform.system()
if current_os == "Windows":
    import ctypes
    from ctypes import windll

    import win32gui
    import win32ui
    import win32con

    from pywinauto import Application, mouse, keyboard

def create_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created successfully!")
    else:
        print(f"Directory '{dir_path}' already exists.")

def os_specific_task(os_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_os = platform.system()
            if current_os == os_name:
                return func(*args, **kwargs)
            else:
                print(f"Skipping {func.__name__} on {current_os}")
        return wrapper
    return decorator

class WindowProcessHandler():
    
    __slot__ = ['process_name','hwnd','window_process']
    
    def __init__(self,  process_name):
        # DPI 인식 활성화
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        
        # self.frame = frame
        self.process_name = process_name
        self.hwnd = self.get_handler_of_window_process(process_name)
        self.window_process = None

    @os_specific_task("Windows")
    def get_handler_of_window_process(self,process_name):
        def callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                if process_name.lower() in win32gui.GetWindowText(hwnd).lower():
                    hwnds.append(hwnd)
            return True
        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        return hwnds[0] if hwnds else None
    
    @os_specific_task("Windows")
    def connect_application_by_handler(self):
        app = Application().connect(handle=self.hwnd)
        self.window_process = app.top_window()
        self.window_process.set_focus()
    
    @os_specific_task("Windows")
    def mouseclick(self, button: str, coords: tuple):
        def task():
            self.window_process.set_focus()
            mouse.click(button=button, coords=coords)
        threading.Thread(target=task).start()
    
    @os_specific_task("Windows")
    def sendkey(self, key: str):
        def task():
            self.window_process.set_focus()
            keyboard.send_keys(key)
        threading.Thread(target=task).start()
    
    @os_specific_task("Windows")
    def captuer_screen_on_application(self):
        
        if not self.hwnd:
            print(f"프로세스 '{self.process_name}'을 찾을 수 없습니다.")
            return None
        
        # Get the window rectangle
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        # Use the default flag to capture the window
        result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 0x00000001)
        # result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 0)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        # Ensure the buffer size matches the expected size
        expected_size = bmpinfo['bmWidth'] * bmpinfo['bmHeight'] * 4  # BGRX format
        if len(bmpstr) != expected_size:
            # Adjust the buffer size to match the expected size
            bmpstr = bmpstr.ljust(expected_size, b'\x00')

        # Convert buffer to numpy array and reshape
        bmp_array = np.frombuffer(bmpstr, dtype=np.uint8)
        bmp_array = bmp_array.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))

        screenshot = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmp_array, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        if result == 1:
            # folder_dir = os.getcwd()+"/screen"
            # create_directory_if_not_exists(folder_dir)
            # saved_file = folder_dir+"/capture.jpg"
            # screenshot.save(saved_file)
            # print(f"Screenshot saved as {saved_file}")
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            print("Failed to capture window")
            return None
        
@os_specific_task("Windows")
def main():
    process_name = "Geometry Dash"  # 예: "Notepad", "Chrome" 등
    # frame = cv2.imread('result.png',0)
    handler = WindowProcessHandler(process_name)
    # handler.captuer_screen_on_application()
    handler.connect_application_by_handler()
    handler.mouseclick('left',(250,250))
    handler.sendkey('{SPACE}') 
    # {ENTER},{TAB},{ESC},{SPACE},{BACKSPACE} ex) 엔터 두번, {ENTER 2}
    # {DELETE},{UP},{DOWN},{LEFT},{RIGHT}
    # + (Shift), ^ (Ctrl), % (Alt)와 함께 사용 가능 (예: ^a는 Ctrl+A, +{INS}는 Shift+Insert)
    # 예를 들어, send_keys('^a^c')는 Ctrl+A로 모두 선택하고 Ctrl+C로 복사하는 동작을 수행
if __name__ == "__main__":
    main()
