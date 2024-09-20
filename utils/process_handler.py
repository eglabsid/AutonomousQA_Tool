

import numpy as np
import cv2
import os
# import utils.common as common

import common

from PIL import Image

import ctypes
from ctypes import windll

import win32gui
import win32ui
import win32con

from pywinauto import Application
import pywinauto.mouse as mouse
import pywinauto.keyboard as keyboard

# DPI 인식 활성화
ctypes.windll.shcore.SetProcessDpiAwareness(2)

class WindowProcessHandler():
    
    __slot__ = ['process_name','hwnd','screenshot_file','window_process']
    
    def __init__(self, process_name):
        self.screenshot_file = "screenshot.jpg"
        self.process_name = process_name
        self.hwnd = self.get_handler_of_window_process(process_name)
        self.window_process = None

    def get_handler_of_window_process(self,process_name):
        def callback(hwnd, hwnds):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                if process_name.lower() in win32gui.GetWindowText(hwnd).lower():
                    hwnds.append(hwnd)
            return True
        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        return hwnds[0] if hwnds else None

    def connect_application_by_handler(self):
        app = Application().connect(handle=self.hwnd)
        self.window_process = app.top_window()
        # self.window_process.set_focus()
        
    def mouseclick(self,button:str,coords:tuple):
        self.window_process.set_focus()
        mouse.click(button=button,coords=coords)
    
    def sendkey(self,key:str):
        self.window_process.set_focus()
        keyboard.send_keys(key)
    
    def capture_window_screen(self):
        
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
            folder_dir = os.getcwd()+"/screen"
            common.create_directory_if_not_exists(folder_dir)
            saved_file = folder_dir+"/"+self.screenshot_file
            screenshot.save(saved_file)
            print(f"Screenshot saved as {saved_file}")
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            print("Failed to capture window")
            return None

def main():
    process_name = "Geometry Dash"  # 예: "Notepad", "Chrome" 등
    handler = WindowProcessHandler(process_name)
    handler.capture_window_screen()

if __name__ == "__main__":
    main()
