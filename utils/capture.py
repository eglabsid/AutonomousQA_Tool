

import ctypes,os
from ctypes import windll

from PIL import Image
import numpy as np
import cv2

import win32gui
import win32ui
import win32con

# DPI 인식 활성화
ctypes.windll.shcore.SetProcessDpiAwareness(2)

def get_process_window(process_name):
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            if process_name.lower() in win32gui.GetWindowText(hwnd).lower():
                hwnds.append(hwnd)
        return True
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds[0] if hwnds else None

def capture_process_window(hwnd):
    # Get the window rectangle
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    # Use the default flag to capture the window
    # result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0x00000002)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

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

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmp_array, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        saved_screenshot = os.getcwd()+"/screenshot.png"
        im.save(saved_screenshot)
        print(f"Screenshot saved as {saved_screenshot}")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    else:
        print("Failed to capture window")
        return None

def main():
    process_name = "Geometry Dash"  # 예: "Notepad", "Chrome" 등
    hwnd = get_process_window(process_name)
    if hwnd:
        capture_process_window(hwnd)
    else:
        print(f"프로세스 '{process_name}'을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
