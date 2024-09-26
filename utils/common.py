
import os

import numpy as np
import mss
import mss.tools

# import gc
from memory_profiler import profile

import pygetwindow as gw

# pyautogui
import pyautogui
import subprocess, psutil

import win32gui
import win32process

def create_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created successfully!")
    else:
        print(f"Directory '{dir_path}' already exists.")


def connect_application_by_process_name(process_name):
    # 실행 중인 프로세스 검색
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            message = check_process_load(proc)
            print(message)

            # 윈도우 활성화
            window = find_window_by_pid(proc.info['pid'])
            if window:
                window.activate()
                message += "\n Window activated successfully!"
            else:
                message += "\n Window not found."
            print(message)
            return message

    message = f"Process '{process_name}' not found."
    print(message)
    return message

# 프로세스 ID를 기반으로 윈도우 찾기
def find_window_by_pid(pid):
    # for window in gw.getWindowsWithTitle(''):
            # if window._hWnd == pid:
    #         return window
    # return None
    def callback(hwnd, pid):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
        if found_pid == pid:
            windows.append(hwnd)
        return True

    windows = []
    win32gui.EnumWindows(callback, pid)
    if windows:
        return gw.Win32Window(windows[0])
    return None

# 프로세스 로드 상태 체크 함수
def check_process_load(proc):
    message = f"Process {proc.info['name']} (PID: {proc.info['pid']}) is running."
    try:
        if proc.is_running():
            cpu_usage = proc.cpu_percent(interval=1)
            memory_info = proc.memory_info()
            memory_usage = memory_info.rss / (1024 * 1024)  # 메모리 사용량 (MB)
            message = f"CPU Usage: {cpu_usage}% \n"
            message += f"Memory Usage: {memory_usage} MB"
        else:
            message = "Process is not running."
    except psutil.NoSuchProcess:
        message = "Process no longer exists."
    return message
    
def mouseclick(button:str,coords:tuple):
    # 마우스 이동
    pyautogui.moveTo(coords[0],coords[1], duration=1)
    # 마우스 클릭
    pyautogui.click()

def sendkey(key:str):        
    pyautogui.press(key)

