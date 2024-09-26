
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

def create_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created successfully!")
    else:
        print(f"Directory '{dir_path}' already exists.")


def connect_application_by_process_name(process_name):
    # 메모장 실행
    process = subprocess.Popen([process_name])
    # psutil을 사용하여 프로세스 정보 가져오기
    proc = psutil.Process(process.pid)

    while not proc.is_running():
        check_process_load(proc)

    # 윈도우 활성화
    window = find_window_by_pid(proc.pid)
    if window:
        window.activate()
        print("Window activated successfully!")
    else:
        print("Window not found.")        

    
def mouseclick(button:str,coords:tuple):
    # 마우스 이동
    pyautogui.moveTo(coords[0],coords[1], duration=1)
    # 마우스 클릭
    pyautogui.click()

def sendkey(key:str):        
    pyautogui.press(key)

# 프로세스 ID를 기반으로 윈도우 찾기
def find_window_by_pid(pid):
    for window in gw.getWindowsWithTitle(''):
        if window._hWnd == pid:
            return window
    return None

# 프로세스 로드 상태 체크 함수
def check_process_load(proc):
    try:
        if proc.is_running():
            cpu_usage = proc.cpu_percent(interval=1)
            memory_info = proc.memory_info()
            memory_usage = memory_info.rss / (1024 * 1024)  # 메모리 사용량 (MB)
            print(f"CPU Usage: {cpu_usage}%")
            print(f"Memory Usage: {memory_usage} MB")
        else:
            print("Process is not running.")
    except psutil.NoSuchProcess:
        print("Process no longer exists.")