from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
import time

# 마우스와 키보드 컨트롤러 초기화
mouse = MouseController()
keyboard = KeyboardController()

# 마우스 움직이기
def move_mouse(x, y):
    mouse.position = (x, y)
    print(f"Mouse moved to ({x}, {y})")

# 마우스 클릭하기
def click_mouse(button=Button.left):
    mouse.click(button)
    print(f"Mouse clicked with {button}")

# 키보드 키 누르기
def press_key(key):
    keyboard.press(key)
    print(f"Key {key} pressed")

# 키보드 키 놓기
def release_key(key):
    keyboard.release(key)
    print(f"Key {key} released")

# 예제 실행
if __name__ == "__main__":
    # 마우스 움직이기
    move_mouse(100, 200)
    time.sleep(1)
    
    # 마우스 클릭하기
    click_mouse()
    time.sleep(1)
    
    # 키보드 키 누르기
    press_key('a')
    time.sleep(1)
    
    # 키보드 키 놓기
    release_key('a')
