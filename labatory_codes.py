import cv2
import asyncio
import concurrent.futures

from utils.process_handler import WindowProcessHandler
from utils.template_matcher import TemplateMatcher

import cv2
import time

import threading
import queue

process = "GeometryDash.exe"
hwdl = WindowProcessHandler()
hwdl.connect_application_by_process_name(process)



window_name = 'Processed Frame' 
# 창을 크기 조절 가능하게 만듭니다.
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
# 창 크기 설정 (너비, 높이)
cv2.resizeWindow(window_name, 800, 800)
#
template = cv2.imread('screen/UI/select_play.jpg', cv2.IMREAD_GRAYSCALE)
labotory = TemplateMatcher(template,scale_range=(0.5,1.0,0.1))

# 원하는 FPS를 설정합니다.
desired_fps = 60
frame_time = 1 / desired_fps

# 프레임을 저장할 큐를 초기화합니다.
frame_queue = queue.Queue()


def process_frame(frame):
    # 프레임을 그레이스케일로 변환
    return labotory.expierience_lab(frame)


async def capture_frames():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while True:
            frame = await loop.run_in_executor(pool, hwdl.captuer_screen_on_application)
            # frame_queue.put(('original', frame))
            
            processed_frame = await loop.run_in_executor(pool, process_frame, frame)
            # frame_queue.put(('processed', processed_frame))
            
            cv2.imshow('Processed Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(frame_time)
        # frame_queue.put((None, None))

async def display_frames():
    while True:
        if not frame_queue.empty():
            frame_type, frame = frame_queue.get()
            # if frame_type is None:
            #     break
            if frame_type == 'original':
                cv2.imshow('Real-time Capture', frame)
            elif frame_type == 'processed':
                cv2.imshow('Processed Frame', frame)
            # frame_queue.task_done()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #객체와 창을 해제합니다.
    cv2.destroyAllWindows()
        
async def main():
    # capture_task = asyncio.create_task(capture_frames())
    # display_task = asyncio.create_task(display_frames())
    # await asyncio.gather(capture_task, display_task)
    
    await capture_frames()
     # 프레임 캡처와 디스플레이를 비동기로 실행합니다.
    # await asyncio.gather(capture_frames(), display_frames())
    

# 비동기 루프를 실행합니다.
asyncio.run(main(),debug=True)
