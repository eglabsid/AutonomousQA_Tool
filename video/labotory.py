import cv2

# 비디오 파일 경로
video_path = 'Official Geometry Dash Trailer.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

paused = False

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse Position: ({x}, {y})")

# 윈도우 생성 및 마우스 콜백 설정
cv2.namedWindow('Video Playback')
cv2.setMouseCallback('Video Playback', mouse_callback)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        cv2.imshow('Video Playback', frame)
    
    key = cv2.waitKey(25) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused

# 모든 윈도우 닫기
cap.release()
cv2.destroyAllWindows()
