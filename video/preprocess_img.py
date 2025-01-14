import cv2
import json
import os
import numpy as np

def rotate_image_with_alpha(image, angle):
    """
    이미지를 회전시키고 배경을 알파 채널로 설정.

    Args:
        image (numpy.ndarray): 입력 이미지 (BGR 또는 BGRA 형식).
        angle (float): 회전 각도 (degree).

    Returns:
        numpy.ndarray: 회전된 이미지 (BGRA 형식).
    """
    # 알파 채널 추가 (이미지가 BGR 형식일 경우)
    if image.shape[2] == 3:  # BGR 이미지
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # 이미지 크기와 중심점
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 회전 변환 행렬 계산
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # 회전 후 새로운 이미지 크기 계산
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 변환 행렬에 새로운 중심점 반영
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # 회전된 이미지 생성
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return rotated_image

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def resize_image(image, scale):
    h,w = image.shape[:2]
    width = int(w * scale)
    height = int(h * scale)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        file = filename.split('/')[-1]
        name = file.split('.')[0]
        img = cv2.imread(img_path)
        if img is not None:
            images.append((name,img))
    return images

folder = f"./video/capture"
load_images = load_images_from_folder(folder)


results = []

# 결과 저장 디렉토리 생성
output_dir = f'./video/capture/arg'
os.makedirs(output_dir, exist_ok=True)

for name,image in load_images:
    if image is None:
        print("Error: Could not load image.")
        exit()
    h,w,_ = image.shape
    name = name.split('_')
    x = int(name[1])
    y = int(name[2])
    coord_x = int((x+w)*0.5)
    coord_y = int((y+h)*0.5)
    box  = [y,x,y+h,x+w]
    name = name[0]
    # 45도씩 회전 및 크기 조절
    for angle in range(0, 360, 15):
        rotated_image = rotate_image_with_alpha(image, angle)
        for scale in [0.5, 1.0, 1.5]:
            resized_image = resize_image(rotated_image, scale)
            result_filename = f'{name}_{angle}_{scale}.png'
            result_path = os.path.join(output_dir, result_filename)
            cv2.imwrite(result_path, resized_image)
            results.append({
                'name': name,
                'box': box,
                'coord': (coord_x,coord_y),
                'file_path': result_path
            })

# 결과를 JSON 파일로 저장
with open(f'{output_dir}/results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Processing complete. Results saved to results.json and output_images directory.")
