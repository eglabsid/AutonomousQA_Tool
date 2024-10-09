import cv2
import json
import os

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

folder = f"capture"
load_images = load_images_from_folder(folder)


results = []

# 결과 저장 디렉토리 생성
output_dir = f'output_images'
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
    for angle in range(0, 360, 45):
        rotated_image = rotate_image(image, angle)
        for scale in [0.5, 1.0, 1.5]:
            resized_image = resize_image(rotated_image, scale)
            result_filename = f'{name}_{angle}_{scale}.jpg'
            result_path = os.path.join(output_dir, result_filename)
            cv2.imwrite(result_path, resized_image)
            results.append({
                'name': name,
                'box': box,
                'coord': (coord_x,coord_y),
                'file_path': result_path
            })

# 결과를 JSON 파일로 저장
with open(f'output_images/results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Processing complete. Results saved to results.json and output_images directory.")
