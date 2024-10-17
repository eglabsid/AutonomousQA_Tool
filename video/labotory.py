import cv2,os
import glob

import json

import asyncio
import concurrent.futures

# from process_handler import WindowProcessHandler
from template_matcher import TemplateMatcher

from collections import OrderedDict # 순서대로 Dict

import cv2
import numpy as np



def make_template(root_folder, is_rotate = False):
    # 이미지 파일 확장자 목록
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

    # 모든 이미지 파일 경로를 저장할 리스트
    all_images = []
    
    # 하위 폴더를 포함한 모든 이미지 파일 검색
    for extension in image_extensions:
        # all_images.extend(glob.glob(os.path.join(root_folder, '**', extension), recursive=False))
        all_images.extend(glob.glob(os.path.join(root_folder, extension), recursive=False))
        
        # 사용 예제
    templates = []
    
    for name in all_images:
        
        # name = name.split('\\')[-1]
        # name = file.split('.')[0]
        # print(name)
        template = {}
        # template[name] = load_and_resize_image(file)
        # template[name] =  cv2.imread(file, cv2.IMREAD_COLOR)
        if not is_rotate:
            template[name] =  cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        else:
            template[name] =  cv2.imread(name, cv2.IMREAD_COLOR)
        templates.append(template)
    return templates
    
def get_match_info(matches):
    match_info =[]
    # gui 관련 파라미터 설정
    for loc, scale, score, template_tuple in matches:    
        path = template_tuple[0].replace('\\','/')
        path = path.split('/')[-1]
        name = path.split('.')[0]
        name = name.split('_')[0]
        # h,w,_= template_tuple[1].shape # 중심좌표
        if loc[0] == 0 and loc[1] == 0:
            continue
        
        h,w= template_tuple[1].shape # 중심좌표
        top_left = loc
        bottom_right = (top_left[0] + int(h * scale), top_left[1] + int(w * scale))
            
        gui_dic = {} # gui 유사도, 좌표, ui name 탐색
        mc_loc = [loc[0] + int(w * scale * 0.5), loc[1] + int(h * scale * 0.5)]        

        bbox = top_left + bottom_right
        
        # 파일명, 이미지데이터, 중심좌표, bbox
        gui_dic[name] = ( template_tuple[0] ,template_tuple[1], mc_loc, bbox )
        match_info.append(gui_dic)
    return match_info


def resize_image(image, scale):
    h,w = image.shape[:2]
    width = int(w * scale)
    height = int(h * scale)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate the rotation matrix for the given angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate the new bounding dimensions of the image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to take into account the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated

def make_train_datas():
    key_lst = ['images','annotations','categories']
    train_datas = {}
    for key in key_lst:
        train_datas[key] = []

    return train_datas

def make_image(images,filename,h,w):
    id = -1
    is_exist = False
    for image in images:
        if image['file_name'] == filename:
            return image, not is_exist
        id = int(image['id'])
    
    if id < 0:
        id = 1
    else:
        id += 1
    
    return add_image(id,filename,h,w), is_exist

def make_annotation(annotations,image_id,category_id,bbox,area,iscrowd=0):
    
    id = -1
    is_exist = False
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            return annotation, not is_exist
        id = int(annotation['id'])
    # area = abs(bbox[0]-bbox[1])*abs(bbox[2]-bbox[3])
    
    if id < 0:
        id = 1
    else:
        id += 1
    return add_annotation(id,image_id,category_id,bbox,area,iscrowd), is_exist

def make_category(categories,name):
    id = -1
    is_exist = False
    for category in categories:
        if category['name'] == name:
            return category, not is_exist
        id = int(category['id'])
    
    if id < 0:
        id = 1
    else:
        id += 1
        
    return add_category(id, name), is_exist
        

def add_image(id,filename,height,width):
    image = OrderedDict()
    image['id'] = id
    image['file_name'] = filename
    image['height'] = height
    image['width'] = width
    return image

def add_annotation(id,image_id,category_id,bbox,area,iscrowd):
    annotation =OrderedDict()
    annotation['id'] = id
    annotation['image_id'] = image_id
    annotation['category_id'] = category_id
    annotation['bbox'] = bbox
    annotation['area'] = area
    annotation['iscrowd'] = iscrowd # 1이면 군집, 0이면 단일
    return annotation

def add_category(id,name):
    category = OrderedDict()
    category['id']=id
    category['name']=name
    # category['supercategory']=supercategory
    return category
    

# 결과 저장 디렉토리 생성
output_dir = f'video/output_images'
os.makedirs(output_dir, exist_ok=True)
    
# Delete the JSON file if it already exists
tmp_file = 'dataset.json'
json_file_path = os.path.join(output_dir, tmp_file)
if os.path.exists(json_file_path):
    os.remove(json_file_path)

def argments_rotate(folder):
    is_rotate = True
    templates = make_template(folder,is_rotate)
    for template in templates:
        for k,v in template.items():
            name = k.split('\\')[-1]
            name = name.split('_')[0]
            for angle in range(0, 360, 30):
                rotated_image = rotate_image(v, angle)
                result_filename = f'{name}_{angle}.jpg'
                result_path = os.path.join(output_dir, result_filename)
                cv2.imwrite(result_path, rotated_image)
            

def make_argments(match_info,frame, frame_cnt):
    
    
    # Load existing results if the file exists
    # results = []
    results = make_train_datas()
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            results =  json.load(json_file)

    while len(match_info) > 0:
        infos = match_info.pop(0)
        # print(f"{infos}")
        for k,v in infos.items():
            # cv2.rectangle(frame, v[-1][:2], v[-1][2:], (0, 0, 255), 4)
            # cv2.putText(frame, f'{k}', (v[-1][0], v[-1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            # resized_image = resize_image(values[0], scale)
            result_filename = f'{k}_{frame_cnt}.jpg'
            path = os.path.join(output_dir, result_filename)
            cv2.imwrite(path, frame)

            # Images : id/파일명/h/w
            # Annotation : id/image_id/category_id/bbx/area/iscrowd
            # categories : id/name
            # 새로운 데이터를 추가할지 여부 확인 및 추가
            # h,w = v[1].shape
            # image, is_exist = make_image(results['images'],v[0],h,w)
            
            h,w,_ = frame.shape
            image, is_exist = make_image(results['images'],result_filename,h,w)
            
            if not is_exist:
                results['images'].append(image)
            category, is_exist = make_category(results['categories'],k)
            if not is_exist:
                results['categories'].append(category)
            area = w*h
            # is_crowed = 0
            annotation,is_exist = make_annotation(results['annotations'],image['id'],category['id'],v[-1],area)
            if not is_exist:
                results['annotations'].append(annotation)
        
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
        
    return results, frame_cnt

labotory = TemplateMatcher(template=None,scale_range=(0.5,1.0,0.1))

# 원하는 FPS를 설정합니다.
desired_fps = 60
frame_time = 1 / desired_fps

def process_frame(frame,templates):
    # 프레임을 그레이스케일로 변환
    return labotory.experience_video(frame,templates)

def process_match(matches):
    return get_match_info(matches)

def process_arg(matches_info,frame,frame_cnt):
    return make_argments(matches_info,frame,frame_cnt)

# 비디오 파일 경로
video_path = 'video/Official Geometry Dash Trailer.mp4'
        
# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

#templates = make_template(f"video/capture")
template_folder = f"video/capture/arg"
async def capture_frames():
    frame_cnt = 0    
    paused = False
    loop = asyncio.get_event_loop()
    # rotate 추가
    # argments_rotate(template_folder) 
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while True:
            if not paused:
                ret, frame = await loop.run_in_executor(pool, cap.read)
                if not ret:
                    print("End of video.")
                    break
            templates = make_template(template_folder)
            matches = await loop.run_in_executor(pool, process_frame, frame,templates)
            matches_info = await loop.run_in_executor(pool, process_match, matches)
            results, _ = await loop.run_in_executor(pool, process_arg, matches_info,frame,frame_cnt)
                
            cv2.imshow('Processed Frame', frame)
            frame_cnt += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('p'):
                paused = not paused
            
            await asyncio.sleep(frame_time)
        # frame_queue.put((None, None))

        
async def main():
    # capture_task = asyncio.create_task(capture_frames())
    # display_task = asyncio.create_task(display_frames())
    # await asyncio.gather(capture_task, display_task)
    
    await capture_frames()
     # 프레임 캡처와 디스플레이를 비동기로 실행합니다.
    # await asyncio.gather(capture_frames(), display_frames())
    

# 비동기 루프를 실행합니다.
asyncio.run(main(),debug=True)