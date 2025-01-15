import torch
import torchvision
# from torchvision.models.detection import detr_resnet50
from torchvision.transforms import functional as F
from transformers import DetrFeatureExtractor, DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

from PIL import Image

import json
import re

def sort_files_by_number(file_list):
    def extract_number(file_name):
        match = re.search(r'(\d+)', file_name)
        return int(match.group(1)) if match else float('inf')
    
    sorted_files = sorted(file_list, key=extract_number)
    return sorted_files

# 역정규화 함수 정의
def denormalize(tensor,mean,std):
    # 정규화에 사용된 값
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

# coco_data = {
#     "images": [
#         {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
#         {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600}
#     ],
#     "annotations": [
#         {"id": 1, "image_id": 1, "category_id": 3, "bbox": [100, 200, 150, 250], "area": 37500, "iscrowd": 0},
#         {"id": 2, "image_id": 1, "category_id": 1, "bbox": [300, 400, 100, 100], "area": 10000, "iscrowd": 0},
#         {"id": 3, "image_id": 2, "category_id": 2, "bbox": [50, 75, 200, 300], "area": 60000, "iscrowd": 0}
#     ],
#     "categories": [
#         {"id": 1, "name": "cat", "supercategory": "animal"},
#         {"id": 2, "name": "dog", "supercategory": "animal"},
#         {"id": 3, "name": "car", "supercategory": "vehicle"}
#     ]
# }

# from torch.utils.data import DataLoader

# 1. 데이터셋 로드
class CocoDetectionDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        """
        COCO 데이터셋을 PyTorch Dataset으로 변환.

        Args:
            root (str): 이미지 파일들이 저장된 경로.
            annotation (str): COCO 포맷의 annotation 파일 경로.
            transforms (callable, optional): 이미지를 전처리하기 위한 transform 함수.
        """
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        # 이미지 ID 가져오기
        img_id = self.ids[index]
        
        # 이미지 경로 및 읽기
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Annotation 가져오기
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # 바운딩 박스와 레이블 준비
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # COCO는 [x, y, width, height] 형식
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 타겟 데이터 생성
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        # Transform 적용
        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.ids)

def prepare_dataloader(root, annotation, batch_size=4):
    """
    COCO 데이터셋을 로드하고 DataLoader를 준비.

    Args:
        root (str): COCO 이미지 파일들이 저장된 경로.
        annotation (str): COCO annotation 파일 경로.
        batch_size (int): 배치 크기.

    Returns:
        DataLoader: PyTorch DataLoader 객체.
    """
    transforms = get_transforms()
    dataset = CocoDetectionDataset(root, annotation, transforms)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))  # Detection Transformer에서 필요
    )
    return dataloader

# 2. 모델 준비
class DetrWithFeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()
        # Load pre-trained model and feature extractor
        self.feature_extractor = DetrImageProcessor.from_pretrained(pretrained_model_name)
        self.model = DetrForObjectDetection.from_pretrained(pretrained_model_name)

        # Update classification head for fine-tuning
        self.model.class_labels_classifier = torch.nn.Linear(
            self.model.class_labels_classifier.in_features, num_classes
        )

    def forward(self, images, targets=None):
        """
        Forward pass for the combined model.

        Args:
            images (List[Image] or List[np.ndarray]): Input images (PIL or NumPy format).
            targets (List[Dict], optional): Target annotations for training. Each dict should contain:
                - 'boxes': Tensor of shape (num_objects, 4), with bounding boxes in xyxy format.
                - 'labels': Tensor of shape (num_objects,), with class labels.

        Returns:
            Model outputs from DetrForObjectDetection.
        """
        # Preprocess images
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # 1. 역정규화
        images = denormalize(images[0], mean, std)
        
        # 2. [0, 1]로 클램핑
        clamped_tensor = torch.clamp(images, 0, 1)
        
        # 3. PIL 이미지로 변환
        pil_image = F.to_pil_image(clamped_tensor)
        # pil_image.show()
        inputs = self.feature_extractor(images=images,annotations=targets, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        # If targets are provided, preprocess them
        if targets is not None:
            processed_targets = []
            for target in targets:
                processed_target = {
                    "boxes": target["boxes"].float(),
                    "labels": target["labels"]
                }
                processed_targets.append(processed_target)
        else:
            processed_targets = None

        # Forward pass through the DETR model
        outputs = self.model(pixel_values=pixel_values, labels=processed_targets)
        return outputs
    
# 3. Transform 정의
def get_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 4. 학습 루프 정의
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    total_loss = 0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    return total_loss / len(data_loader)

# 5. 평가 루프 정의
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss / len(data_loader)

# 6. Fine-tuning 실행
def main():
    pretrained_model_name = "facebook/detr-resnet-50"
    # Paths
    data_dir = f"./"  # Replace with your dataset directory
    num_classes = 2  # COCO has 80 classes + background
    annotation_dir = "./video/dataset/annotations.json"
    
    # Device
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")

    # Dataset and Dataloader

    # Dataset and Dataloader
    data_loader = prepare_dataloader(data_dir,annotation_dir)

    # Model, optimizer, and scheduler
    # model = get_model(num_classes).to(device)
    model = DetrWithFeatureExtractor(pretrained_model_name,num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        print(f"Epoch {epoch}, Training Loss: {train_loss:.4f}")

        if epoch % 2 == 0:  # Save the model every 2 epochs
            torch.save(model.state_dict(), f"detr_finetuned_epoch_{epoch}.pth")
            print(f"Model saved: detr_finetuned_epoch_{epoch}.pth")

    print("Training complete.")

if __name__ == "__main__":
    main()