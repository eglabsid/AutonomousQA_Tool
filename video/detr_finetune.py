import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import matplotlib.pyplot as plt
from PIL import Image
import requests, glob
import os,json
from torch.utils.tensorboard import SummaryWriter


import re

def sort_files_by_number(file_list):
    def extract_number(file_name):
        match = re.search(r'(\d+)', file_name)
        return int(match.group(1)) if match else float('inf')
    
    sorted_files = sorted(file_list, key=extract_number)
    return sorted_files


root_folder = f"video/output_images"
# 이미지 파일 확장자 목록
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

# 모든 이미지 파일 경로를 저장할 리스트
all_images = []

# 하위 폴더를 포함한 모든 이미지 파일 검색
for extension in image_extensions:
    # all_images.extend(glob.glob(os.path.join(root_folder, '**', extension), recursive=False))
    all_images.extend(glob.glob(os.path.join(root_folder, extension), recursive=False))

# print(all_images)
all_images = sort_files_by_number(all_images)
# print(all_images)

results = []
json_file_path = os.path.join(root_folder,f"results.json")
if os.path.exists(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        results =  json.load(json_file)

annotations = []    

# add ={}
# add['boxes'] = []
# add['labels'] = [] #annotation['labels'][:-1]+'er'
for annotation in results:
    add ={}
    add['boxes']=annotation['boxs']
    add['labels']=0
    annotations.append(add)

# Transformations
transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor()
])


# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, annotations, transform=None):
        self.images = images
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        target = self.annotations[idx]

        if self.transform:
            image = self.transform(image)

        # Convert annotations to tensor
        target = {
            "boxes": torch.tensor(target["boxes"], dtype=torch.float32),
            "labels": torch.tensor(target["labels"], dtype=torch.int64)
        }
        return image, target

# Parameters
batch_size = 2
num_epochs = 10
learning_rate = 1e-5

# Initialize dataset and dataloader
dataset = CustomDataset(all_images, annotations, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained model and feature extractor
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')

# TensorBoard writer
writer = SummaryWriter('runs/detr_finetuning')

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(dataloader):
        # Prepare inputs
        encodings = feature_extractor(images=list(images), annotations=targets, return_tensors="pt")
        # encodings = feature_extractor(images=images, annotations=targets, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encodings.items()}
        
        
        # Forward pass
        outputs = model(**inputs)
        
        # Compute loss
        loss = outputs.loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Log loss to TensorBoard
        if i % 10 == 9:  # Log every 10 batches
            writer.add_scalar('training loss', running_loss / 10, epoch * len(dataloader) + i)
            running_loss = 0.0
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Save the model
    model_save_path = f'model_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

print("Fine-tuning completed!")
writer.close()

