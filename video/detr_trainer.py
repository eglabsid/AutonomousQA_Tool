import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import json
import os

from detr.models.detr import DETR
from detr.models.backbone import Backbone
from detr.models.position_encoding import PositionEmbeddingLearned
from detr.models.transformer import Transformer
        
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file
# COCO 형식의 데이터셋 클래스 정의
class CustomCocoDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        with open(json_file) as f:
            self.coco = json.load(f)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.coco['images'])

    def __getitem__(self, idx):
        img_info = self.coco['images'][idx]
        img_path = f"{self.root_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        annotation = [ann for ann in self.coco['annotations'] if ann['image_id'] == img_info['id']]
        target = {}
        target['boxes'] = torch.tensor([ann['bbox'] for ann in annotation], dtype=torch.float32)
        target['labels'] = torch.tensor([ann['category_id'] for ann in annotation], dtype=torch.int64)

        return image, target


class DETRTrainer:
    def __init__(self, json_file, root_dir, batch_size=2, num_epochs=10, lr=1e-4, device=None):
        self.transform = T.Compose([T.ToTensor(), T.Resize((480, 640))])
        # self.transform = T.Compose([T.ToTensor(), T.Resize((800, 800))])
        self.dataset = CustomCocoDataset(json_file=json_file, root_dir=root_dir, transform=self.transform)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs

        self.backbone = Backbone('resnet50', pretrained=True, train_backbone=True, return_interm_layers=False)
        self.position_embedding = PositionEmbeddingLearned(128)
        self.transformer = Transformer(d_model=256, return_intermediate_dec=True)
        self.model = DETR(self.backbone, self.position_embedding, self.transformer, num_classes=91)
        
        # safetensor 파일 로드
        weights_path = 'detr/models/models.safetensors'
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter('runs/detr_experiment')
        self.checkpoint_path = 'detr_checkpoint.pth'

    def train(self):
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for images, targets in self.data_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # 모델에 데이터 입력
                outputs = self.model(images)

                # 손실 계산
                loss = sum(self.criterion(output, target) for output, target in zip(outputs, targets))
                epoch_loss += loss.item()

                # 역전파 및 옵티마이저 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_loss = epoch_loss / len(self.data_loader)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss}")

            # 모델 저장
            torch.save(self.model.state_dict(), self.checkpoint_path)

        self.writer.close()
        print("모델 학습이 완료되었습니다! 모델이 저장되었습니다.")


def main():
    json_file = 'dataset.json'
    output_dir = 'video/output_images'
    json_path = os.path.join(output_dir, json_file)
    # 모델 학습 실행
    trainer = DETRTrainer(json_file=json_path, root_dir=output_dir)
    trainer.train()    

if __name__ == '__main__':
    main()

