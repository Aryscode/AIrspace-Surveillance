# train.py
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from dataset import MatlabAnnotationDataset
import torchvision
import os

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_model(epochs=2, batch_size=4):
    dataset = MatlabAnnotationDataset("images", "annotations", "train_index.csv", transforms=ToTensor())

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    model = get_model(num_classes=5)  # 4 classes + background
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), "fasterrcnn.pth")
    print("✅ Model trained and saved as fasterrcnn.pth")

if __name__ == "__main__":
    train_model(epochs=2, batch_size=4)
