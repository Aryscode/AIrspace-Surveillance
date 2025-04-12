# evaluate.py
import torch
from torch.utils.data import DataLoader
from dataset import MatlabAnnotationDataset
from train import get_model, collate_fn
from torchvision.transforms import ToTensor
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import os
from PIL import Image
import matplotlib.pyplot as plt

CLASSES = ["__background__", "bird", "drone", "airplane", "chopper"]

def show_prediction(image, prediction):
    img = image.mul(255).byte().cpu()
    pred_boxes = prediction['boxes'][prediction['scores'] > 0.5]
    pred_labels = prediction['labels'][prediction['scores'] > 0.5]

    draw = draw_bounding_boxes(img, pred_boxes, [CLASSES[i] for i in pred_labels], colors="red", width=2)
    plt.imshow(F.to_pil_image(draw))
    plt.axis("off")
    plt.show()

def evaluate_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(num_classes=5)
    model.load_state_dict(torch.load("fasterrcnn.pth"))
    model.to(device)
    model.eval()

    dataset_test = MatlabAnnotationDataset("images", "annotations", "test_index.csv", transforms=ToTensor())
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Visualize a few predictions
    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
            image = images[0].to(device)
            prediction = model([image])[0]
            show_prediction(image.cpu(), prediction)

            if idx >= 4:  # visualize 5 samples
                break

if __name__ == "__main__":
    evaluate_model()
