# test_dataset.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import MatlabAnnotationDataset
from torchvision.transforms import ToTensor
import cv2
import torch

def show_sample(image, target):
    # Convert tensor to numpy
    img = image.permute(1, 2, 0).numpy()
    _ , ax = plt.subplots(1)
    ax.imshow(img)

    for box, label in zip(target["boxes"], target["labels"]):
        x, y, x2, y2 = box
        width = x2 - x
        height = y2 - y
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, str(label.item()), color='yellow', fontsize=10, backgroundcolor='black')

    plt.show()

def main():
    dataset = MatlabAnnotationDataset(
        image_root="images",
        annotations_folder="annotations",
        index_csv="train_index.csv",
        transforms=ToTensor()
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Test a few random samples
    for i in [0, 10, 50]:
        if i >= len(dataset):
            continue
        image, target = dataset[i]
        print(f"Sample {i} - Boxes: {target['boxes'].shape}, Labels: {target['labels']}")
        show_sample(image, target)

if __name__ == "__main__":
    main()
