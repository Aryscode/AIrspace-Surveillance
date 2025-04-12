# dataset.py
import os
import torch
import pandas as pd
import cv2
from torch.utils.data import Dataset

CLASS_NAMES = ['bird', 'drone', 'aeroplane', 'chopper']
CLASS_TO_IDX = {cls: i+1 for i, cls in enumerate(CLASS_NAMES)}

class MatlabAnnotationDataset(Dataset):
    def __init__(self, image_root, annotations_folder, index_csv, transforms=None):
        self.image_root = image_root
        self.transforms = transforms
        self.samples = []

        self.index = pd.read_csv(index_csv)

        for _, row in self.index.iterrows():
            prefix = row["video"]
            frame_idx = row["frame"]
            label_file = os.path.join(annotations_folder, f"{prefix}_LABELS.csv")
            if os.path.exists(label_file):
                df = pd.read_csv(label_file, header=None)
                df.columns = ["timestamp", "aeroplane", "bird", "drone", "chopper"]
                if frame_idx < len(df):
                    self.samples.append((prefix, frame_idx, df.iloc[frame_idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_name, frame_idx, row = self.samples[index]
        image_name = f"{video_name}_frame{frame_idx:04}.jpg"
        image_path = os.path.join(self.image_root, video_name, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = [], []
        for cls in CLASS_NAMES:
            value = str(row[cls]).strip()
            if value and value.lower() != 'nan':
                for item in value.split(';'):
                    try:
                        x, y, w, h = map(float, item.strip('[]').split(','))
                        boxes.append([x, y, x + w, y + h])
                        labels.append(CLASS_TO_IDX[cls])
                    except (ValueError, IndexError):
                        continue

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target
