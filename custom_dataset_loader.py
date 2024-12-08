import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import os
import model_compression_toolkit as mct
from torch.utils.data import DataLoader
from ultralytics import YOLO

class CustomImageDataset(Dataset):
    def __init__(self, annotations_folder, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_folder
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_labels))-1

    def __getitem__(self, idx):
        img_label_files = os.listdir(self.img_labels)
        img_label_files.remove('.DS_Store')
        img_path = os.path.join(self.img_dir, img_label_files[idx].split(".")[0]+".png")
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = np.loadtxt(os.path.join(self.img_labels, img_label_files[idx]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
dataset = CustomImageDataset(annotations_folder = "/opt/homebrew/datasets/cuad_video10062024/labels/test", img_dir="/opt/homebrew/datasets/cuad_video10062024/images/test")
batch_size = 16
n_iter = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def representative_dataset_gen():
    dataloader_iter = iter(dataloader)
    for _ in range(n_iter):
        yield [next(dataloader_iter)[0]]

target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')

model = YOLO("yolov8n.pt")  
results = model.train(data="video1_obb_data.yaml", epochs=1, imgsz=640)
quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=results,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=target_platform_cap
)

mct.exporter.pytorch_export_model(quantized_model, save_model_path='qmodel.onnx', repr_dataset=representative_dataset_gen)