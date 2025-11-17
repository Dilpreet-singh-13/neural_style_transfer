import torch
import os
from torchvision import transforms

# global config
COCO_TRAIN_IMAGES = "/data/coco-2017-dataset/coco2017/train2017"  # path pointing to the train data folder
STYLE_IMAGE_PATH = "images/style_image.jpg"
OUTPUT_DIR = "./"

NUM_IMAGES = 20000  # change to 40000 for better results
BATCH_SIZE = 6  # 4-16 batch size depending on GPU
IMAGE_SIZE = 256
EPOCHS = 2
LEARNING_RATE = 1e-3
CONTENT_WEIGHT = 12
STYLE_WEIGHT = 60

# model will be saved every SAVE_INTERVAL batches
SAVE_INTERVAL = 2000
LOG_INTERVAL = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),  # Scale to [0, 255]
    ]
)
