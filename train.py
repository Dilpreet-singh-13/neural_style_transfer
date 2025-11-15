# train.py (replace original)
import os
import glob
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import vgg
import transformer
import utils

# -----------------------------
# USER CONFIG (edit as needed)
# -----------------------------
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "/kaggle/working/train2017"  # set to the folder containing COCO train2017 .jpg files
NUM_EPOCHS = 2
STYLE_IMAGE_PATH = "images/mosaic.jpg"
BATCH_SIZE = 6
CONTENT_WEIGHT = 12
STYLE_WEIGHT = 60
ADAM_LR = 1e-3
SAVE_MODEL_PATH = "models/"
SAVE_IMAGE_PATH = "images/out/"
SAVE_MODEL_EVERY = 2000  # iterations
SEED = 35
NUM_WORKERS = 4
PLOT_LOSS = 1
# -----------------------------

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)

# Repro
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -----------------------------
# Simple Dataset for folder of images (COCO style)
# -----------------------------
class ImageFolderCOCO(Dataset):
    def __init__(self, root_dir, image_size):
        # gather jpg/jpeg/png
        self.files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.files.extend(glob.glob(os.path.join(root_dir, ext)))
        self.files.sort()
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def _read_and_preprocess(self, path):
        # load with OpenCV (BGR)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Unable to read image {path}")
        h, w, _ = img.shape

        # Resize preserving aspect and center crop to image_size
        if min(h, w) != self.image_size:
            # scale so smaller side == image_size, then center-crop
            scale = float(self.image_size) / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        # center crop
        h, w, _ = img.shape
        top = (h - self.image_size) // 2
        left = (w - self.image_size) // 2
        img = img[top : top + self.image_size, left : left + self.image_size]

        # Convert to float32 and shape C,H,W and scale to 0..255 (already in 0..255)
        img = img.astype("float32")
        # to tensor
        tensor = (
            torch.from_numpy(img).permute(2, 0, 1).contiguous()
        )  # BGR order, 0..255
        return tensor

    def __getitem__(self, idx):
        img_path = self.files[idx]
        tensor = self._read_and_preprocess(img_path)
        return tensor, 0  # follow ImageFolder signature (data, label)


# -----------------------------
# Prepare data
# -----------------------------
train_dataset = ImageFolderCOCO(DATASET_PATH, TRAIN_IMAGE_SIZE)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print("Found {} images".format(len(train_dataset)))

# -----------------------------
# Models
# -----------------------------
TransformerNetwork = transformer.TransformerNetwork().to(device)
VGG = vgg.VGG16().to(device)
TransformerNetwork.train()

# -----------------------------
# Prepare style features (keep existing mean-subtraction, BGR order)
# -----------------------------
imagenet_neg_mean = (
    torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32)
    .view(1, 3, 1, 1)
    .to(device)
)
style_image = utils.load_image(STYLE_IMAGE_PATH)  # OpenCV BGR
style_tensor = utils.itot(style_image).to(device)  # returns 1xCxHxW in BGR 0..255
style_tensor = style_tensor.add(imagenet_neg_mean)  # subtract mean
B, C, H, W = style_tensor.shape
# expand style to batch size when computing gram
style_feats = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
style_gram = {k: utils.gram(v) for k, v in style_feats.items()}

# Optimizer
optimizer = optim.Adam(TransformerNetwork.parameters(), lr=ADAM_LR)

# Loss
MSELoss = nn.MSELoss().to(device)

# Training loop
batch_count = 1
start_time = time.time()
content_loss_history = []
style_loss_history = []
total_loss_history = []

for epoch in range(NUM_EPOCHS):
    print("====== Epoch {}/{} ======".format(epoch + 1, NUM_EPOCHS))
    for content_batch, _ in train_loader:
        curr_batch_size = content_batch.shape[0]
        optimizer.zero_grad()

        # content_batch is BGR 0..255 in shape [B, C, H, W]
        content_batch = content_batch.to(device)

        generated_batch = TransformerNetwork(content_batch)
        # add mean before feeding to VGG
        content_features = VGG(content_batch.add(imagenet_neg_mean))
        generated_features = VGG(generated_batch.add(imagenet_neg_mean))

        # Content loss (relu2_2)
        content_loss = CONTENT_WEIGHT * MSELoss(
            generated_features["relu2_2"], content_features["relu2_2"]
        )

        # Style loss
        style_loss = 0.0
        for key, value in generated_features.items():
            g = utils.gram(value)
            # style_gram[key] has shape [BATCH_SIZE, C, C] -- slice to curr_batch_size if needed
            target = style_gram[key][:curr_batch_size]
            style_loss += MSELoss(g, target)
        style_loss = STYLE_WEIGHT * style_loss

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()

        # logging and saves
        if batch_count % SAVE_MODEL_EVERY == 0:
            elapsed = time.time() - start_time
            print("--- Iter {} ---".format(batch_count))
            print("\tContent Loss: {:.4f}".format(content_loss.item()))
            print("\tStyle Loss:   {:.4f}".format(style_loss.item()))
            print("\tTotal Loss:   {:.4f}".format(total_loss.item()))
            print("\tElapsed: {:.1f} sec".format(elapsed))

            # save model
            ckpt = os.path.join(SAVE_MODEL_PATH, f"checkpoint_{batch_count}.pth")
            torch.save(TransformerNetwork.state_dict(), ckpt)
            print("Saved checkpoint:", ckpt)

            # save a sample output
            TransformerNetwork.eval()
            with torch.no_grad():
                sample = generated_batch[0].unsqueeze(0).cpu()
                sample_img = utils.ttoi(sample)  # returns HWC (still BGR)
                # convert BGR to RGB for saving / viewing (optional)
                sample_img = sample_img[..., ::-1]  # BGR->RGB
                sample_path = os.path.join(SAVE_IMAGE_PATH, f"sample_{batch_count}.png")
                utils.saveimg(sample_img, sample_path)
                print("Saved sample image:", sample_path)
            TransformerNetwork.train()

        # increment
        batch_count += 1

# final save
print("Training complete. Saving final model.")
final_path = os.path.join(SAVE_MODEL_PATH, "transformer_final.pth")
torch.save(TransformerNetwork.state_dict(), final_path)
print("Saved:", final_path)
