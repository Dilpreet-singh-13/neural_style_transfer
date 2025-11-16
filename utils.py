import torch
import numpy as np
from PIL import Image

from .config import OUTPUT_DIR

def gram_matrix(tensor):
    """Compute Gram matrix"""
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H * W)
    x_t = x.transpose(1, 2)
    return torch.bmm(x, x_t) / (C * H * W)


def save_checkpoint(model, batch_count, output_dir=OUTPUT_DIR):
    """Save model checkpoint"""
    checkpoint_path = f"{output_dir}models/checkpoint_{batch_count}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint at {checkpoint_path}")


def save_sample_image(tensor, batch_count, output_dir=OUTPUT_DIR):
    """Save sample stylized image"""
    img = tensor[0].clone().detach()
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img)
    save_path = f"{output_dir}images/sample_{batch_count}.png"
    img_pil.save(save_path)
    print(f"Saved sample image at {save_path}")