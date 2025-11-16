import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from .transformers import TransformerNetwork
from .config import DEVICE, OUTPUT_DIR

# ------------ path congif ------------
INFERENCE_IMG_PATH = "images/city_photo.jpg"  # change image name
MODEL_PATH = "models/transformer_weight_final.pth"
OUTPUT_IMG_PATH = f"{OUTPUT_DIR}/images/styled_img.jpg"


def stylize_image(model_path, input_image_path, output_path):
    """Apply style transfer to a single image"""
    transform_net = TransformerNetwork().to(DEVICE)
    transform_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    transform_net.eval()

    inference_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))]
    )

    img = Image.open(input_image_path).convert("RGB")
    original_size = img.size
    img_tensor = inference_transform(img).unsqueeze(0)

    img_tensor = img_tensor[:, [2, 1, 0]].to(DEVICE)

    with torch.no_grad():
        stylized = transform_net(img_tensor)

    stylized = stylized[:, [2, 1, 0]].squeeze(0).cpu().numpy()
    stylized = stylized.transpose(1, 2, 0)
    stylized = np.clip(stylized, 0, 255).astype(np.uint8)

    stylized_pil = Image.fromarray(stylized)
    stylized_pil = stylized_pil.resize(original_size, Image.BICUBIC)
    stylized_pil.save(output_path)
    print(f"Stylized image saved to {output_path}")

    return stylized_pil


if __name__ == "__main__":
    stylize_image(
        model_path=MODEL_PATH,
        input_image_path=INFERENCE_IMG_PATH,
        output_path=OUTPUT_IMG_PATH,
    )
