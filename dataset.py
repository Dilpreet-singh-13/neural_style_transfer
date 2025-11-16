from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

from .config import transform, BATCH_SIZE


class COCODataset(Dataset):
    """COCO Dataset for style transfer training"""

    def __init__(self, img_dir, transform, max_images=None):
        self.img_dir = Path(img_dir)
        self.transform = transform

        # get all image files
        self.image_files = sorted(list(self.img_dir.glob("*.jpg")))

        if max_images:
            self.image_files = self.image_files[:max_images]

        print(f"Loaded {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))


def build_dataloader(img_dir, max_images=None, num_workers=2):
    ds = COCODataset(img_dir, transform, max_images=max_images)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
