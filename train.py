import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from .config import (
    COCO_TRAIN_IMAGES,
    STYLE_IMAGE_PATH,
    OUTPUT_DIR,
    NUM_IMAGES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    CONTENT_WEIGHT,
    STYLE_WEIGHT,
    DEVICE,
    LOG_INTERVAL,
    SAVE_INTERVAL,
    transform,
)
from .dataset import build_dataloader
from .transformers import TransformerNetwork, VGG16
from .utils import gram_matrix, save_checkpoint, save_sample_image


def train():
    print("Loading dataset...")
    dataloader = build_dataloader(
        COCO_TRAIN_IMAGES, max_images=NUM_IMAGES, num_workers=2
    )

    print("Loading style image...")
    style_img = Image.open(STYLE_IMAGE_PATH).convert("RGB")
    style_img = transform(style_img).unsqueeze(0).to(DEVICE)

    print("Initializing networks...")
    TransformerNet = TransformerNetwork().to(DEVICE)
    VGG = VGG16().to(DEVICE)
    VGG.eval()

    # ImageNet normalization for VGG (BGR format, negative mean)
    imagenet_neg_mean = (
        torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32)
        .reshape(1, 3, 1, 1)
        .to(DEVICE)
    )

    print("Extracting style features...")
    with torch.no_grad():
        style_tensor = style_img.add(imagenet_neg_mean)
        B, C, H, W = style_tensor.shape
        style_features = VGG(style_tensor.expand([BATCH_SIZE, C, H, W]))
        style_grams = {k: gram_matrix(v) for k, v in style_features.items()}

    optimizer = optim.Adam(TransformerNet.parameters(), lr=LEARNING_RATE)

    batch_content_loss_sum = 0
    batch_style_loss_sum = 0
    batch_total_loss_sum = 0

    print(f"Starting training for {EPOCHS} epochs...")
    batch_count = 1
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"========Epoch {epoch + 1}/{EPOCHS}========")

        for content_batch in dataloader:
            curr_batch_size = content_batch.shape[0]

            torch.cuda.empty_cache()

            optimizer.zero_grad()

            # Convert RGB to BGR and move to device
            content_batch = content_batch[:, [2, 1, 0]].to(DEVICE)

            # Generate stylized images
            generated_batch = TransformerNet(content_batch)

            # Extract features
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))

            MSELoss = nn.MSELoss().to(DEVICE)

            # Content Loss (from relu2_2 layer)
            content_loss = CONTENT_WEIGHT * MSELoss(
                generated_features["relu2_2"], content_features["relu2_2"]
            )
            batch_content_loss_sum += content_loss.item()

            # Style Loss (from all layers)
            style_loss = 0
            for key, value in generated_features.items():
                generated_gram = gram_matrix(value)
                style_gram = style_grams[key][:curr_batch_size]
                s_loss = MSELoss(generated_gram, style_gram)
                style_loss += s_loss
            style_loss *= STYLE_WEIGHT
            batch_style_loss_sum += style_loss.item()

            total_loss = content_loss + style_loss
            batch_total_loss_sum += total_loss.item()

            total_loss.backward()
            optimizer.step()

            if batch_count % LOG_INTERVAL == 0:
                avg_content = batch_content_loss_sum / batch_count
                avg_style = batch_style_loss_sum / batch_count
                avg_total = batch_total_loss_sum / batch_count
                elapsed = time.time() - start_time
                print(f"Iteration {batch_count}/{EPOCHS * len(dataloader)}")
                print(f"  Content Loss: {avg_content:.2f}")
                print(f"  Style Loss:   {avg_style:.2f}")
                print(f"  Total Loss:   {avg_total:.2f}")
                print(f"  Time elapsed: {elapsed:.0f} seconds")

            if (batch_count % SAVE_INTERVAL == 0) or (
                batch_count == EPOCHS * len(dataloader)
            ):
                print(
                    f"========Iteration {batch_count}/{EPOCHS * len(dataloader)}========"
                )
                print(f"  Content Loss: {batch_content_loss_sum / batch_count:.2f}")
                print(f"  Style Loss:   {batch_style_loss_sum / batch_count:.2f}")
                print(f"  Total Loss:   {batch_total_loss_sum / batch_count:.2f}")
                print(f"Time elapsed: {time.time() - start_time:.0f} seconds")

                save_checkpoint(TransformerNet, batch_count, OUTPUT_DIR)
                save_sample_image(generated_batch, batch_count, OUTPUT_DIR)

            batch_count += 1

    print("\nDone Training the Transformer Network!")
    print(f"Training Time: {time.time() - start_time:.0f} seconds")

    TransformerNet.eval()
    TransformerNet.cpu()
    final_path = f"{OUTPUT_DIR}models/transformer_weight_final.pth"
    torch.save(TransformerNet.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
