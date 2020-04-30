import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from pathlib import Path
import cv2
import numpy as np

from model_unet import UNet
from model_deeplab import DeepLabV3Plus
from utils import utils

import time


def segment_background(deeplab_weights_path=None, input_images_path=None):
    # Set testing device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up model
    # model = UNet(
    #     backbone="mobilenetv2",
    #     num_classes=2,
    # 	pretrained_backbone=None
    # )
    # trained_dict = torch.load('UNet_MobileNetV2.pth', map_location="cpu")['state_dict']
    # model.load_state_dict(trained_dict, strict=False)
    # model = model.to(device)
    # model.eval()
    model = DeepLabV3Plus(backbone="resnet18")
    trained_dict = torch.load(deeplab_weights_path, map_location="cpu")["state_dict"]
    model.load_state_dict(trained_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Output folder to save extracted backgrounds
    output_folder = Path(input_images_path).parent.joinpath("extracted_backgrounds")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Extract backgrounds
    for img_path in Path(input_images_path).iterdir():
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Processing:", str(img_path))
        h, w = img.shape[:2]

        X, pad_up, pad_left, h_new, w_new = utils.preprocessing(
            img, expected_size=320, pad_value=0
        )

        with torch.no_grad():
            mask = model(X.cuda())
            mask = mask[..., pad_up : pad_up + h_new, pad_left : pad_left + w_new]
            mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=True)
            mask = F.softmax(mask, dim=1)
            mask = mask[0, 1, ...].cpu().numpy()

        mask = np.where(mask > 0.9, 0, 1)

        # Apply mask on original RGB image to extract background
        img[:, :, 0] = mask * img[:, :, 0]
        img[:, :, 1] = mask * img[:, :, 1]
        img[:, :, 2] = mask * img[:, :, 2]

        output_name = output_folder.joinpath(img_path.stem + img_path.suffix)
        cv2.imwrite(str(output_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    # Path to DSFD weights
    deeplab_weights_path = "DeepLabV3Plus_ResNet18.pth"

    # Path to input images
    test_img_folder = "test_images/images"

    # Extract faces
    start = time.time()
    segment_background(
        deeplab_weights_path=deeplab_weights_path, input_images_path=test_img_folder
    )
    end = time.time()
    print(end - start)
    print("Finished!")


if __name__ == "__main__":
    main()
