import os
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomFaceAgeClassificationDataset(Dataset):
    def __init__(self, imgs_path, transform=None):
        self.img_dir = Path(imgs_path)
        self.transform = transform

        self.all_img_names = os.listdir(self.img_dir)


    def __len__(self):
        return len(self.all_img_names)


    def __getitem__(self, idx):
        current_img_path = self.img_dir.joinpath(self.all_img_names[idx])

        current_img = Image.open(current_img_path).convert("RGB")

        if self.transform:
            current_img = self.transform(current_img)

        return current_img, current_img_path


def main():
    # Path to input images
    unsorted_imgs_path = 'test_images/unclassified_faces'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CustomFaceAgeClassificationDataset(imgs_path=unsorted_imgs_path, transform=transform)

    sample, path = dataset[0]

    print('Test')


if __name__ == '__main__':
    main()