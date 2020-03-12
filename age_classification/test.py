import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from model import get_model
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ----------
#  Setup
# ----------
# Set testing device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Set up Datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

images_root = Path('test_images/faces_test_images')
test_dataset = ImageFolder(root=images_root, transform=transform)

# Set up dataloaders
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=True, num_workers=0)

# Initialize model
model = get_model()
model.load_state_dict(torch.load(Path('trained_age_recognition_model.pth')))
model = model.to(device)
model.eval()

# Label translation 
label_translation = {
    0: "Adult",
    1: "Child"
}

# For un-doing normalization for visualizing image tensor
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

# Display test images, predictions, and ground truth labels
for inputs, labels in test_loader:
        # Get input data and corresponding labels
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward 
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)

        # Plot image and print labels
        npimg = inv_normalize(inputs.squeeze()).cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

        print('Predicted Label: {}, Actual Label: {}'.format(label_translation[predicted.item()], label_translation[labels.item()]))