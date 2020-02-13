import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import get_model
from dataset import CroppedAppaRealDataset
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ----------
#  Setup
# ----------
# Set testing device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data directories
test_csv = Path('gt_avg_test.csv')
test_data_dir = Path("test/")

# Set up Datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = CroppedAppaRealDataset(test_csv, test_data_dir, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=True, num_workers=0) 

# Initialize model
model = get_model()
model.load_state_dict(torch.load(Path('trained_age_recognition_model.pth')))
model = model.to(device)
model.eval()

# Display test images, predictions, and ground truth labels
for inputs, labels in test_loader:
        # Get input data and corresponding labels
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward 
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)

        # Plot image and print labels
        npimg = inputs.cpu().numpy().squeeze()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

        print('Predicted Age: {} Actual Age: {}'.format(predicted, labels))