import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

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

images_root = Path('test_images/three_class_test_images/faces')
test_dataset = ImageFolder(root=images_root, transform=transform)

# Set up dataloaders
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=True, num_workers=0)

# Initialize model
model = get_model()
model.load_state_dict(torch.load(Path('three_class_trained_age_recognition_model.pth')))
model = model.to(device)
model.eval()

# Label translation 
label_translation = {
    0: "<= 12",
    1: "13 to 17",
    2: ">= 18"
}

# For un-doing normalization for visualizing image tensor
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

# Display test images, predictions, and ground truth labels
preds = []
ground_truth = []
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
        plt.title('Predicted Age: {}, Actual Age: {}'.format(label_translation[predicted.item()], label_translation[labels.item()])) 
        plt.show()

        preds.append(predicted)
        ground_truth.append(labels)

# Calculate statistics on test set
preds = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truth, dim=0).cpu().numpy()

accuracy = accuracy_score(ground_truth, preds)
precision = precision_score(ground_truth, preds, average=None)
recall = recall_score(ground_truth, preds, average=None)

print('Accuracy: {:.4f}'.format(accuracy))
print('Precision <=12: {:.4f} Precision 13-17: {:.4f} Precision >=18: {:.4f}'.format(
    precision[0], precision[1], precision[2]))
print('Recall <=12: {:.4f} Recall 13-17: {:.4f} Recall >=18: {:.4f}'.format(
    recall[0], recall[1], recall[2]))
