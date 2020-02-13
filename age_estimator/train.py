import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

from model import get_model
from dataset import CroppedAppaRealDataset
from pathlib import Path

# ----------
#  Setup
# ----------
# Set training device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data directories
train_csv = Path('gt_avg_train.csv')
valid_csv = Path('gt_avg_valid.csv')
test_csv = Path('gt_avg_test.csv')
train_data_dir = Path("train/")
valid_data_dir = Path("valid/")
test_data_dir = Path("test/")

# Set up Datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CroppedAppaRealDataset(train_csv, train_data_dir, transform=transform)
valid_dataset = CroppedAppaRealDataset(valid_csv, valid_data_dir, transform=transform)
test_dataset = CroppedAppaRealDataset(test_csv, test_data_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                          shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64,
                                          shuffle=False, num_workers=0)  
test_loader = torch.utils.data.DataLoader(test_data_dir, batch_size=64,
                                          shuffle=False, num_workers=0) 

# Initialize model
model = get_model()
model = model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define loss
criterion = nn.CrossEntropyLoss()

# Scheduler
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# ----------
#  Training
# ----------
# Train the network
best_mae = 9999999
for epoch in range(101):  # Loop over the dataset multiple times
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Get input data and corresponding labels
        inputs = inputs.to(device)
        labels = labels.long().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))

    # Every set number of epochs, check validation error on
    # validation dataset
    if epoch % 10 == 0:
        # Set model to evaluation mode for validation
        model.eval()

        correct = 0
        total = 0
        preds = []
        ground_truth = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                # Get input data and corresponding labels
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                
                # Forward pass
                outputs = model(inputs)

                # Calculate statistics
                predicted = torch.argmax(outputs, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                preds.append(predicted)
                ground_truth.append(labels)

        # Calculate mean absolute error for ages
        preds = torch.cat(preds, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)
        diff = (preds - ground_truth).float()
        mae = torch.mean(torch.abs(diff))

        # Save best performing model
        if mae < best_mae:
            best_mae = mae

            trained_model_path = Path('trained_age_recognition_model.pth')
            torch.save(model.state_dict(), trained_model_path)

        # Print statistics
        print('Validation accuracy: {:.4f} MAE: {:.4f}'.format(100 * correct / total, mae))

        # Set model back to training mode after validation
        model.train()

    # Update learning rate
    scheduler.step()


print('Finished Training')



