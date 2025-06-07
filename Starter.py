# Phase 7: Training the Siamese Model with Contrastive Loss and Preprocessing
# File: face_verification_project/model/train_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model.siamese_model import SiameseNetwork
from model.loss import ContrastiveLoss
from utils.siamese_dataset import SiameseFaceDataset
from preprocessing.detect_and_crop import detect_faces_from_folder
import os

# Step 1: Preprocess and crop faces
data_dir = "D:\\Study material\\face_verification_project\\lfw-deepfunneled"
preprocessed_data_dir = "D:\\Study material\\face_verification_project\\faces_cropped"
os.makedirs(preprocessed_data_dir, exist_ok=True)
detect_faces_from_folder(input_folder = data_dir, save_cropped=True, output_folder=preprocessed_data_dir)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
epochs = 20
learning_rate = 0.001
margin = 1.0  # for contrastive loss

# Transforms
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Dataset and DataLoader
dataset = SiameseFaceDataset(root_dir=preprocessed_data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = SiameseNetwork().to(device)

# Loss and optimizer
criterion = ContrastiveLoss(margin=margin)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for img1, img2, label in dataloader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        # Forward
        output1, output2 = model.forward_once(img1), model.forward_once(img2)
        loss = criterion(output1, output2, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/siamese_model.pth")
print("\nModel trained and saved to 'saved_models/siamese_model.pth'")
