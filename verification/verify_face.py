# Phase 8: Inference Script

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from model.siamese_model import SiameseNetwork
from preprocessing.detect_and_crop import detect_face_hog
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("saved_models/siamese_model.pth", map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

def preprocess_single_image(image_path):
    """Loads an image, detects face, crops and applies transform."""
    image = cv2.imread(image_path)
    detections = detect_face_hog(image)
    
    if len(detections) == 0:
        raise Exception(f"No face detected in {image_path}")
    
    # Assume first face
    x, y, w, h = detections[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100))
    face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    
    return transform(face).unsqueeze(0).to(device)

def verify(image_path1, image_path2, threshold=0.5):
    img1 = preprocess_single_image(image_path1)
    img2 = preprocess_single_image(image_path2)

    with torch.no_grad():
        output1 = model.forward_once(img1)
        output2 = model.forward_once(img2)

    euclidean_distance = torch.norm(output1 - output2).item()
    
    print(f"Distance: {euclidean_distance:.4f}")
    
    if euclidean_distance < threshold:
        print("✅ Same person")
    else:
        print("❌ Different person")

# Example usage
if __name__ == "__main__":
    verify("data/test/img1.jpg", "data/test/img2.jpg")
