# Phase 9: Real-Time Face Verification via Webcam

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.siamese_model import SiameseNetwork
from preprocessing.detect_and_crop import detect_face_hog
import numpy as np

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("saved_models/siamese_model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

def capture_and_crop_face():
    """Capture a single face from webcam and return processed tensor."""
    cap = cv2.VideoCapture(0)
    print("Press 's' to capture the face.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            detections = detect_face_hog(frame)
            if not detections:
                print("⚠️ No face detected. Try again.")
                continue

            x, y, w, h = detections[0]
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            cap.release()
            cv2.destroyAllWindows()
            return face_tensor

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

def verify_faces(tensor1, tensor2, threshold=0.5):
    """Run the Siamese network and print similarity."""
    with torch.no_grad():
        output1 = model.forward_once(tensor1)
        output2 = model.forward_once(tensor2)
        distance = torch.norm(output1 - output2).item()

    print(f"Distance: {distance:.4f}")
    if distance < threshold:
        print("✅ Same Person")
    else:
        print("❌ Different Person")

        
# When the webcam opens, press s to capture the first image.
# Repeat the process for the second image.
if __name__ == "__main__":
    print("\nStep 1: Capture the first face...")
    face1 = capture_and_crop_face()
    if face1 is None:
        print("Aborted.")
        exit()

    print("\nStep 2: Capture the second face...")
    face2 = capture_and_crop_face()
    if face2 is None:
        print("Aborted.")
        exit()

    print("\n🔍 Verifying...")
    verify_faces(face1, face2)
