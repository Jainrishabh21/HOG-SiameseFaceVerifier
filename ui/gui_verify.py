# Phase 10: GUI Face Verification with Tkinter

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from model.siamese_model import SiameseNetwork
from preprocessing.detect_and_crop import detect_face_hog
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("saved_models/siamese_model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

class FaceVerificationApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Verification App")

        self.label = tk.Label(master, text="Select two face images to verify:")
        self.label.pack()

        self.image1_button = tk.Button(master, text="Select Image 1", command=self.load_image1)
        self.image1_button.pack()

        self.image2_button = tk.Button(master, text="Select Image 2", command=self.load_image2)
        self.image2_button.pack()

        self.verify_button = tk.Button(master, text="Verify Faces", command=self.verify_faces)
        self.verify_button.pack()

        self.image1 = None
        self.image2 = None

    def load_image1(self):
        path = filedialog.askopenfilename()
        if path:
            self.image1 = self.process_image(path)
            self.display_image(path, "Image 1")

    def load_image2(self):
        path = filedialog.askopenfilename()
        if path:
            self.image2 = self.process_image(path)
            self.display_image(path, "Image 2")

    def process_image(self, path):
        image = cv2.imread(path)
        detections = detect_face_hog(image)
        if not detections:
            messagebox.showerror("Error", "No face detected.")
            return None
        x, y, w, h = detections[0]
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        return transform(face_pil).unsqueeze(0).to(device)

    def display_image(self, path, title):
        img = Image.open(path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(image=img)
        panel.image = img
        panel.pack()

    def verify_faces(self):
        if self.image1 is None or self.image2 is None:
            messagebox.showwarning("Input Missing", "Both images are required.")
            return

        with torch.no_grad():
            output1 = model.forward_once(self.image1)
            output2 = model.forward_once(self.image2)
            distance = torch.norm(output1 - output2).item()

        msg = f"Distance: {distance:.4f}\n"
        if distance < 0.5:
            msg += "✅ Same Person"
        else:
            msg += "❌ Different Person"

        messagebox.showinfo("Verification Result", msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVerificationApp(root)
    root.mainloop()
