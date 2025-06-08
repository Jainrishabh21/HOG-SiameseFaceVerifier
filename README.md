# HOG-SiameseFaceVerifier
A self-contained facial identity verification system using handcrafted HOG-based detection and a custom-trained Siamese neural network with contrastive loss.


---

## 🚀 Phases Implemented

### ✅ Phase 1: Data Collection
Capture face images via webcam and save them in class-wise folders.

### ✅ Phase 2: Face Detection
Custom face detection using **HOG features** and **sliding window** with Gaussian pyramids.

### ✅ Phase 3: Capture faces
Auto face collector from webcam. Capture and save face images from webcam

### ✅ Phase 4: Siamese Model Design
Deep learning model that learns face similarity using **contrastive learning**.

### ✅ Phase 5: Contrastive Loss
Implemented from scratch using margin-based distance loss.

### ✅ Phase 6: Dataset Loader
Custom PyTorch dataset that returns pairs of face images with similarity labels.

### ✅ Phase 7: Training
Trains the Siamese model on paired data and saves the trained weights.

### ✅ Phase 8: Image-Based Verification
Given two input images, predicts if they belong to the same identity.

### ✅ Phase 9: Webcam-Based Verification
Real-time face verification using webcam and trained model.

### ✅ Phase 10: UI
Streamlit UI to upload or capture face images and check verification result.

---

## How It Works
Detection: Detect faces using HOG features + pyramid sliding windows.
Training: Train a Siamese network on positive/negative face pairs.
Verification: Compute embeddings of two faces, measure Euclidean distance.
Decision: If distance < threshold, label as "same person".


## Switch to KYC_IntelliScan branch for E KYC verification model
