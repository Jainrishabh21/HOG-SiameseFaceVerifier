# Phase 10: Integrate face verification + OCR for full KYC decision

import os
import cv2
import torch
from model.siamese_model import SiameseNetwork
from preprocessing.detect_and_crop import detect_and_crop_face
from verification.verify_face import verify
from ocr.extract_text import extract_text_from_image, extract_kyc_fields

# Load Siamese model
MODEL_PATH = "model/siamese_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = SiameseNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# Main KYC verifier
def run_kyc(id_image_path, selfie_image_path, face_threshold=0.5):
    if not os.path.exists(id_image_path) or not os.path.exists(selfie_image_path):
        return {"status": "error", "reason": "Missing input files"}

    # Load and crop faces
    id_face = detect_and_crop_face(id_image_path)
    selfie_face = detect_and_crop_face(selfie_image_path)

    if id_face is None or selfie_face is None:
        return {"status": "fail", "reason": "Face not detected in one or both images"}

    # Resize for model input
    id_face = cv2.resize(id_face, (100, 100))
    selfie_face = cv2.resize(selfie_face, (100, 100))

    # Face Verification
    model = load_model()
    similarity_score = verify(model, id_face, selfie_face)

    face_verified = similarity_score < face_threshold

    # OCR on ID image
    raw_text = extract_text_from_image(id_image_path)
    kyc_fields = extract_kyc_fields(raw_text)

    # Basic field check
    fields_valid = all(v != "Not found" for v in kyc_fields.values())

    # Final KYC result
    kyc_pass = face_verified and fields_valid

    return {
        "status": "success",
        "kyc_result": "✅ Pass" if kyc_pass else "❌ Fail",
        "similarity_score": round(similarity_score, 4),
        "face_verified": face_verified,
        "kyc_fields": kyc_fields
    }

# Demo
if __name__ == "__main__":
    id_img = "sample_data/id.jpg"
    selfie_img = "sample_data/selfie.jpg"

    result = run_kyc(id_img, selfie_img)
    print("\n--- KYC RESULT ---")
    for key, value in result.items():
        print(f"{key}: {value}")
