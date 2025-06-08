# OCR & field extraction

import cv2
import pytesseract
import re
import os

def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing: optional - thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 4)

    # OCR
    raw_text = pytesseract.image_to_string(thresh)

    return raw_text

def extract_kyc_fields(raw_text):
    text = raw_text.lower()

    # Try to extract Name
    name_match = re.search(r'name\s*[:\-]?\s*([A-Z][a-z]+(?: [A-Z][a-z]+)*)', raw_text, re.IGNORECASE)
    name = name_match.group(1) if name_match else "Not found"

    # Try to extract DOB
    dob_match = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', text)
    dob = dob_match.group(1) if dob_match else "Not found"

    # Try to extract ID number (e.g., Aadhaar, PAN)
    id_match = re.search(r'([A-Z]{5}[0-9]{4}[A-Z])', text)  # PAN pattern
    if not id_match:
        id_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text)  # Aadhaar pattern
    id_number = id_match.group(0) if id_match else "Not found"

    return {
        "Name": name,
        "DOB": dob,
        "ID Number": id_number
    }

# Demo run
if __name__ == "__main__":
    test_image = "sample_id.jpg"  # can Replace with test image path
    text = extract_text_from_image(test_image)
    print("\n--- Extracted Raw Text ---\n", text)

    fields = extract_kyc_fields(text)
    print("\n--- Parsed KYC Fields ---")
    for k, v in fields.items():
        print(f"{k}: {v}")
