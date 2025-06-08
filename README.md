# 🔐 E-Video KYC System
Built a custom E-KYC system using HOG-based face detection, Siamese neural networks, and OCR for real-time identity verification via webcam and ID card.

---

A complete Python-based E-KYC (Know Your Customer) verification system that performs:

- 🎯 Face detection (custom HOG-based)
- 🔍 Face verification using a Siamese neural network
- 🧾 OCR to extract ID card details
- 🧑‍💻 Streamlit UI for real-time webcam capture & KYC processing

---

## 🚀 Features

- ✅ **No inbuilt models used** — everything is created from scratch  
- 🧠 Siamese Network trained on your own dataset of face pairs  
- 📸 Webcam capture + Upload options for both ID and selfie  
- 📄 OCR using Tesseract + regex-based field extraction  
- 📊 Live KYC result + optional logging/reporting support

---

## 🧪 Sample KYC Output

```json
{
  "status": "success",
  "kyc_result": "✅ Pass",
  "similarity_score": 0.3215,
  "face_verified": true,
  "kyc_fields": {
    "Name": "RISHABH JAIN",
    "DOB": "15/08/2001",
    "ID Number": "ABCDE1234F"
  }
}
