# face_verification_project/ui/app.py

import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from kyc.verify_user import run_kyc

st.set_page_config(page_title="E-Video KYC", layout="centered")
st.title("📸 E-Video KYC System")
st.markdown("Upload your **Selfie** and **ID Card Image** to verify identity.")

# File upload
selfie_file = st.file_uploader("Upload Selfie Image", type=["jpg", "jpeg", "png"])
id_file = st.file_uploader("Upload ID Card Image", type=["jpg", "jpeg", "png"])

# Run only if both files are uploaded
if selfie_file and id_file:
    if st.button("✅ Verify KYC"):
        # Save to temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_selfie, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_id:
            tmp_selfie.write(selfie_file.read())
            tmp_id.write(id_file.read())
            tmp_selfie_path = tmp_selfie.name
            tmp_id_path = tmp_id.name

        # Run verification
        with st.spinner("Processing..."):
            result = run_kyc(tmp_id_path, tmp_selfie_path)

        if result["status"] == "success":
            st.success(result["kyc_result"])
            st.markdown(f"**Similarity Score:** `{result['similarity_score']}`")
            st.markdown(f"**Face Verified:** `{result['face_verified']}`")
            st.subheader("📄 Extracted KYC Fields")
            for key, val in result["kyc_fields"].items():
                st.markdown(f"- **{key}:** {val}")
        else:
            st.error(f"❌ KYC Failed: {result['reason']}")
