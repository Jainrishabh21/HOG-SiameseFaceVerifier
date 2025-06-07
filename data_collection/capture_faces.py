import cv2
import os
import uuid
from preprocessing.detect_and_crop import detect_face_hog

def collect_faces(label_name, output_dir="face_verification_project/data_collection/captured_faces", max_images=20):
    cap = cv2.VideoCapture(0)
    os.makedirs(f"{output_dir}/{label_name}", exist_ok=True)

    count = 0
    print(f"📸 Starting face capture for: {label_name}")
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_face_hog(frame)

        for (x, y, w, h) in detections:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            filename = os.path.join(output_dir, label_name, f"{uuid.uuid4().hex}.jpg")
            cv2.imwrite(filename, face)
            count += 1
            print(f"✅ Captured: {count}/{max_images}")
            break  # Save one face per frame

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("🛑 Face collection complete.")
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# collect_faces("rishabh")
