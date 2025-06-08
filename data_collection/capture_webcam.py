# Real-time image capture -> capture a live selfie or ID image using their webcam

import cv2
import os

def capture_image(save_path="captured_image.jpg", window_title="Webcam Capture"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("[INFO] Press 'c' to capture image. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(window_title, frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            cv2.imwrite(save_path, frame)
            print(f"[INFO] Image saved to {save_path}")
            break
        elif key == ord('q'):
            print("[INFO] Capture aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path if os.path.exists(save_path) else None
