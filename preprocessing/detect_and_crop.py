import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage.transform import pyramid_gaussian

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_face_hog(image, window_size=(64, 64), step_size=8, scale=1.5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = []
    
    pyramid = tuple(pyramid_gaussian(image, downscale=scale, max_layer=3)) # guassian pyramid is used here.
    
    for resized in pyramid:
        scale_factor = image.shape[0] / resized.shape[0]
        for (x, y, window) in sliding_window(resized, step_size, window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue

            features = hog(window, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            score = np.mean(features)  # Replace with trained face classifier score. ⚠️ This is not a real classifier. Normally you'd use an SVM or another model trained on HOG features.



            if score > 0.3:  # Dummy threshold; replace with model in later phase
                rx, ry = int(x * scale_factor), int(y * scale_factor)
                rw, rh = int(window_size[0] * scale_factor), int(window_size[1] * scale_factor)
                detections.append((rx, ry, rw, rh))

    return detections


def detect_faces_from_folder(input_folder, output_folder, save_cropped=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for foldername in os.listdir(input_folder):
        fldr_adres = os.path.join(input_folder, foldername)
        # print(foldername)
        for filename in os.listdir(fldr_adres):    
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(fldr_adres, filename)
                image = cv2.imread(img_path)

                detections = detect_face_hog(image)
                if detections:
                    for idx, (x, y, w, h) in enumerate(detections):
                        cropped_face = image[y:y+h, x:x+w]
                        if save_cropped:
                            save_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_face{idx}.jpg")
                            cv2.imwrite(save_path, cropped_face)

