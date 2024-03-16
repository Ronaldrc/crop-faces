import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Read the image
#img = cv2.imread('ron_hike.jpg')
# img = cv2.imread('IMG_5749.JPG')
img = cv2.imread('image_97_2.png')

# Path to save cropped images to
save_dir = 'cropped/'

# Convert to grayscale (face detection works better on grayscale images)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the pre-trained model (for face detection)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces - returns list of rectangles
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Crop image to only contain faces
if len(faces) > 0:
    for i in range(0, len(faces)):
        x, y, w, h = faces[i]
        # Crop the face region
        face_region = img[y:y+h, x:x+w]
        # Save the cropped image
        cv2.imwrite(f'cropped{i + 1}.png', face_region)
    print("Faces cropped and saved.")
else:
    print("No faces detected.")
    
# # Process each detected face
# for i in range(0, len(faces)):
#     # Get the bounding box
#     x, y, width, height = faces[i]
#     # Crop the face region
#     face_region = img[y:y+height, x:x+width]
    
#     # # Convert RGB to BGR (OpenCV format) for saving
#     # face_region = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)
    
#     # Save each cropped face
#     save_path = os.path.join(save_dir, f"cropped{i+1}.png")
#     cv2.imwrite(save_path, face_region)
#     print(f"Saved: {save_path}")

# Display image using matplotlib
plt.imshow(img)
plt.waitforbuttonpress()
plt.close('all')

# Display all cropped images using matplotlib
for x in range(1, len(faces) + 1):
    cropped_filename = f"{save_dir}/cropped{x}.png"
    cropped_image = cv2.imread(cropped_filename)
    plt.imshow(cropped_image)
    plt.waitforbuttonpress()
    plt.close('all')