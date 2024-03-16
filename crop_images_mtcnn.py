from mtcnn import MTCNN
import cv2
import os
import matplotlib.pyplot as plt

# Directory to save cropped faces
save_dir = 'cropped/'
os.makedirs(save_dir, exist_ok=True)

# Initialize MTCNN face detector
detector = MTCNN()

# Load specific image
# image_path = 'IMG_5749.jpg'
# image_path = 'IMG_5098.jpg'
# image_path = 'IMG_5142.jpg'
# image_path = 'image_97_2.png'
# image_path = 'image_99_2.png'
# image_path = 'image_108_3.png'
# image_path = 'image_109_2.png'
# image_path = 'image_144_2.png'
# image_path = 'image_345_3.png'
image_path = 'image_600_3.png'

img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Detect faces
faces = detector.detect_faces(img)

copy_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Draw bounding boxes around faces
for i, face in enumerate(faces):
    # Get the bounding box
    x, y, width, height = face['box']
    # Draw rectangle around the face
    cv2.rectangle(copy_image, (x, y), (x+width, y+height), (255, 0, 0), 1)

# Not needed - Only convert to BGR if need to save using OpenCV (OpenCV format)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Test - draw image to screenUse plt to display the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(copy_image)
plt.axis('off')  # Hide axes ticks
plt.show()

# Process each detected face
for i, face in enumerate(faces):
    # Get the bounding box
    x, y, width, height = face['box']
    cropped_face = img[y:y+height, x:x+width]
    
    # Convert RGB to BGR (OpenCV format) for saving
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
    
    # Save each cropped face
    save_path = os.path.join(save_dir, f"cropped{i+1}.png")
    cv2.imwrite(save_path, cropped_face)
    print(f"Saved: {save_path}")
