from mtcnn import MTCNN
import cv2
import os

# Initialize MTCNN face detector
detector = MTCNN()

# Load your image
# image_path = 'IMG_5749.jpg'
# image_path = 'IMG_5098.jpg'
# image_path = 'IMG_5142.jpg'
image_path = 'image_97_2.png'

img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# Detect faces
faces = detector.detect_faces(img)

# Directory to save cropped faces
save_dir = 'cropped/'
os.makedirs(save_dir, exist_ok=True)

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
