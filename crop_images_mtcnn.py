from mtcnn import MTCNN
import cv2
import os
import matplotlib.pyplot as plt

def create_directory(dictionary_path):
    os.makedirs(dictionary_path, exist_ok=True)

# Initialize MTCNN face detector
def intialize_detector():
    return MTCNN()

def load_image(image_path):
    try:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        print("Image was opened")   # DELETE ME
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    return image

# Load specific image
# image_path = 'orig-images/image_97_2.png'
# image_path = 'orig-images/image_99_2.png'
# image_path = 'orig-images/image_108_3.png'
# image_path = 'orig-images/image_109_2.png'
# image_path = 'orig-images/image_144_2.png'
# image_path = 'orig-images/image_345_3.png'
# image_path = 'orig-images/image_600_3.png'

def draw_bounding_boxes(image, faces):
    for face in faces:
        x, y, width, height = face['box']   # Get dimensions of bounding box
        print(face['keypoints']) # DELETE ME
        cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 1)

def detect_and_crop_faces(detector, image, save_dir):
    faces = detector.detect_faces(image)
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        cropped_face = image[y:y+height, x:x+width]
        # Convert RGB to BGR (OpenCV format) for saving using OpenCV
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        # Save each cropped face
        save_path = os.path.join(save_dir, f"cropped{i+1}.png")
        cv2.imwrite(save_path, cropped_face)
        print(f"Saved: {save_path}")

def display_image(image):
    print("display image")  # DELETE ME
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Hide axes ticks
    plt.show()

# Not needed - Only convert to BGR if need to save using OpenCV (OpenCV format)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    save_dir = "cropped/"
    create_directory(save_dir)

    detector = intialize_detector()
    image_path = "orig-images/image_109_2.png"
    image = load_image(image_path)
    display_image(image)

    if image is not None:
        faces = detector.detect_faces(image)
        draw_bounding_boxes(image, faces)
        display_image(image)
        detect_and_crop_faces(detector, image, save_dir)
    else:
        print("Failed to load image.")

# Run main function
if __name__ == '__main__':
    main()