import os
import shutil
import cv2

# Define source and destination folders
source_folder = "dataset/unclassified"
wanted_folder = "dataset/wanted"
unwanted_folder = "dataset/unwanted"

# Create directories if they don't exist
os.makedirs(wanted_folder, exist_ok=True)
os.makedirs(unwanted_folder, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Function to resize image to a max of 700x700 pixels while maintaining aspect ratio
def resize_image(img, max_size=700):
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h)  # Scale based on the larger dimension
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Loop through images for labeling
for image in image_files:
    img_path = os.path.join(source_folder, image)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Skipping {image}, unable to load.")
        continue

    # Resize image if it's larger than 700x700
    img = resize_image(img, max_size=700)

    # Create a named window and bring it to the front
    cv2.namedWindow("Image Labeling", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty("Image Labeling", cv2.WND_PROP_TOPMOST, 1)  # Make it appear on top

    # Show image
    cv2.imshow("Image Labeling", img)
    
    cv2.waitKey(1)  # Small delay to render the image properly
    key = cv2.waitKey(0)  # Wait for key press
    
    if key == ord('w'):  # 'w' key for wanted images
        shutil.move(img_path, os.path.join(wanted_folder, image))
    elif key == ord('u'):  # 'u' key for unwanted images
        shutil.move(img_path, os.path.join(unwanted_folder, image))
    elif key == 27:  # Escape key to exit
        break

cv2.destroyAllWindows()
