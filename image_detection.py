import cv2

from detector import detect_animals

# Load image
image_path = r"C:\Users\Dell\Desktop\AI_CNN\Animal_Detection_YOLO\images\dog.jpg"

frame = cv2.imread(image_path)

# Detect animals
frame = detect_animals(frame)

# Show image
cv2.imshow("Image Detection", frame)

# Save output image
cv2.imwrite("outputs/detected_image.jpg", frame)

# Wait until key press
cv2.waitKey(0)

cv2.destroyAllWindows()