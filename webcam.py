import cv2
import time

from detector import detect_animals

cap = cv2.VideoCapture(0) # Starts webcam capture (0) means laptop webcam
prev_time = 0

while True:
    ret,frame = cap.read() # Reads one frame from webcam.

    if not ret:
        break

    # Detect animals
    frame = detect_animals(frame)
    # Calculate FPS
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # Display FPS
    cv2.putText(frame,f"FPS: {int(fps)}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 255),2)

    cv2.imshow(" Animal Detection System", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # Press S to save screenshort
    if key == ord("s"):
        # Create unique filename using timestamp
        filename = f"outputs/detected_{int(time.time())}.jpg"
        cv2.imwrite(filename,frame)
        print(f"Screenshort saved as {filename}")

    # Press q to quit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()