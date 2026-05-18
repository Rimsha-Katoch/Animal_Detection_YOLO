import cv2

from detector import detect_animals

video_path = r"C:\Users\Dell\Desktop\AI_CNN\Animal_Detection_YOLO\videos\cat.mp4"

cap = cv2.VideoCapture(video_path)

# get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# save output video

out = cv2.VideoWriter("outputs/output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"),fps,(frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_animals(frame)
    out.write(frame)
    cv2.imshow("Video Animal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()