import cv2
import time

# Load Haar cascade classifiers
detector_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detector_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
detector_smile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Capture video (use 0 for webcam, or provide a video file path)
video_path = r"C:\Users\PRIYADARSHENE S\Downloads\4586954-uhd_3840_2160_25fps.mp4"
capture = cv2.VideoCapture(video_path if video_path else 0)

# Get video dimensions and scale down for efficiency
frame_width = int(capture.get(3) * 0.5)
frame_height = int(capture.get(4) * 0.5)

paused = False  # Pause control

while capture.isOpened():
    if not paused:
        start_time = time.time()  # Start timer for FPS calculation

        ret, frame = capture.read()
        if not ret:
            break

        # Resize frame to reduce computational load
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes within face
            eyes = detector_eye.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Detect smiles within face
            smiles = detector_smile.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

        # Calculate and display FPS
        fps = int(1 / (time.time() - start_time))
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display output
        cv2.imshow("Face, Smile & Eye Detection", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('p'):  # Pause/Resume
        paused = not paused

# Release resources
capture.release()
cv2.destroyAllWindows()
