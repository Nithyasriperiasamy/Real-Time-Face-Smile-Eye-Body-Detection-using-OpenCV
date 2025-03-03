Real-Time-Face-Smile-Eye-Body-Detection-using-OpenCV


Overview
This project uses OpenCV's pre-trained Haar cascade classifiers to detect faces, smiles, eyes, and full bodies in a video file or webcam stream. The program processes each frame in real-time, highlights detected features with bounding boxes, and displays the annotated video.

About the Code
The script reads frames from a video file or webcam, converts them to grayscale, and applies Haar cascade classifiers to detect different features. It draws bounding boxes around detected faces, eyes, smiles, and full bodies, then displays the processed video in real-time. Press 'q' to exit the video stream.

Features
- Detects faces using `haarcascade_frontalface_default.xml`
- Detects eyes within detected faces using `haarcascade_eye.xml`
- Detects smiles within detected faces using `haarcascade_smile.xml`
- Detects full bodies using `haarcascade_fullbody.xml`
- Works with both video files and live webcam feed

Prerequisites
Ensure you have Python and OpenCV installed before running the script.

Install Dependencies

pip install opencv-python


Usage
1. Run with a Video File
Modify the `video_path` variable in the script to your video's location.

Then, run the script:

python face_detection.py


Run with a Webcam
Change the video source to `0` in the script:

video_path = 0 


How It Works
1. Loads OpenCV Haar cascade classifiers.
2. Reads frames from the video or webcam.
3. Converts frames to grayscale for efficient detection.
4. Detects faces, eyes, smiles, and bodies, drawing bounding boxes.
5. Displays the processed video stream.
6. Press 'q' to exit.

Expected Output
Blue Box → Face
Yellow Box → Eyes
Red Box → Smile
Green Box → Full Body


