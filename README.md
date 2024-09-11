This Python script is designed to detect driver drowsiness and monitor posture in real-time using a camera. It uses OpenCV for video capture and image processing, Dlib for facial landmark detection, and Mediapipe for body pose estimation. The key components of the script are as follows:

Camera Setup: The script captures real-time video using the computer's camera.
Facial Landmark Detection: Dlib detects facial landmarks, focusing on the eyes to calculate the Eye Aspect Ratio (EAR). This helps detect drowsiness by monitoring eyelid movements.
Pose Detection: Mediapipe's Pose solution detects the position of key body landmarks like shoulders, hips, and head to check if the user is sitting with proper posture.
Drowsiness Detection: If the EAR falls below a threshold, indicating drowsiness, a timer starts. If drowsiness persists for more than 3 seconds, an alert is displayed.
Posture Monitoring: The script compares the Y-coordinate differences of the shoulders and hips to assess posture. If poor posture is detected for more than 3 seconds, the system prompts the user to adjust their seat.
User Interaction: If poor posture is detected, the system prompts the user to set a seat adjustment range manually.
The system continuously displays messages like "DROWSINESS DETECTED" or "PROPERLY SITTING" based on real-time analysis and allows the user to interact when adjustments are needed. The video feed with detection results is displayed, and pressing 'Esc' exits the program.
