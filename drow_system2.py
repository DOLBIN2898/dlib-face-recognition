import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
import time

# Setting up the camera
cap = cv2.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye


shoulder_diff_threshold = 0.05
hip_diff_threshold = 0.05
head_pose_threshold = 15

def check_pose(landmarks):
    shoulder_diff = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - 
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    hip_diff = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - 
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
    
    if shoulder_diff > shoulder_diff_threshold or hip_diff > hip_diff_threshold:
        return False

    head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

    head_pose_angle = np.degrees(np.arctan2(abs(head_y - (left_shoulder_y + right_shoulder_y) / 2), 1))
    if head_pose_angle > head_pose_threshold:
        return False

    return True


drowsiness_start_time = None
posture_start_time = None
drowsiness_detected = False
posture_detected = False
adjustment_range_prompted = False
adjustment_range = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = 42 if n == 47 else n + 1
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = 36 if n == 41 else n + 1
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = round((left_Eye + right_Eye) / 2, 2)

        if Eye_Rat < 0.25:
            if not drowsiness_detected:
                drowsiness_start_time = time.time()
                drowsiness_detected = True
            elif time.time() - drowsiness_start_time >= 3:
                if not adjustment_range_prompted:
                    print("Vehicle is stopping, please wake up")
        else:
            drowsiness_detected = False

        if drowsiness_detected and time.time() - drowsiness_start_time < 3:
            cv2.putText(frame, "DROWSINESS DETECTED", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "AWAKE", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        if check_pose(landmarks):
            if posture_detected:
                posture_detected = False
                adjustment_range_prompted = False
                cv2.putText(frame, "PROPERLY SITTING", (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        else:
            if not posture_detected:
                posture_start_time = time.time()
                posture_detected = True
            elif time.time() - posture_start_time >= 3:
                if not adjustment_range_prompted:
                    print("Not properly sitting, select a seat adjustment range (1-10):")
                    try:
                        adjustment_range = int(input("Enter adjustment range (1-10): "))
                        if 1 <= adjustment_range <= 10:
                            print(f"Seat adjustment range set to: {adjustment_range}")
                        else:
                            print("Invalid range. Please enter a number from 1 to 10.")
                    except ValueError:
                        print("Invalid input. Please enter a valid number.")
                    adjustment_range_prompted = True
                cv2.putText(frame, f"Seat adjustment range set to: {adjustment_range}", (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        if adjustment_range_prompted:
            cv2.putText(frame, f"Seat adjustment range set to: {adjustment_range}", (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    else:
        posture_detected = False

    cv2.imshow("Drowsiness and Pose Detector", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
