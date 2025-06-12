import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import threading
import time

mp_face_mesh = mp.solutions.face_mesh
recognizer = sr.Recognizer()
mic = sr.Microphone()
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

cam = cv2.VideoCapture(1)

def captureSpeech(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error", e)

with mic as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(mic, captureSpeech)

FACE_IDX = {
    'nose_tip': 1,
    'chin': 152,
    'left_eye_corner': 263,
    'right_eye_corner': 33,
    'left_mouth_corner': 287,
    'right_mouth_corner': 57
}

while True:
    ret, frame = cam.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    h, w = frame.shape[:2]
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(color_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark # Take face landmark
        image_points = np.array([
            [landmarks[FACE_IDX['nose_tip']].x * w, landmarks[FACE_IDX['nose_tip']].y * h],
            [landmarks[FACE_IDX['chin']].x * w, landmarks[FACE_IDX['chin']].y * h],
            [landmarks[FACE_IDX['left_eye_corner']].x * w, landmarks[FACE_IDX['left_eye_corner']].y * h],
            [landmarks[FACE_IDX['right_eye_corner']].x * w, landmarks[FACE_IDX['right_eye_corner']].y * h],
            [landmarks[FACE_IDX['left_mouth_corner']].x * w, landmarks[FACE_IDX['left_mouth_corner']].y * h],
            [landmarks[FACE_IDX['right_mouth_corner']].x * w, landmarks[FACE_IDX['right_mouth_corner']].y * h]
        ], dtype="double")

        model_points = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye
            [225.0, 170.0, -135.0],    # Right eye
            [-150.0, -150.0, -125.0],  # Left mouth
            [150.0, -150.0, -125.0]    # Right mouth
        ])

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        nose_end_3d = np.array([[0, 0, 1000]], dtype="double")
        nose_tip_2d, _ = cv2.projectPoints(
            nose_end_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs
        )

        p1 = tuple(image_points[0].astype(int))  # Nose tip
        p2 = tuple(nose_tip_2d[0][0].astype(int))  # Direction

        cv2.line(frame, p1, p2, (0, 255, 0), 3)  # Draw line from nose
        cv2.circle(frame, p1, 5, (0, 0, 255), -1)
        cv2.circle(frame, p2, 10, (0, 0, 255), -1)

        cv2.line(canvas, (p2[0], p2[1] - 15), (p2[0], p2[1] + 15), (0, 0, 255), 3)
        cv2.line(canvas, (p2[0] - 15, p2[1]), (p2[0] + 15, p2[1]), (0, 0, 255), 3)

    small_frame = cv2.resize(frame, (w // 4, h // 4))

    canvas[0:h // 4, 0:w // 4] = small_frame
    cv2.imshow("Canvas with Camera", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
