import cv2
import cv2.data

cam = cv2.VideoCapture(1)
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_region_gray = gray[y:y+h//2, x:x+w]
        face_region_color = frame[y:y+h, x:x+w]
        eyes = eye_detection.detectMultiScale(face_region_gray, 1.2, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_region_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_gray = face_region_gray[ey:ey+eh, ex:ex+ew]
            eye_color = face_region_color[ey:ey+eh, ex:ex+ew]
            thresh = cv2.adaptiveThreshold(eye_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            
            if contours:
                (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                cx, cy = int(cx), int(cy)
                cv2.line(frame, (cx, cy), (cx + 50, cy), (0, 255, 0), 2)  # Horizontal line
                cv2.line(frame, (cx, cy), (cx, cy + 50), (0, 255, 0), 2)  # Vertical line
    if not ret:
        break
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()