import face_recognition
import cv2
import numpy as np
import os

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir('faces'):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join('faces', filename)
        try:
            # Read image using OpenCV
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"Skipping {filename}: failed to load image.")
                continue

            # Convert to RGB format for face_recognition
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            encoding = face_recognition.face_encodings(image_rgb)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])
            else:
                print(f"No face found in {filename}, skipping.")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_face_names[best_match]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
