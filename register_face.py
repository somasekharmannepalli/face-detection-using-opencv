import cv2
import os

name = input("Enter the name of the person: ")
folder = 'faces'
if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Register Face - Press 's' to save, 'q' to quit", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        path = os.path.join(folder, f"{name}.jpg")
        cv2.imwrite(path, frame)
        print(f"Face saved to {path}")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
