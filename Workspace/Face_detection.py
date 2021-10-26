import numpy as np
import cv2

print('cv2 version : ', cv2.__version__)

def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


face_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_eye.xml"
)
#cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
img = cv2.imread('/home/cooptek/Desktop/Workspace/lena.png')

while True:
    touche = cv2.waitKey(33)
    #ret_val, img = cam.read()
    #if not ret_val:
    #    break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("Frame", img)
    if touche == ord("q"):
        break

cv2.destroyAllWindows
cam.release()
