# Tuto : https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/

import cv2
import numpy as np
from imutils.video import FPS
import imutils

print('cv2 version : ', cv2.__version__)

# Paramètres
use_gpu = True
confidence_criteria = 0.5

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

prototxt_file = "/home/cooptek/Desktop/Workspace/MobileNetSSD/MobileNetSSD_deploy.prototxt"
caf_model_file = "/home/cooptek/Desktop/Workspace/MobileNetSSD/MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt_file, caf_model_file)

if use_gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#cam = cv2.VideoCapture("http://admin:Astronomie2552@192.168.1.38/video.cgi?.mjpg")
img = cv2.imread('/home/cooptek/Desktop/Workspace/lena.png') 
fps = FPS().start()

while True:
    touche = cv2.waitKey(33)
    #(ret_val, frame) = cam.read()
    frame = img
    #if not ret_val:
    #    break
    # frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_criteria:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
            )
            # check to see if the output frame should be displayed to our

    cv2.imshow("Frame", frame)
    if touche == ord("q"):
        break

    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows
cam.release()
