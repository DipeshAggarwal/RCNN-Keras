from core.nms import non_max_suppression
from core import config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/1.jpg", help="Path to input image")
args = vars(ap.parse_args())

print("[INFO] Loading Model and Label binarizer...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)

print("[INFO] Running selective search...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

proposals = []
boxes = []

for x, y, w, h in rects[:config.MAX_PROPOSALS_INFER]:
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
    
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
    
    proposals.append(roi)
    boxes.append((x, y, x + w, y + h))
    
proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")
print("[INFO] Proposal Shape: {}".format(proposals.shape))

print("[INFO] Classifying proposals...")
probs = model.predict(proposals)

print("[INFO] Applying NMS...")
labels = lb.classes_[np.argmax(probs, axis=1)]
idxs = np.where(labels == "racoon")[0]

boxes = boxes[idxs]
probs = probs[idxs][:, 1]

idxs = np.where(probs >= config.MIN_PROB)
boxes = boxes[idxs]
probs = probs[idxs]

clone = image.copy()

for box, prob in zip(boxes, probs):
    start_x, start_y, end_x, end_y = box
    cv2.rectangle(clone, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
    text = "Racoon: {:.2f}%".format(prob * 100)
    cv2.putText(clone, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
cv2.imshow("Before NMS", clone)

box_idxs = non_max_suppression(boxes, probs)

for i in box_idxs:
    start_x, start_y, end_x, end_y = box_idxs[i]
    cv2.rectangle(clone, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
    text = "Racoon: {:.2f}%".format(probs[i] * 100)
    cv2.putText(clone, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

cv2.imshow("After NMS", image)
cv2.waitKey(0)
