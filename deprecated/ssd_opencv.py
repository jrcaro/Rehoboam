import numpy as np
import cv2

image_path = '/home/jrcaro/rehoboam/images/camara1017-15072020_103324.jpg'
class_path = '/home/jrcaro/rehoboam/darknet/cfg/coco.names'
weigths_path = '/home/jrcaro/rehoboam/SSD300/MobileNetSSD_deploy.caffemodel'
model_path = '/home/jrcaro/rehoboam/SSD300/deploy.prototxt'
threshold = 0.2

# leer fichero de clases
classes = None
with open(class_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

#generar colores para las diferentes clases
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

print("[INFO] loading model…")
net = cv2.dnn.readNet(model_path, weigths_path)
image = cv2.imread(image_path)

(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
print("[INFO] computing object detections…")

net.setInput(blob)
detections = net.forward()

for i in np.arange(0, detections.shape[2]):
     confidence = detections[0, 0, i, 2]
     
     if confidence > threshold:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

print("[INFO] {}".format(label))     
cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
y = (startY - 15) if (startY -15) > 15 else (startY +15)     
cv2.putText(image, label, (startX, y),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
cv2.imshow("Output", image)
cv2.waitKey(0)