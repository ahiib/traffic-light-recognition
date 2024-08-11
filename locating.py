from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolov8n.pt") # pretrained YOLOv8n model

img=cv2.imread("green/green0.jpg")

# Run batched inference on a list of images
results = model(img, stream=True,classes=9)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk