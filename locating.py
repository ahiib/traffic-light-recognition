from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pylab as plt

# Load a model
model = YOLO("yolov8s.pt") # pretrained YOLOv8n model

img=cv2.imread("red/red2.jpg")

# Run batched inference on a list of images
results = model(img, stream=True,classes=9)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    #masks = result.masks  # Masks object for segmentation masks outputs
    #keypoints = result.keypoints  # Keypoints object for pose outputs
    #probs = result.probs  # Probs object for classification outputs
    #obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    #result.save(filename="result.jpg")  # save to disk

xywh = boxes.xywh

if xywh.shape[0]!=0:

    col = int(xywh[0][2])
    halfCol = int(col/2) 
    row = int(xywh[0][3])
    halfRow = int(row/2)
    x = int(xywh[0][0])
    y = int(xywh[0][1])

    img_located = np.empty((2*halfRow+1,2*halfCol+1,3))
        
    #for r in range(y-halfRow,y+halfRow):
    #    for c in range(x-halfCol,x+halfCol):
    #        for ch in 0,1,2:
    #            img_located[r-y+halfRow][c-x+halfCol] = img[r][c]

    img_located[:,:,:] = img[y-halfRow:y+halfRow+1,x-halfCol:x+halfCol+1,:]

    img_display = img_located[:,:,::-1]

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img_display/255)
    ax.axis("off")
    plt.show()
