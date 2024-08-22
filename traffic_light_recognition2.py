import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pylab as plt
 
#Capturing the frames
def video_recognition(video):
    vidcap = cv2.VideoCapture(video)
    count = 0
    green,red = 0,0
    timeF = 12
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            if count % timeF == 0:
                res = traffic_light_recognition(image)
                if res != None:
                    if res:
                        green += 1
                    else:
                        red += 1
            count += 1
        else:
            break
    print("green: "+str(green)+" red: "+str(red))
    cv2.destroyAllWindows()
    vidcap.release()

#RGB to grayscale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def traffic_light_recognition(img):
    
    #Locating the traffic light
    model = YOLO("yolov8n.pt")

    results = model(img, stream=True,classes=9,verbose=False)

    for result in results:
        boxes = result.boxes
        #result.show()

        xywh = boxes.xywh

        if xywh.shape[0]!=0:

            col = int(xywh[0][2])
            halfCol = int(col/2) 
            row = int(xywh[0][3])
            halfRow = int(row/2)
            x = int(xywh[0][0])
            y = int(xywh[0][1])

            img_located = np.empty((2*halfRow+1,2*halfCol+1,3))
            
            img_located[:,:,:] = img[y-halfRow:y+halfRow+1,x-halfCol:x+halfCol+1,:]

            img_located = img_located[:,:,::-1]

            #fig, ax = plt.subplots(figsize=(10,10))
            #ax.imshow(img_located/255)
            #ax.axis("off")
            #plt.show()

            #Identifying the color
            img_grayscale = rgb2gray(img_located)

            mask = np.zeros((img_located.shape[0],img_located.shape[1],3))

            for r in range(0,img_located.shape[0]):
                for c in range(0,img_located.shape[1]):
                    if img_grayscale[r][c]>110:
                        for ch in 0,1,2:
                            mask[r][c][ch]=1

            img_sampled = img_located*mask

            numOfGreen,numOfRed = 0,0

            for r in range(0,img_sampled.shape[0]):
                for c in range(0,img_sampled.shape[1]):
                    if sum(img_sampled[r][c]!=0):
                        if img_sampled[r][c][1]>img_sampled[r][c][0]:
                            numOfGreen+=1
                        else:
                            numOfRed+=1

            if numOfGreen>numOfRed:
                go=True
            else:
                go=False

            #fig, ax = plt.subplots(figsize=(10,10))
            #ax.imshow(img_sampled/255)
            #ax.axis("off")
            #plt.show()

            return(go)

video_recognition('video_traffic_light.mp4')