from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pylab as plt

#RGB to grayscale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Load a model
model = YOLO("yolov8n.pt") # pretrained YOLOv8n model

num = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

for i in range(0,14):

    img=cv2.imread("green/green"+str(num[i])+".jpg")

    results = model(img,stream=True,classes=9,verbose=False)

    for result in results:
        boxes = result.boxes

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
        #ax.imshow(img_display/255)
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
                #if np.var(img_sampled[r][c])>500:
                if sum(img_sampled[r][c])!=0:
                    if img_sampled[r][c][1]>img_sampled[r][c][0]:
                        numOfGreen+=1
                    else:
                        numOfRed+=1
                #else:
                    #img_sampled[r][c]=[0,0,0]

        if numOfGreen>numOfRed:
            print("picture "+str(num[i])+" is green")
        else:
            print("picture "+str(num[i])+" is red")

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(img_sampled/255)
        ax.axis("off")
        plt.title("picture "+str(num[i]))
        plt.show()

    else:
        print("picture "+str(num[i])+" not spotted")