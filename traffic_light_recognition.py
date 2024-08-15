import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pylab as plt
 
# 视频路径 输出路径
def video_recognition(video):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    # 打开摄像头 参数为输入流，可以为摄像头或视频文件
    vidcap = cv2.VideoCapture(video)
    count = 0
    green,red = 0,0
    # 视频的帧率
    timeF = 24
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            # 一帧执行一次
            if count % timeF == 0:
                if traffic_light_recognition(image) != None:
                    if traffic_light_recognition(image):
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

    results = model(img, stream=True,classes=9)

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

            numOfZeros = img_located.shape[0]*img_located.shape[1]

            for r in range(0,img_located.shape[0]):
                for c in range(0,img_located.shape[1]):
                    if img_grayscale[r][c]>110:
                        for ch in 0,1,2:
                            mask[r][c][ch]=1
                        numOfZeros-=1

            img_sampled = img_located*mask

            avgEnergy = img_sampled.sum(axis=(0,1))/(img_located.shape[0]*img_located.shape[1]-numOfZeros)

            if (avgEnergy[1]>130 or avgEnergy[0]>130) and avgEnergy[2]<180:

                #fig, ax = plt.subplots(figsize=(10,10))
                #ax.imshow(img_sampled/255)
                #ax.axis("off")
                #plt.show()

                if avgEnergy[1]>avgEnergy[0]:
                    go = True
                else:
                    go = False

                return(go)

video_recognition('video_traffic_light.mp4')