from picamera2 import Picamera2
import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pylab as plt
import RPi.GPIO as GPIO
from time import sleep

ncnn_model = YOLO("yolov8n_ncnn_model",task="detect")

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

#RGB to grayscale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def traffic_light_recognition(img):
    
    #Locating the traffic light
    results = ncnn_model(img, stream=True,classes=9,verbose=False)
    
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

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()

    results = ncnn_model(frame,classes=9,verbose=False)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    res = traffic_light_recognition(frame)

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(29, GPIO.OUT)

    if(res!=None):
        if(res):
            GPIO.output(29, GPIO.LOW)
            sleep(0.2)
        else:
            GPIO.output(29, GPIO.HIGH)
            sleep(0.2)
    else:
        GPIO.output(29, GPIO.HIGH)
        sleep(0.2)
        
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources and close windows
cv2.destroyAllWindows()
GPIO.cleanup()