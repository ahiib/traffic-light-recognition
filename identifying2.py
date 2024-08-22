import numpy as np
import cv2
import matplotlib.pylab as plt
from glob import glob

list_img = glob('green/*jpg')
img = cv2.imread(list_img[15])
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_grayscale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

mask = np.zeros((img.shape[0],img.shape[1],3))

for r in range(0,img.shape[0]):
    for c in range(0,img.shape[1]):
        if img_grayscale[r][c]>110:
            for ch in 0,1,2:
                mask[r][c][ch]=1

img_sampled = img*mask

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
    result='The light is green'
else:
    result='The light is red'

print(result)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(img_sampled/255)
ax.axis("off")
plt.show()
#print(img_grayscale)