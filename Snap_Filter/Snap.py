import numpy as np
import cv2
import pandas as pd

eye = cv2.CascadeClassifier("third-party/frontalEyes35x16.xml")
most = cv2.CascadeClassifier("third-party/Nose18x15.xml")

img = cv2.imread("Before.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

glasses = cv2.imread("glasses.png",cv2.IMREAD_UNCHANGED)
glasses = cv2.cvtColor(glasses,cv2.COLOR_BGRA2RGBA)

mostache=cv2.imread("mustache.png",-1)
mostache=cv2.cvtColor(mostache,cv2.COLOR_BGRA2RGBA)

eyes = eye.detectMultiScale(img,1.1,5)

x,y,w,h = eyes[0]

overlay = cv2.imread("glasses.png",cv2.IMREAD_UNCHANGED)
overlay = cv2.cvtColor(overlay,cv2.COLOR_BGRA2RGBA)
overlay = cv2.resize(overlay,(w,h))

for i in range(overlay.shape[0]):
    for j in range(overlay.shape[1]):
        if(overlay[i,j,3]>0):
            img[y+i,x+j,:]=overlay[i,j,:-1]


mst = most.detectMultiScale(img,1.5,5)

x,y,w,h = mst[0]

mostache=cv2.resize(mostache,(w,h))

for i in range(mostache.shape[0]):
    for j in range(mostache.shape[1]):
        if(mostache[i,j,3]>0):
            img[y+i,x+j,:]=mostache[i,j,:-1]

cv2.imshow("Final Picture",img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("final.jpg", img)
img = img.reshape((-1,3))

df = pd.DataFrame(img, columns=["Channel 1","Channel 2", "Channel 3"]).to_csv("Result.csv", index=False)