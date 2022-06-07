import cv2
import numpy as np

import os

def nothing(x):
    pass

images = []
for count, f in enumerate(os.listdir("TRAINING")):
    images.append(cv2.imread("TRAINING\\"+f))
    

cv2.namedWindow("window")

cv2.createTrackbar("B","window", 0,len(images)-1,nothing)
cv2.createTrackbar("T","window", 0,255,nothing)
cv2.createTrackbar("I","window", 0,20,nothing)
cv2.createTrackbar("K","window", 0,20,nothing)
cv2.createTrackbar("X","window", 0,500,nothing)
cv2.createTrackbar("Y","window", 0,500,nothing)
cv2.createTrackbar("W","window", 500,500,nothing)
cv2.createTrackbar("H","window", 500,500,nothing)

b=0
t=0
i=0
k=0
x=0
y=0
h=500
w=500

while True:
    img = images[b]
    try:
        cimg = img[y:y+h,x:x+w,:]
    except Exception as e:
        print(e)
        cimg=img.copy()
    rimg = cimg.copy()
    gimg = cv2.cvtColor(cimg.copy(),cv2.COLOR_BGR2GRAY)
    _, th1 = cv2.threshold(gimg.copy(), t,255,cv2.THRESH_BINARY)
    dilation = cv2.erode(th1.copy(),np.ones((k,k),np.uint8),iterations=i)
    contours, _ = cv2.findContours(dilation.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    l = 0
    
    for c in contours:
        if len(c)>l:
            l=len(c)

            maxY = max(c.T[1][0])
            minY = min(c.T[1][0])
            maxX = max(c.T[0][0])
            minX = min(c.T[0][0])
        
    try:
        rimg = cv2.rectangle(rimg,(minX,minY),(maxX,maxY),(0,255,0),3)
    except:
        pass
    rimg = cv2.drawContours(rimg,contours,-1,color=(0,0,255),thickness=1)
    
    cv2.imshow("window1",rimg)
    cv2.imshow("window2",th1)
    cv2.imshow("window3",dilation)
    
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

    b = cv2.getTrackbarPos("B","window")
    t = cv2.getTrackbarPos("T","window")
    i = cv2.getTrackbarPos("I","window")
    k = cv2.getTrackbarPos("K","window")
    x = cv2.getTrackbarPos("X","window")
    y = cv2.getTrackbarPos("Y","window")
    w = cv2.getTrackbarPos("W","window")
    h = cv2.getTrackbarPos("H","window")

   

cv2.destroyAllWindows()

