import cv2
import os
import urllib.request
import numpy as np
import imutils

dataset='image_data'

#u have to chnage the foldernames on different persons
foldername='name'

path=os.path.join(dataset,foldername)
if not os.path.isdir(path):
    os.mkdir(path)
    
(width,height)=(130,100)

alg='haarcascade_frontalface_default.xml'
start=False
#loading haarcascade fronterface algorithm
haar_cascade=cv2.CascadeClassifier(alg)
print(haar_cascade)

#using mobile cam
def mobilecam(count):
    url='http://192.168.18.38:8080/shot.jpg'
    while True:
        imgPath=urllib.request.urlopen(url)
        imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
        image=cv2.imdecode(imgNp,-1)
        image=imutils.resize(image,width=450)
# =============================================================================
# cam=cv2.VideoCapture(0)
# while count<50:
# =============================================================================
#    retv,frame=cam.read()
        img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face=haar_cascade.detectMultiScale(img,1.3,4)
        for x,y,w,h in face:
            if start:
                cv2.rectangle(image,(x,y),(x+w,y+h),(245,50,45),2)
                cimg=img[y:y+h,x:x+w]
                resizedImg=cv2.resize(cimg,(width,height))
                cv2.imwrite('%s/%s.jpg' %(path,count),resizedImg)
                count+=1
            cv2.imshow('Frame',image)
            key=cv2.waitKey(10000)
            if k==ord('a'):
                start= not start
            if key==ord('q'):
                break
    #cam.release()
    cv2.destroyAllWindows()

#using webcam
def webcam(count,start):
    
    cam=cv2.VideoCapture(0)
    while count<50:

        retv,frame=cam.read()
        img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face=haar_cascade.detectMultiScale(img,1.3,4)
        for x,y,w,h in face:
            if start:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(245,50,45),2)
                cimg=img[y:y+h,x:x+w]
                resizedImg=cv2.resize(cimg,(width,height))
                cv2.imwrite('%s/%s.jpg' %(path,count),resizedImg)
                count+=1
            cv2.imshow('Frame',frame)
            key=cv2.waitKey(1000)
            if key==ord('a'):
                start= not start
            if key==ord('q'):
                break
    cam.release()
    cv2.destroyAllWindows()
count=1
#mobilecam(count)
start=False
webcam(count,start)
    
