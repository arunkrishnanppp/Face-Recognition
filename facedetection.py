import cv2,numpy as np,os

alg='haarcascade_frontalface_default.xml'
#image_data is the folder containing images fo faces
dataset='image_data'

print('Training Begin')
(images,labels,names,id)=([],[],{},0)


for (subdirs,dirs,files) in os.walk(dataset):
    for subdir in dirs:
        names[id]=subdir
        subjectPath=os.path.join(dataset,subdir)
        for filename in os.listdir(subjectPath):
            path=subjectPath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1
        
print(images)

def mapper(lab):
    return names[lab]
        
images,labels=[np.array(lis) for lis in [images,labels]]
print(images.shape)
print(labels.shape)
width,height=130,130
model=cv2.face.LBPHFaceRecognizer_create()
#model=cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)
cam=cv2.VideoCapture(0)
haar_cascade=cv2.CascadeClassifier(alg)
cnt=0
while True:
    retv,frame=cam.read()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(img,1.3,4)
    if not len(list(face)) == 0:
        no_faces=face.shape[0]
        print(no_faces)
    
    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(245,50,45),2)
        cimg=img[y:y+h,x:x+w]
        resizedImg=cv2.resize(cimg,(width,height))
        pred=model.predict(resizedImg)[1]
        print(pred)
        if pred<88:
            cv2.putText(frame,mapper(model.predict(resizedImg)[0]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 0, 0),2)
#        print(mapper(model.predict(resizedImg)[0]))
            cnt=0
        else:
            print(cnt)
            cnt+=1
            cv2.putText(frame,'UNKNOWN',(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 0, 0),2)
            if cnt>100:
                print("unknown Person")
                cv2.imwrite('unknown.jpg',resizedImg)
                cnt=0
            
        
    cv2.imshow('CAM',frame)
    if ord('q')==cv2.waitKey(10):
        break
cam.release()
cv2.destroyAllWindows()
    


#load the model