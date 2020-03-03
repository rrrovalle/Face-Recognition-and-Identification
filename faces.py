import numpy as np
import cv2
import pickle


recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('treino/haarcascade_frontalface_alt2.xml')
recognizer.read("treino/trainner.yml")

labels = {"person_name": 1}
with open("labels.picle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]  #(ycord-start, ycord_end)#location of the frame
        roi_color = frame[y:y+h, x:x+w]

        #recognize
        id_, conf = recognizer.predict(roi_gray)
        if conf>= 45: # and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,2,color,stroke,cv2.LINE_AA)


        img_item = "treino/my_image.png"

        cv2.imwrite(img_item, roi_gray)

        color = (255,0,0) #BGR COLOR 0-255
        stroke = 2
        end_cor_x = x + w
        end_cor_y = y + h
        cv2.rectangle(frame, (x,y), (end_cor_x,end_cor_y),color,stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()