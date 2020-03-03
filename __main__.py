import numpy as np
import cv2
import os
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

face_cascade = cv2.CascadeClassifier('treino/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}  #empty dictionary
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("png"):
         path = os.path.join(root,file)
         label = os.path.basename(root).replace(" ","-").lower()
         print(label, path)
         if not label in label_ids:
             label_ids[label] = current_id
             current_id += 1
         id_ = label_ids[label]
         print(label_ids)

         #y_labels.append(label) #some number
         #x_train.append(path) #verify this image, turn into a numpy array and gray
         pil_image = Image.open(path).convert("L")  #grayscale
         image_array = np.array(pil_image,"uint8") #type
         #print(image_array) #convert image in number
         faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

         for(x,y,w,h) in faces:
             roi = image_array[y:y+h, x:x+w]
             x_train.append(roi)
             y_labels.append(id_)

#print(y_labels)
#print(x_train)

#turning the file into a .picle file
with open("labels.picle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("treino/trainner.yml")