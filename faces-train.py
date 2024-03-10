import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
test_dir = os.path.join(BASE_DIR, "testimages")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() # recognizer

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower() #OR os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            id_ = label_ids[label]
            #print(label_ids)
            #y_labels.append(label) # some number
            #x_train.append(path) # verify this image, turn into NUMPY array, GRAY
            
            pil_image = Image.open(path).convert("L") # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8") #converting into array of grayscale values from pil object with unsigned 8-bit values
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            
            if len(faces) > 0:
                print("Face detected in " + path)
            else:
                print("No face detected in " + path)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w] # value to pointer instead of actual value? -> look into this later

                x_train.append(roi)
                y_labels.append(id_)
                
                image_2 = image_array.copy()
                # make a rectangle around the face to see where it detects it
                cv2.rectangle(image_2, (x, y), (x+w, y+h), (255, 0, 0), 10)

                # save this detected face in the testimages folder to see if it is detecting the face correctly
                img_new = str(label) + "-" + str(file)
                cv2.imwrite(os.path.join(test_dir, img_new), image_2)

         
            
            #print(image_array)
            #pil_image.show()

print(label_ids)
print(y_labels)
#print(np.array(y_labels))
#print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")


'''
print(BASE_DIR)
print(os.path.join(BASE_DIR, 'cascades/data/haarcascade_frontalface_default.xml'))
print(os.path.abspath(__file__))

print(image_dir)
'''