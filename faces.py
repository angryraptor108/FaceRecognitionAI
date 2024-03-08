import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
#eye_casecade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create() # recognizer
recognizer.read("trainer.yml")


labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # capture frame by frame
    _, frame = cap.read()
    if not _:
        break
    frame2 = cv2.flip(frame, 1) # just to display a non-inverted image
    width_orig = int(frame.shape[1])
    #height_orig = int(frame.shape[0]) dont need this since it's only horizontally inverted

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.6, minNeighbors=5)

    #print("FRAME: ")
    #print(frame)
    #print("faces: ")
    #print(faces)

    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w] # (ycord_start, ycord_end)
        roi_colour = frame[y:y+h, x:x+w]

        #recognize? this is not a deep learning model prediction like keras, tensorflow, pytorch, scikit learn
        id_, conf = recognizer.predict(roi_gray)
        print(conf)
        if conf >= 35 and conf <= 85:
            #print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(frame2, name, (width_orig-x-w, y-20), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "random lurker"
            cv2.putText(frame2, name, (width_orig-x-w, y-20), font, 1, (255, 0, 255), 2, cv2.LINE_AA)



        img_item = "detected-image.png"
        cv2.imwrite(img_item, roi_gray)

        colour = (255, 0, 0) # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame2, (width_orig-x, y), (width_orig-end_cord_x, end_cord_y), colour, stroke)


        # EYES DETECTION
        #eyes = eye_casecade.detectMultiScale(frame, 1.5, 20)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(frame2, (width_orig-ex-ew, ey), (width_orig-ex+ew, ey+eh), (0, 255, 0), 2)



    # display the resulting frame
    cv2.imshow('who are you?', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()

