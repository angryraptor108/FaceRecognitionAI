import numpy as np
import cv2

#using alt-2 from previous testing
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
    # capture frame by frame
    _, frame = cap.read()
    if not _:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w] # (ycord_start, ycord_end)
        roi_colour = frame[y:y+h, x:x+w]

        #recognize?

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        colour = (255, 0, 0) # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), colour, stroke)



    # display the resulting frame
    frame = cv2.flip(frame, 1)
    cv2.imshow('chinmay camera feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()

