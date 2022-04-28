import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
from tkinter import *
import time

def Start():
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    sound2 = mixer.Sound('phonewarning.mp3')
    face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

    model2 = load_model('converted_keras\\keras_model.h5')

    lbl = ['Close', 'Open']

    model = load_model('models/cnncat2.h5')
    path = os.getcwd()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count = 0
    score = 0
    score2 = 0
    thicc = 2
    rpred = [99]
    lpred = [99]

    while (True):
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        frame2 = cv2.resize(frame, size)
        # #turn the image into a numpy array
        image_array = np.asarray(frame2)
        # # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # # Load the image into the array
        data[0] = normalized_image_array
        #
        prediction = model2.predict(data)
        prediction = np.argmax(prediction, axis=1)
        print(prediction)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            count = count + 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = model.predict(r_eye)
            rpred = np.argmax(rpred, axis=1)
            print(rpred)
            if (rpred[0] == 1):
                lbl = 'Open'
            if (rpred[0] == 0):
                lbl = 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            count = count + 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = model.predict(l_eye)
            lpred = np.argmax(lpred, axis=1)
            if (lpred[0] == 1):
                lbl = 'Open'
            if (lpred[0] == 0):
                lbl = 'Closed'
            break

        if (rpred[0] == 0 and lpred[0] == 0):
            score = score + 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score = score - 2
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if (prediction[0] == 0):
            score2 = score2 + 1
        else:
            score2 = score2 - 1
            if(score2<0):
                score2=0
        if (score2 > 8):
            sound2.play()
            score2 = 0

        cv2.putText(frame, 'Score 2:' + str(score2), (200, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if (score < 0):
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if (score > 20):
            # person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()

            except:  # isplaying = False
                pass
            if (thicc < 16):
                thicc = thicc + 2
            else:
                thicc = thicc - 2
                if (thicc < 2):
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def Stop():
    exit(1)

root=Tk()
root.title("Safe Driving System")
root.geometry('350x200')
lb=Label(root,text="Safe Driving System",font="50",bg="blue",height="5",width="40")
lb.pack()
# lb.grid(column=0,row=0)

btn=Button(root,text="Start System",command=Start,font="40",bg="green")
#btn.grid(column=0,row=1)
btn.pack(side=LEFT)
btn2=Button(root,text="Stop System",command=Stop,font="40",bg="red")
btn2.pack(side=RIGHT)
#btn2.grid(column=1,row=1)
root.mainloop()
