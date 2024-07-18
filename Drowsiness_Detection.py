import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance
import cv2
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")

# url = 'http://192.168.31.244:8081/video'
cap = cv2.VideoCapture(0)
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#status marking for the current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0,0,0)


def eye_aspect_ratio(eye):
    # vertical points
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # horizontal points
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B)/(2.0*C)

    #checking if the eye is blinked or not
    if ear > 0.25:   # normal
        return 2
    elif 0.21 < ear <= 0.25:  # drowsy
        return 1
    else:   # sleeping
        return 0

while(True):
    # cap.read() returns 2 values i.e boolean value true or false whether the input is taken or not
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect(gray, 0)  #detecting face using get_frontal_face_detector in grayscale color
    for face in faces:
        (x,y,w,h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        landmarks = predict(gray, face)  #detecting landmakrs in the face
        landmarks = face_utils.shape_to_np(landmarks)  # converting the face to the x,y coordinates using numpy

        # landmarks of the eye and the mouth
        left_blink = landmarks[36:42]
        right_blink = landmarks[42:48]
        # mouth = landmarks[48:68]

        ear_left = eye_aspect_ratio(left_blink)
        ear_right = eye_aspect_ratio(right_blink)
        # mar = mouth_aspect_ratio(mouth)

        if(ear_left==0 or ear_right ==0 ):
            sleep+=1
            drowsy=0
            active=0
            if(sleep==6):
                status = "Sleeping!!!"
                color = (0,0,255)
                mixer.music.play()

        elif(ear_left==1 or ear_right ==1):
            sleep =0
            drowsy+=1
            active=0
            if(drowsy ==6):
                status = "Drowsy!!!"
                color = (255,0,0)

        else:
            sleep=0
            drowsy=0
            active+=1
            if(active==6):
                status = "Active!!!"
                color = (0,255,0)

        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,color, 3)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
