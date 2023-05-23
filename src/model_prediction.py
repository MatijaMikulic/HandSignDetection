import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import time
import tensorflow
from tensorflow import keras
from cvzone.ClassificationModule import Classifier

handDetector = HandDetector(maxHands=2)
#loading model
classifier = Classifier("sign_rec_model/keras_model.h5","sign_rec_model/labels.txt")
cv.namedWindow("frame",cv.WINDOW_NORMAL)
cap = cv.VideoCapture(0)

#loading VGG16 model
model=keras.models.load_model("cnn_model")

imgSize = 400
offset = 15

#classes
labels =["A","C","G"]
label="unknown"

while True:
    ret,frame = cap.read(0)
    frame_height,frame_width, _ = frame.shape
    hands, frame =  handDetector.findHands(frame)
    frame_img = np.ones((imgSize,imgSize,3),np.uint8) * 255

    if len(hands) == 2:
        sorted_hands = sorted(hands, key=lambda hand: hand["bbox"][0])

        left_hand  = sorted_hands[0]
        right_hand = sorted_hands[1]

        xl,yl,wl,hl = left_hand['bbox']
        xr,yr,wr,hr = right_hand['bbox']

        x_min=min(xl,xr)
        y_min=min(yl,yr)
        x_max=max(xl+wl,xr+wr)
        y_max=max(yl+hl,yr+hr)

        if x_min-offset >= 0 and y_min-offset >= 0 and x_max+offset <= frame_width and y_max+offset <= frame_height:
            
            imgCrop = frame[y_min-offset:y_max+offset,x_min-offset:x_max+offset]
            try:
                aspectRation = (y_max-y_min)/(x_max-x_min)

                if aspectRation>1:
                    const = imgSize/(y_max-y_min)
                    width_calc = math.ceil(const*(x_max-x_min))
                    imgCrop_resize = cv.resize(imgCrop,(width_calc,imgSize))

                    #center image on frame
                    width_gap = math.ceil((imgSize -width_calc) / 2)
                    frame_img[:,width_gap:width_calc+width_gap] = imgCrop_resize

                else:
                    const = imgSize/(x_max-x_min)
                    height_calc = math.ceil(const*(y_max-y_min))
                    imgCrop_resize = cv.resize(imgCrop,(imgSize,height_calc))

                    #center image on frame
                    height_gap = math.ceil((imgSize -height_calc) / 2)
                    frame_img[height_gap:height_calc+height_gap,:] = imgCrop_resize


                #cv.imshow("Image crop",imgCrop)
                prediction,index = classifier.getPrediction(frame)
                print(index)
                cv.imshow("Frame img",frame_img)
            except:
                print("Hands are out of the boundaries of the image")

    elif len(hands)==1:
        hand = hands[0] 

        x,y,w,h = hand['bbox']

        if x-offset>=0 and y-offset>=0 and x+w+offset <=frame_width and y+h+offset <=frame_height:
            imgCrop = frame[y-offset:y+h+offset,x-offset:x+h+offset]
            try:
                aspectRation = h/w

                if aspectRation>1:
                    const = imgSize/h
                    width_calc = math.ceil(const*w)
                    imgCrop_resize = cv.resize(imgCrop,(width_calc,imgSize))

                    #center image on frame
                    width_gap = math.ceil((imgSize -width_calc) / 2)
                    frame_img[:,width_gap:width_calc+width_gap] = imgCrop_resize

                else:
                    const = imgSize/w
                    height_calc = math.ceil(const*h)
                    imgCrop_resize = cv.resize(imgCrop,(imgSize,height_calc))

                    #center image on frame
                    height_gap = math.ceil((imgSize -height_calc) / 2)
                    frame_img[height_gap:height_calc+height_gap,:] = imgCrop_resize
                
                cv.imshow("Frame img",frame_img)
                prediction,index = classifier.getPrediction(frame)
                print(index)
            except:
                print("Hands are out of the boundaries of the image")

    cv.imshow("frame",frame)
    key = cv.waitKey(1)

    