import cv2
import os 
from tensorflow.keras.models import load_model
import numpy as np

cap = cv2.VideoCapture(0)

class_labes = ["da_vinci", "van_gogh", "picasso"]
model = load_model("vgg_painting.h5")
#img = cv2.imread("mouse_53.png")

while True:
    ret, img = cap.read()
    if ret:
        img_scaled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        data = img_scaled
        data = data.astype("float")/255.0
        #print(data.shape)
        X= np.asarray([data])
        #print(X.shape)

        s = model(X, training=False)
        index = np.argmax(s)
        if index == 0:
            print("da_vinci")
            strr = 'da_vinci'
        elif index == 1:
            print("van_gogh")
            strr = 'van_gogh'
        elif index == 2:
            print("picasso")
            strr = 'picasso'
        print(index)
        cv2.putText(img, strr, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.imshow('win', img)
        if cv2.waitKey(1)&0xff == ord('q'):
            break

cv2.destroyAllWindows()