import cv2
import os 
from tensorflow.keras.models import load_model
import numpy as np

cap = cv2.VideoCapture(0)

class_labes = ["bluebell", "buttercup", "coltsfoot", "cowslip", "crocus", "daffodil", "daisy", "dandelion", "fritillary", "iris", "lilyvalley",
               "pansy", "snowdrop", "sunflower", "tigerlily", "tulip", "windflower"]
model = load_model("vgg_flower.h5")
#img = cv2.imread("mouse_53.png")

while True:
    ret, img = cap.read()
    if ret:
        img_scaled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        # img_path = 'pandas.png'
        # label = img_path[0:-7]
        data = img_scaled  # [[[255, 123, 45 ] [   ]...]]]

        data = data.astype("float")/255.0  # [[[0.896, ] [   ]...]]]
        
        X= np.asarray([data]) # [[[[0.896, ] [   ]...]]]]

        s = model(X, training=False)  # [ 0.5 0.001 0.023]
        index = np.argmax(s)
        if index == 0:
            print("bluebell")
            strr = 'bluebell'
        elif index == 1:
            print("buttercup")
            strr = 'buttercup'
        elif index == 2:
            print("coltsfoot")
            strr = 'coltsfoot'
        elif index == 3:
            print("cowslip")
            strr = 'cowslip'
        elif index == 4:
            print("crocus")
            strr = 'crocus'
        elif index == 5:
            print("daffodil")
            strr = 'daffodil'
        elif index == 6:
            print("daisy")
            strr = 'daisy'
        elif index == 7:
            print("dandelion")
            strr = 'dandelion'
        elif index == 8:
            print("fritillary")
            strr = 'fritillary'
        elif index == 9:
            print("iris")
            strr = 'iris'
        elif index == 10:
            print("lilyvalley")
            strr = 'lilyvalley'
        elif index == 11:
            print("pansy")
            strr = 'pansy'
        elif index == 12:
            print("snowdrop")
            strr = 'snowdrop'
        elif index == 13:
            print("sunflower")
            strr = 'sunflower'
        elif index == 14:
            print("tigerlily")
            strr = 'tigerlily'
        elif index == 15:
            print("tulip")
            strr = 'tulip'
        elif index == 16:
            print("windflower")
            strr = 'windflower'
        print(index)
        cv2.putText(img, strr, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.imshow('win', img)
        if cv2.waitKey(1)&0xff == ord('q'):
            break

cv2.destroyAllWindows()