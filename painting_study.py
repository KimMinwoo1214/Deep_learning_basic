from imutils import paths
import cv2
import os 
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from aspectawarepreprocessor import AspectAwarePreprocessor
from tensorflow.keras.preprocessing.image import ImageDataGenerator

imagePaths = list(paths.list_images("./paintings"))
#print(imagePaths)

data = []
labels = [] 

aap = AspectAwarePreprocessor(64, 64)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath) 
    #image = cv2.resize(image, (64, 64))
    image = aap.preprocess(image)
    data.append(image)
    label =  imagePath.split(os.path.sep)[-2]
    if label == 'da_vinci':
        labels.append(0)
    elif label == 'picasso':
        labels.append(1)
    elif label == 'van_gogh':
        labels.append(2)
    # labels.append(label)

#print(labels[0])
#print(data[0]) 

data = np.array(data)
labels = np.array(labels)

data = data.astype("float")/255

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42, shuffle=True)

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

opt = SGD(learning_rate=0.01, decay =0.01/40, momentum=0.9, nesterov=True)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

model = Sequential()

# CNN First Layer Build
model.add(Conv2D(32, (3,3), padding="same", input_shape=(64, 64, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# CNN Second Layer Build
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Flatten
model.add(Flatten()) #Perceptron
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Class
model.add(Dense(3))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer=opt, metrics=["accuracy"])

#H = model.fit_generator(aug.flow(trainX, trainY), validation_data=(testX, testY), batch_size=64, epochs=15, verbose=1)
H = model.fit(aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,
	epochs=40, verbose=1)

model.save("vgg_painting.h5")
plt.style.use("painting_plot")
plt.figure()
plt.plot(np.arange(0, 10), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 10), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 10), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 10), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("painting_plot.png")
plt.show()