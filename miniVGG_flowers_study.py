import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

# 데이터 경로 설정 및 셔플
imagePaths = list(paths.list_images('./flower17_dataset/images'))
random.shuffle(imagePaths)

data = []
labels = []
label_names = ['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil', 
               'daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 
               'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']

label_dict = {name: idx for idx, name in enumerate(label_names)}

# 이미지와 라벨을 리스트에 추가
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    if image is None:
        continue
    image = cv2.resize(image, (64, 64))
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    if label in label_dict:
        labels.append(label_dict[label])
    else:
        print(f"Warning: {label} is not in label_dict")

data = np.array(data)
labels = np.array(labels)

# 데이터와 라벨 길이 확인
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# 데이터 정규화
data = data.astype("float") / 255.0

# 데이터 분할
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42, shuffle=True)

# 라벨 이진화
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 이미지 데이터 증강 설정
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# 모델 정의
model = Sequential()

# 첫 번째 CONV => RELU => POOL
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 두 번째 CONV => RELU => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# FC => RELU layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Softmax classifier
model.add(Dense(len(label_names)))  # 여기서 출력 클래스 수를 17로 설정
model.add(Activation("softmax"))

# 모델 컴파일
opt = SGD(learning_rate=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# tf.data.Dataset을 사용하여 데이터셋을 반복하도록 설정
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
val_dataset = val_dataset.batch(32).repeat()

steps_per_epoch = len(trainX) // 32
validation_steps = len(testX) // 32

# 모델 훈련
H = model.fit(train_dataset, validation_data=val_dataset, 
              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=40, verbose=1)

# 모델 저장
model.save("vgg_flower.h5")

# 학습 곡선 플롯
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Flower Classificsation")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("vgg_flowers.png")
plt.show()
