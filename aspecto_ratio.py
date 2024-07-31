import cv2
from aspectawarepreprocessor import AspectAwarePreprocessor

uga = AspectAwarePreprocessor(64, 64)

img_cat = cv2.imread('./Can_classifier/miniVGGNet_animals/animals/cats/cats_00002.jpg')
print(img_cat.shape)
cv2.imshow('cats', img_cat)
cv2.waitKey(0)