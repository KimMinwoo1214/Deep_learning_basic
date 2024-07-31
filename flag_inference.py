import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 비디오 캡처 설정
cap = cv2.VideoCapture(0)

# 클래스 라벨 설정 및 모델 로드
class_labels = ["france", "ghana", "northkorea"]
model = load_model("vgg_flag.h5")

while True:
    ret, img = cap.read()
    if ret:
        img_scaled = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        data = img_scaled.astype("float") / 255.0
        X = np.asarray([data])

        # 모델 예측
        s = model(X, training=False)
        index = np.argmax(s)
        strr = class_labels[index]
        print(strr)

        # 텍스트 표시
        cv2.putText(img, strr, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # 국기 인식 부분 추출을 위한 마스크 생성
        mask = np.zeros(img.shape[:2], dtype="uint8")
        img_scaled_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        _, binary_scaled = cv2.threshold(img_scaled_gray, 127, 255, cv2.THRESH_BINARY)

        # 마스크를 원본 이미지 크기로 리사이즈
        mask_resized = cv2.resize(binary_scaled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

        # 윤곽선 검출
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 윤곽선 그리기
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        # 결과 이미지 표시
        cv2.imshow('win', img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

# 모든 창 닫기 및 비디오 캡처 해제
cv2.destroyAllWindows()
cap.release()
