import cv2 as cv

cap = cv.VideoCapture(0)

index = 0

while True:
    ret, frame = cap.read()

    cv.imwrite(f'./assets/images/originals/camera_image{index}.jpg', frame)
    index += 1

    if not ret:
        break

    if cv.waitKey(1) & 0xFF == ord('q'):
        break