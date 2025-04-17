import cv2 as cv
from PIL.ImageOps import grayscale

original = cv.imread("assets/test.jpg")

cv.imshow("Original", original)
cv.waitKey(0)

grayscale_image = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

faces = face_cascade.detectMultiScale(grayscale_image)

for (x, y, w, h) in faces:
    cv.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv.imshow("Faces", original)
cv.waitKey(0)