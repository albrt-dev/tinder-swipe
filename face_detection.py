import cv2
import numpy

FACES_XML = 'haarcascade_frontalface_default.xml'
EYES_XML = 'haarcascade_eye.xml'

def __cascade(file):
    return cv2.CascadeClassifier(file)

def __imfaces(image):
    face_cascade = __cascade(FACES_XML)
    return face_cascade.detectMultiScale(image)

def __imeyes(image):
    eye_cascade = __cascade(EYES_XML)
    return eye_cascade.detectMultiScale(image)

def __imgray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def __imroi(image, x, y, w, h):
    return image[y: y + h, x: x + w]

def imread(file):
    return cv2.imread(file)

def parser(image):
    result = []
    faces = __imfaces(__imgray(image))
    for (x, y, w, h) in faces:
        roi = __imroi(image, x, y, w, h)
        eyes = __imeyes(__imgray(roi))
        if len(eyes) >= 2:
            result.append(roi)
    return result

def parser_gray(image):
    return parser(imgray(image))
