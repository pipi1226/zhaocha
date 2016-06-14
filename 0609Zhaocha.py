#coding=utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt

#img1 = cv2.imread('h1.jpg', cv2.IMREAD_COLOR)          # queryImage
#img3 = cv2.imread('h2.jpg', cv2.IMREAD_COLOR)          # trainImage

img1 = cv2.imread('311.jpg', cv2.IMREAD_COLOR)          # queryImage
img3 = cv2.imread('322.jpg', cv2.IMREAD_COLOR)          # trainImage

#img1 = cv2.imread('111.jpg', cv2.IMREAD_COLOR)          # queryImage
#img2 = cv2.imread('222.jpg', cv2.IMREAD_COLOR)          # trainImage

w, h, r = img1.shape
img2 = img3[0:w, 0:h]

blu1 = cv2.GaussianBlur(img1, (3, 3), 0)
blu2 = cv2.GaussianBlur(img3, (3, 3), 0)

img3 = cv2.subtract(blu1, blu2)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

# 方格匹配
gridW = 20
gridH = 20



imgGray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

thr1 = cv2.adaptiveThreshold(imgGray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,11,2)

retr1,threshImg = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY)

cv2.imshow('subImg', imgGray)
cv2.imshow('threshImg', threshImg)


cv2.waitKey(0)
cv2.destroyAllWindows()