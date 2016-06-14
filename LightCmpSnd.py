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

print 'w=', w, 'h=', h

blu1 = cv2.GaussianBlur(img1, (3, 3), 0)
blu2 = cv2.GaussianBlur(img3, (3, 3), 0)

r1, g1, b1 = cv2.split(img1)
r2, g2, b2 = cv2.split(img2)



hsv1 = cv2.cvtColor(blu1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(blu2, cv2.COLOR_BGR2HSV)

h1, s1, v1 = cv2.split(hsv1)
h2, s2, v2 = cv2.split(hsv2)


sizeW = 10
sizeH = sizeW
thresholdValue = 10
ww = w / sizeW
hh = h / sizeW

rw, gw, bw = img1[42, 7]
print rw, gw, bw

rn, gn, bn = img3[42, 7]
print rn, gn, bn


h3 = cv2.subtract(h1, h2)

s3 = cv2.subtract(s1, s2)

v3 = cv2.subtract(v1, v2)

cv2.imshow('subImghue', h3)
cv2.imshow('subImgIns', s3)
cv2.imshow('subImgver', v3)


img3 = cv2.subtract(img1, img2)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)


imgGray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

thr1 = cv2.adaptiveThreshold(imgGray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,11,2)

retr1,threshImg = cv2.threshold(imgGray, 30, 255, cv2.THRESH_BINARY)

cv2.imshow('subImg', imgGray)
cv2.imshow('threshImg', threshImg)


cv2.waitKey(0)
cv2.destroyAllWindows()