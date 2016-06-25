
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('41.jpg')

imt = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w, r = img.shape
print 'w, h, r', w, h,r

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img_gray,None)

gridW = 20
gridH = 20

cntKP = 0

for eachKP in kp:
    if eachKP.pt[1] - gridH > 0 and eachKP.pt[0] - gridW > 0:
        if eachKP.pt[1] + gridH < h and eachKP.pt[1] + gridH > h/5 and eachKP.pt[0] + gridW < w/2:
            sigKP = eachKP
            print 'sigKP=', sigKP.pt[0], sigKP.pt[1]
            cntKP += 1
            minLeftX = sigKP.pt[0]
            minLeftY = sigKP.pt[1]
            imgsub = img[minLeftY-gridH:minLeftY+gridH, minLeftX - gridW:minLeftX+gridW]
            res = cv2.matchTemplate(img,imgsub,cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where( res >= threshold)
            minRighX = w
            minRighY = h
            for pt in zip(*loc[::-1]):
                cv2.rectangle(imt, pt, (pt[0] + gridW, pt[1] + gridH), (0,0,255), 2)
                print "find point", pt[0] + gridW, pt[1] + gridH
                if abs(pt[1] + gridH - minLeftY) < 3:
                    if pt[0] < minRighX and pt[0] > minLeftX:
                        minRighY = pt[1]+gridW
                        minRighX = pt[0]+gridH
                        print 'minLeftX=', minLeftX, 'minLeftY=', minLeftY, 'minRightX=', minRighX, 'minRightY=', minRighY
                        cntKP = 5
                        break

            if cntKP == 5:
                break

if minRighX < 999:
    singleW = (minRighX - minLeftX)
    imgLeft = img[0:h, 0:singleW]
    imgRight = img[0:h, w-singleW:w]
else:
    singleW = w / 2 - gridW
    minRighX = w/2
    minLeftX = 0
    imgLeft = img[0:h, 0:singleW]
    imgRight = img[0:h, w-singleW:w]

print "w=",w, "singleW=",singleW
print 'minLeftX=', minLeftX, 'minLeftY=', minLeftY, 'minRightX=', minRighX, 'minRightY=', minRighY

singleH = h

#imgsub = img[sigKP.pt[0] - gridW:sigKP.pt[0]+gridW, sigKP.pt[1] - gridH:sigKP.pt[1]+gridH]

# compute the descriptors with ORB
#kp, des = orb.compute(img, kp)


# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(img, kp, imt)
#plt.imshow(img2),plt.show()

#plt.imshow(img)



#plt.subplot(121), plt.imshow(imgLeft)
#plt.subplot(122),plt.imshow(imgRight)
#plt.show()


cv2.imwrite('cutLeft.jpg', imgLeft)
cv2.imwrite('cutRight.jpg', imgRight)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(imgLeft, None)
kp2, des2 = orb.detectAndCompute(imgRight, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
imgCmp = cv2.drawMatches(imgLeft, kp1, imgRight, kp2,matches[:5], None, flags=2)

print matches[:5]

trainX = 0
trainY = 0

queryX = 0
queryY = 0

for mt in matches[:5]:
    print mt.queryIdx, mt.trainIdx
    queryIdx = mt.queryIdx
    trainIdx = mt.trainIdx
    print "kp1 ", kp1[queryIdx].pt[0], kp1[queryIdx].pt[1]
    print "kp2 ", kp2[trainIdx].pt[0], kp2[trainIdx].pt[1]
    queryX = kp1[queryIdx].pt[0]
    queryY = kp1[queryIdx].pt[1]
    trainX = kp2[trainIdx].pt[0]
    trainY = kp2[trainIdx].pt[1]

plt.imshow(imgCmp)
plt.show()

matchW = 50
matchH = 50

# Top
matCnt = 1
kp1L = queryX
kp1R = queryX
kp1T = queryY
kp1B = queryY

kp2L = trainX
kp2R = trainX
kp2T = trainY
kp2B = trainY

while kp1L > 0 and kp1T > 0:
    if kp2L > 0 and kp2T > 0:
        if kp1B < h and kp2B < h:
            if kp1R < singleW and kp2R < singleW:

                kp1L = kp1L - matchW
                kp1R = kp1R - matchW

            else:
                break
        else:
            break
    else:
        break



# Bottom
matCnt = 1
kp1L = queryX
kp1R = queryX
kp1T = queryY
kp1B = queryY

kp2L = trainX
kp2R = trainX
kp2T = trainY
kp2B = trainY

