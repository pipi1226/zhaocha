
import numpy as np
import cv2
from matplotlib import pyplot as plt

def cmpArea(imgSrc, imgCmp):

    return



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

#plt.imshow(imgCmp)
#plt.show()

matchW = 50
matchH = 50

# Top
wleft, hleft, sleft = imgLeft.shape
wright, hright, sright = imgRight.shape

widthMin = min(wleft, wright)
heightMin = min(hleft, hright)

matWCnt = widthMin / matchW
matLeftWFCnt = queryX / matchW
matLeftWBCnt = (widthMin - queryX) / matchW

matRightWFCnt = trainX / matchW
matRightWBCnt = (widthMin - trainX) / matchW

matWFCnt = int(min(matLeftWFCnt, matRightWFCnt))
matWBCnt = int(min(matLeftWBCnt, matRightWBCnt))

matHCnt = heightMin / matchH
matLeftHFCnt = queryY / matchH
matLeftHBCnt = (heightMin - queryY) / matchH

matRightHFCnt = trainY / matchH
matRightHBCnt = (heightMin - trainY) / matchH

matHFCnt = int(min(matLeftHFCnt, matRightHFCnt))
matHBCnt = int(min(matLeftHBCnt, matRightHBCnt))

kp1L = int(queryX)
kp1R = int(queryX)
kp1T = int(queryY)
kp1B = int(queryY)

kp2L = int(trainX)
kp2R = int(trainX)
kp2T = int(trainY)
kp2B = int(trainY)

print "matWFCnt = ", matWFCnt, "matHFCnt=", matHFCnt
print "matWFCnt=", matWFCnt,"kp1L=", kp1L, "kp1R=", kp1R, "matHFCnt=", matHFCnt ,"kp1t=", kp1T, "kp1b=",kp1B

# left top
bWFlag = 0
bHFlag = 0
for iWFCnt in range(1, matWFCnt):
    bWFlag = 0
    kp1R = kp1L
    kp1L = kp1L - matchW

    kp2R = kp2L
    kp2L = kp2L - matchW

    if kp1L < 0 or kp2L < 0:
        bWFlag = 1
        break
    kp1T = int(queryY)
    for iHFCnt in range(1, matHFCnt):
        bHFlag = 0
        kp1B = kp1T
        kp1T = kp1T - matchH

        kp2B = kp2T
        kp2T = kp2T - matchH

        if kp1T < 0 or kp2T < 0:
            bHFlag = 1
            break
        print "iWFCnt=", iWFCnt,"kp1L=", kp1L, "kp1R=", kp1R, "iHFCnt=", iHFCnt ,"kp1t=", kp1T, "kp1b=",kp1B
        imgLeftCut = imgLeft[kp1T:kp1B, kp1L:kp1R]
        imgRightCut = imgRight[kp2T:kp2B, kp1L:kp2R]
        cv2.imwrite('cutLeftSub.jpg', imgLeftCut)
        cv2.imwrite('cutRightSub.jpg', imgRightCut)

    if kp2T < 0 and kp2B > 0:
        kp2T = 0

    if kp1T < 0 and kp1B > 0:
        kp1T = 0

    if bHFlag == 1:
        print "bflag == 1 ======kp1L=", kp1L, "kp1R=", kp1R, "kp1t=", kp1T, "kp1b=",kp1B
        imgLeftCut = imgLeft[kp1T:kp1B, kp1L:kp1R]
        imgRightCut = imgRight[kp1T:kp1B, kp1L:kp1R]
        cv2.imwrite('cutLeftSub.jpg', imgLeftCut)
        cv2.imwrite('cutRightSub.jpg', imgRightCut)

if bWFlag == 1:
    if kp1L < 0 and kp1R > 0:
        if kp2L < 0 and kp2R > 0:
            kp1L = 0
            kp1B = 0
            kp1T = int(queryY)

            kp2L = 0
            kp2B = 0
            kp2T = int(trainY)
            for iHFCnt in range(1, matHFCnt):
                bHFlag = 0
                kp1B = kp1T
                kp1T = kp1T - matchH

                kp2B = kp2T
                kp2T = kp2T - matchH

                if kp1T < 0 or kp2T < 0:
                    bHFlag = 1
                    break
                #print "kp1L=", kp1L, "kp1R=", kp1R, "kp1t=", kp1T, "kp1b=",kp1B
                print "kp1L=", kp1L, "kp1R=", kp1R, "iHFCnt=", iHFCnt ,"kp1t=", kp1T, "kp1b=",kp1B
                imgLeftCut = imgLeft[kp1T:kp1B, kp1L:kp1R]
                imgRightCut = imgRight[kp1T:kp1B, kp1L:kp1R]
                cv2.imwrite('cutLeftSub.jpg', imgLeftCut)
                cv2.imwrite('cutRightSub.jpg', imgRightCut)
        elif kp2L > 0:
            for iHFCnt in range(1, matHFCnt):
                bHFlag = 0
                kp1B = kp1T
                kp1T = kp1T - matchH

                kp2B = kp2T
                kp2T = kp2T - matchH

                if kp1T < 0 or kp2T < 0:
                    bHFlag = 1
                    break
                #print "kp1L=", kp1L, "kp1R=", kp1R, "kp1t=", kp1T, "kp1b=",kp1B
                print "kp1L=", kp1L, "kp1R=", kp1R, "iHFCnt=", iHFCnt ,"kp1t=", kp1T, "kp1b=",kp1B
                imgLeftCut = imgLeft[kp1T:kp1B, kp1L:kp1R]
                imgRightCut = imgRight[kp1T:kp1B, kp1L:kp1R]
                cv2.imwrite('cutLeftSub.jpg', imgLeftCut)
                cv2.imwrite('cutRightSub.jpg', imgRightCut)

