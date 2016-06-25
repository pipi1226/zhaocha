
import numpy as np
import cv2
from matplotlib import pyplot as plt

iCcnt = 0

def cmpArea(imgSrc, imgCmp, iccnt):
    srcW, srcH, srcS = imgSrc.shape
    cmpW, cmpH, cmpS = imgCmp.shape
    mW = min(srcW, cmpW)
    mH = min(srcH, cmpH)

    if iccnt == 0:
        imgSrcSub = imgSrc[srcH-mH:srcH, srcW - mW:srcW]
        imgCmpSub = imgCmp[cmpH-mH:cmpH, cmpW - mW:cmpW]
    elif iccnt == 1:
        imgSrcSub = imgSrc[0:mH, srcW - mW:srcW]
        imgCmpSub = imgCmp[0:mH, cmpW - mW:cmpW]
    elif iccnt == 2:
        imgSrcSub = imgSrc[srcH-mH:srcH, 0:mW]
        imgCmpSub = imgCmp[cmpH-mH:cmpH, 0:mW]
    else:
        imgSrcSub = imgSrc[0:mH, 0:mW]
        imgCmpSub = imgCmp[0:mH, 0:mW]

    strName = "imgSrc" + str(iccnt)+ ".jpg"
    cv2.imwrite(strName, imgSrcSub)
    strName = "imgCmp" + str(iccnt)+ ".jpg"
    cv2.imwrite(strName, imgCmpSub)
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

queryXI = int(queryX)
queryYI = int(queryY)

trainXI = int(trainX)
trainYI = int(trainY)

iHFlag = 0
if queryYI < trainYI:
    iHFlag = 1
elif queryYI > trainYI:
    iHFlag = 2

kpTS = abs(queryYI - trainYI)

iWFlag = 0
if queryXI < trainXI:
    iWFlag = 1
elif queryXI > trainXI:
    iWFlag =2

kpLS = abs(queryXI - trainXI)

print "matWFCnt = ", matWFCnt, "matHFCnt=", matHFCnt
print "matWFCnt=", matWFCnt,"kp1L=", queryXI, "kp1R=", kp1R, "matHFCnt=", matHFCnt ,"kp1t=", queryYI, "kp1b=",kp1B
print "matWFCnt=", matWFCnt,"kp2L=", trainXI, "kp1R=", kp2R, "matHFCnt=", matHFCnt ,"kp2t=", trainYI, "kp2b=",kp2B

kp1L = 0
kp1R = widthMin
kp1T = 0
kp1B = heightMin

kp2L = 0
kp2R = widthMin
kp2T = 0
kp2B = heightMin


if iWFlag == 1: # queryXI < trainXI:
    kp2L = kpLS
    kp1R = widthMin - kpLS
    if iHFlag == 1: # queryYI < trainYI:
        kp2T = kpTS
        kp1B = heightMin - kpTS
    elif iHFlag ==2:    # queryYI > trainYI:
        kp1T = kpTS
        kp2B = heightMin - kpTS
elif iWFlag == 2:   # queryXI > trainXI:
    kp1L = kpLS
    kp2R = widthMin - kpLS
    if iHFlag == 1: # queryYI < trainYI:
        kp2T = kpTS
        kp1B = heightMin - kpTS
    elif iHFlag ==2:    # queryYI > trainYI:
        kp1T = kpTS
        kp2B = heightMin - kpTS

else: # iwflag = 0
    if iHFlag == 1: # queryYI < trainYI:
        kp2T = kpTS
        kp1B = heightMin - kpTS
    elif iHFlag == 2: # queryYI > trainYI:
        kp1T = kpTS
        kp2B = heightMin - kpTS



imgLLeftUp = imgLeft[kp1T:queryYI, kp1L:queryXI]
imgLLeftBottom = imgLeft[queryYI:kp1B, kp1L:queryXI]

imgLRightUp = imgLeft[kp1T:queryYI, queryXI:kp1R]
imgLRightBottom = imgLeft[queryYI:kp1B, queryXI:kp1R]

imgRLeftUp = imgRight[kp2T:trainYI, kp2L:trainXI]
imgRLeftBottom = imgRight[trainYI:kp2B, kp2L:trainXI]

imgRRightUp = imgRight[kp2T:trainYI, trainXI:kp2R]
imgRRightBottom = imgRight[trainYI:kp2B, trainXI:kp2R]

cmpArea(imgLLeftUp, imgRLeftUp, 0)
cmpArea(imgLLeftBottom, imgRLeftBottom, 1)

cmpArea(imgLRightUp, imgRRightUp,2)
cmpArea(imgLRightBottom, imgRRightBottom, 3)


plt.imshow(imgCmp)
plt.show()