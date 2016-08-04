
import numpy as np
import cv2
from matplotlib import pyplot as plt

iCcnt = 0

def substractImg(imgSrc, imgDst):
    imgSrc2=imgSrc.copy()
    imgDst2=imgDst.copy()
    #imgSrc2 = cv2.Canny(imgSrc, 30, 255)
    #imgDst2 = cv2.Canny(imgDst, 30, 255)
    #imgCmp = abs(imgSrc2-imgDst2)
    imgCmp = cv2.subtract(imgSrc2, imgDst2)
    return imgCmp

def cmpArea(imgSrc, imgDst, iccnt):
    hSrc, wSrc, sSrc = imgSrc.shape
    hDst, wDst, sDst = imgDst.shape
    topSrc=0
    botSrc=hSrc
    leftSrc=0
    rightSrc=wSrc

    topDst=0
    botDst=hDst
    leftDst=0
    rightDst=wDst

    if iccnt == 0:
        if wSrc > wDst:
            leftSrc=abs(wSrc-wDst)
        else:
            leftCmp=abs(wSrc-wDst)
        if hSrc > hDst:
            topSrc=abs(hSrc-hDst)
        else:
            topCmp=abs(hSrc-hDst)

    elif iccnt == 1:
        if wSrc > wDst:
            rightSrc=wDst
        else:
            rightCmp=wSrc
        if hSrc > hDst:
            topSrc=abs(hSrc-hDst)
        else:
            topCmp=abs(hSrc-hDst)

    elif iccnt == 2:
        if wSrc > wDst:
            leftSrc=abs(wSrc-wDst)
        else:
            leftCmp=abs(wSrc-wDst)
        if hSrc > hDst:
            botSrc=hDst
        else:
            botCmp=hSrc
    else:
        if wSrc > wDst:
            rightSrc=wDst
        else:
            rightCmp=wSrc

        if hSrc > hDst:
            botSrc=hDst
        else:
            botCmp=hSrc

    print "imgSrc top=", topSrc, "bot=", botSrc, "l=",leftSrc, "r=",rightSrc
    print "imgCmp top=", topDst, "bot=", botDst, "l=",leftDst, "r=",rightDst

    imgSrc1=imgSrc[topSrc:botSrc, leftSrc:rightSrc]
    imgDst1 = imgDst[topDst:botDst, leftDst:rightDst]

    #imgSrc1=imgSrc[0:5, 0:5]
    #imgCmp1 = imgCmp[0:5, 0:5]

    hS2, wS2, sS2 = imgSrc1.shape
    hC2, wC2, sC2 = imgDst1.shape

    imgSrcB, imgSrcG, imgSrcR = cv2.split(imgSrc) #cv2.cvtColor(imgSrc1,cv2.COLOR_BGR2GRAY)
    imgDstB, imgDstG, imgDstR = cv2.split(imgDst) #cv2.cvtColor(imgCmp1,cv2.COLOR_BGR2GRAY)

    #cv2.imwrite("imgSrc2.jpg", imgSrc2)
    #cv2.imwrite("imgCmp2.jpg", imgCmp2)

    #cv2.absdiff(imgSrc, imgCmp, imgSub)

    imgCmpB = substractImg(imgSrcB, imgDstB)
    name = "sub" + str(iccnt) + "B.jpg"
    cv2.imwrite(name, imgCmpB)

    imgCmpG = substractImg(imgSrcG, imgDstG)
    name = "sub" + str(iccnt) + "G.jpg"
    cv2.imwrite(name, imgCmpG)

    imgCmpR = substractImg(imgSrcR, imgDstR)
    name = "sub" + str(iccnt) + "R.jpg"
    cv2.imwrite(name, imgCmpR)
    alpha = 0.5

    imgSubs = cv2.addWeighted(imgCmpG, alpha, imgCmpB, 1.0-alpha, 0.0)   #(imgCmpG, imgCmpB) #imgCmpB + imgCmpR
    imgSubs = cv2.addWeighted(imgSubs, alpha, imgCmpR, 1.0-alpha,0.0)  #imgSubs + imgCmpG
    name = "add" + str(iccnt) + ".jpg"
    cv2.imwrite(name, imgSubs)

    #imgAdd = cv2.addWeighted(imgSrc, alpha, imgDst, 1.0-alpha, 0.0)
    #name = "addss" + str(iccnt) + ".jpg"
    #cv2.imwrite(name, imgAdd)

    kernel = np.ones((3,3),np.uint8)
    #imgOpen = cv2.morphologyEx(imgSubs, cv2.MORPH_OPEN, kernel)

    imgOpen = imgSubs.copy()
    imgOpen = cv2.erode(imgOpen, kernel)
    ret, thres = cv2.threshold(imgOpen, 10, 255, cv2.THRESH_BINARY_INV)
    thres1 = cv2.adaptiveThreshold(imgOpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    _, contours, hierarchy = cv2.findContours(thres1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    imgCpy = imgSrc.copy()
    cv2.drawContours(imgCpy,contours, -1,(255,0,255),1)
    #plt.subplot(121), plt.imshow(imgOpen)
    return imgCpy

def cmpArea2(imgSrc, imgDst, iccnt):
    hSrc, wSrc, sSrc = imgSrc.shape
    hDst, wDst, sDst = imgDst.shape
    topSrc=0
    botSrc=hSrc
    leftSrc=0
    rightSrc=wSrc

    topDst=0
    botDst=hDst
    leftDst=0
    rightDst=wDst

    if iccnt == 0:
        if wSrc > wDst:
            leftSrc=abs(wSrc-wDst)
        else:
            leftCmp=abs(wSrc-wDst)
        if hSrc > hDst:
            topSrc=abs(hSrc-hDst)
        else:
            topCmp=abs(hSrc-hDst)

    elif iccnt == 1:
        if wSrc > wDst:
            rightSrc=wDst
        else:
            rightCmp=wSrc
        if hSrc > hDst:
            topSrc=abs(hSrc-hDst)
        else:
            topCmp=abs(hSrc-hDst)

    elif iccnt == 2:
        if wSrc > wDst:
            leftSrc=abs(wSrc-wDst)
        else:
            leftCmp=abs(wSrc-wDst)
        if hSrc > hDst:
            botSrc=hDst
        else:
            botCmp=hSrc
    else:
        if wSrc > wDst:
            rightSrc=wDst
        else:
            rightCmp=wSrc

        if hSrc > hDst:
            botSrc=hDst
        else:
            botCmp=hSrc

    print "imgSrc top=", topSrc, "bot=", botSrc, "l=",leftSrc, "r=",rightSrc
    print "imgCmp top=", topDst, "bot=", botDst, "l=",leftDst, "r=",rightDst

    imgSrc1=imgSrc[topSrc:botSrc, leftSrc:rightSrc]
    imgDst1 = imgDst[topDst:botDst, leftDst:rightDst]

    imgSrc1= cv2.cvtColor(imgSrc,cv2.COLOR_BGR2GRAY)
    imgDst1 = cv2.cvtColor(imgDst,cv2.COLOR_BGR2GRAY)

    imgSubs = substractImg(imgSrc1, imgDst1)
    name = "add" + str(iccnt) + ".jpg"
    cv2.imwrite(name, imgSubs)

    kernel = np.ones((3,3),np.uint8)
    #imgOpen = cv2.morphologyEx(imgSubs, cv2.MORPH_OPEN, kernel)

    imgOpen = imgSubs.copy()
    imgOpen = cv2.erode(imgOpen, kernel)
    ret, thres = cv2.threshold(imgOpen, 5, 255, cv2.THRESH_BINARY_INV)
    thres1 = cv2.adaptiveThreshold(imgOpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #_, contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(thres,contours,1,(255,0,255),3)
    #plt.subplot(121), plt.imshow(imgOpen)
    return thres1, imgOpen

img = cv2.imread('img/3.jpg')

imt = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.blur(img, (3, 3))

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
    #queryX = kp1[queryIdx].pt[0]
    #queryY = kp1[queryIdx].pt[1]
    #trainX = kp2[trainIdx].pt[0]
    #trainY = kp2[trainIdx].pt[1]
    queryX = kp1[queryIdx].pt[0]
    queryY = kp1[queryIdx].pt[1]
    trainX = kp2[trainIdx].pt[0]
    trainY = kp2[trainIdx].pt[1]


matchW = 50
matchH = 50

# Top
hleft, wleft, sleft = imgLeft.shape
hright, wright, sright = imgRight.shape

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

print "queryXI=", queryXI, "queryYI=", queryYI, "trainXI=", trainXI, "trainYI=",trainYI

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

#print "matWFCnt=", matWFCnt,"kp1L=", queryXI, "kp1R=", kp1R, "matHFCnt=", matHFCnt ,"kp1t=", queryYI, "kp1b=",kp1B
#print "matWFCnt=", matWFCnt,"kp2L=", trainXI, "kp2R=", kp2R, "matHFCnt=", matHFCnt ,"kp2t=", trainYI, "kp2b=",kp2B

imgLLeftUp = imgLeft[kp1T:queryYI, kp1L:queryXI]
print "imgLLeftUp, top=", kp1T, "bottom=", queryYI, "left=", kp1L ,"right=", queryXI
cv2.imwrite("imgLLeftUp.jpg", imgLLeftUp)

imgLLeftBottom = imgLeft[queryYI:kp1B, kp1L:queryXI]
print "imgLLeftBottom, top=", queryYI, "bottom=", kp1B, "left=", kp1L ,"right=", queryXI
cv2.imwrite("imgLLeftBottom.jpg", imgLLeftBottom)

imgLRightUp = imgLeft[kp1T:queryYI, queryXI:kp1R]
print "imgLRightUp, top=", kp1T, "bottom=", queryYI, "left=", queryXI ,"right=", kp1R
cv2.imwrite("imgLRightUp.jpg", imgLRightUp)

imgLRightBottom = imgLeft[queryYI:kp1B, queryXI:kp1R]
print "imgLRightBottom, top=", queryYI, "bottom=", kp1B, "left=", queryXI ,"right=", kp1R
cv2.imwrite("imgLRightBottom.jpg", imgLRightBottom)

imgRLeftUp = imgRight[kp2T:trainYI, kp2L:trainXI]
print "imgRLeftUp, top=", kp2T, "bottom=", trainYI, "left=", kp2L ,"right=", trainXI
cv2.imwrite("imgRLeftUp.jpg", imgRLeftUp)

imgRLeftBottom = imgRight[trainYI:kp2B, kp2L:trainXI]
print "imgRLeftBottom, top=", trainYI, "bottom=", kp2B, "left=", kp2L ,"right=", trainXI
cv2.imwrite("imgRLeftBottom.jpg", imgRLeftBottom)

imgRRightUp = imgRight[kp2T:trainYI, trainXI:kp2R]
print "imgRRightUp, top=", kp1T, "bottom=", queryYI, "left=", trainXI ,"right=", kp2R
cv2.imwrite("imgRRightUp.jpg", imgRRightUp)

imgRRightBottom = imgRight[trainYI:kp2B, trainXI:kp2R]
print "imgRRightBottom, top=", trainYI, "bottom=", kp2B, "left=", trainXI ,"right=", kp2R
cv2.imwrite("imgRRightBottom.jpg", imgRRightBottom)

imgOpLUp = cmpArea(imgLLeftUp, imgRLeftUp, 0)
imgOpLBot = cmpArea(imgLLeftBottom, imgRLeftBottom, 1)

imgOpRUp = cmpArea(imgLRightUp, imgRRightUp,2)
imgOpRBot = cmpArea(imgLRightBottom, imgRRightBottom, 3)


#imgOpLUp = cmpArea2(imgLLeftUp, imgRLeftUp, 0)
#imgOpLBot = cmpArea2(imgLLeftBottom, imgRLeftBottom, 1)

#imgOpRUp = cmpArea2(imgLRightUp, imgRRightUp,2)
#imgOpRBot = cmpArea2(imgLRightBottom, imgRRightBottom, 3)

imgHUp = np.concatenate((imgOpLUp, imgOpRUp), axis=1) #cv2.hconcat(imgOpLUp, imgOpRUp)
imgHBot = np.concatenate((imgOpLBot, imgOpRBot), axis=1)  #cv2.hconcat(imgOpLBot, imgOpRBot)
imgVCombine = np.concatenate((imgHUp, imgHBot), axis=0)  #cv2.vconcat(imgHUp, imgHBot)


#plt.imshow(imgCmp)
#plt.show()

#plt.subplot(421), plt.imshow(imgOpLUp)
#plt.subplot(422), plt.imshow(imgOpLBot)
#plt.subplot(423), plt.imshow(imgOpRUp)
#plt.subplot(424), plt.imshow(imgOpRBot)
#plt.subplot(425), plt.imshow(imgLeft)
#plt.subplot(426), plt.imshow(imgRight)
#plt.subplot(427), plt.imshow(imgVCombine)

plt.subplot(131), plt.imshow(imgLeft)
plt.subplot(132), plt.imshow(imgRight)
plt.subplot(133), plt.imshow(imgVCombine)
plt.show()