import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('h1.jpg', cv2.IMREAD_COLOR)

img = img[0:20, 0:20]

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

img2 = cv2.imread('h2.jpg', cv2.IMREAD_COLOR)
img2 = img2[0:20, 0:20]

hist2,bins2 = np.histogram(img2.flatten(),256,[0,256])

cdf2 = hist2.cumsum()
cdf_normalized2 = cdf2 * hist2.max()/ cdf2.max()

print 'end'
plt.subplot(211), plt.plot(cdf_normalized, color = 'b'), plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

plt.subplot(212), plt.plot(cdf_normalized2, color = 'b'), plt.hist(img2.flatten(),256,[0,256], color = 'r')

#plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()