import cv2 as cv2
from matplotlib import pyplot as plt
print(cv2.__version__)

I = cv2.imread('D:/EtienneData/trainmeddisp/_0_disp_.png')
blurred = cv2.GaussianBlur(I,ksize = (3,3), sigmaX = 1)
grad = cv2.Sobel(src = I, ddepth = -1, dx = 1, dy = 1)
cv2.imshow(mat = I,winname = "fenetre")
cv2.imshow(mat= grad, winname = "gradient")
hist = cv2.calcHist(grad,[0], mask = None, histSize = [256], ranges = [0,256])
plt.plot(hist, color = 'b')
plt.xlim([0,10])
plt.show()
cv2.waitKey()

