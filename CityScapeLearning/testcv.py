import cv2 as cv2

print(cv2.__version__)

I = cv2.imread('D:/EtienneData/valless/_0_im_.png')
cv2.imshow(mat = I,winname = "fenetre")
cv2.waitKey()
