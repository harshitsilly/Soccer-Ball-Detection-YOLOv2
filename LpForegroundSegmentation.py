import cv2

image=cv2.imread('scImages\lpYoloLitenoweights\\38.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("image",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
