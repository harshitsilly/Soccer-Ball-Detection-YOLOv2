import numpy as np
import cv2


def filter_colors(image):
    """
    Filter the image to include only yellow and white pixels
    """
    # Filter white pixels
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

    return image2

image=cv2.imread('scImages/lpYoloTinysc1/133.jpg',0)
# image = filter_colors(image)
cv2.imshow('image1',image)
cv2.imwrite('test.jpg',image)
edges = cv2.Canny(image,50,150)
# lines = cv2.HoughLines(edges,1,np.pi/180,100) # used to be 200
# for rho,theta in lines[0]:   # displays only the first line 
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

# cv2.line(image,(x1,y1),(x2,y2),(255,0,255),1)
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()