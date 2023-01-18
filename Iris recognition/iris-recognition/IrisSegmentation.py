import cv2
import math
import numpy as np

def IrisSeg(imgSrc, minDist = 100, param1 = 30, param2 = 50, maxRadius1 = 200):
    apret = 5
    
    img = cv2.imread(imgSrc)
    scaling_factor = 0.7
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    blur = cv2.GaussianBlur(dst,(5,5),0)
    inv = cv2.bitwise_not(blur)
    thresh = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    ret,thresh1 = cv2.threshold(erosion,220,255,cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    fx = 0
    fy = 0

    fradius = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        x, y, width, height = rect
        radius = 0.25 * (width + height)
        (xr,yr),radius = cv2.minEnclosingCircle(contour)
        area_condition = (200 <= area <= 1700)

        symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.5)

        fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= .51)

        if area_condition and symmetry_condition and fill_condition:
            fy = yr
            
            fx = xr
            fradius = radius
            
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (int(fx), int(fy)), int(fradius), (255,255,255), -1)
    mask = cv2.bitwise_not(mask)
    img = cv2.bitwise_and(img, mask)

    img = img[int(fy - 200):  int(fy + 45), int(fx - 265):int(fx + 45)]
    
    img = cv2.resize(img, (100,100))
    imag = cv2.GaussianBlur(img, (7, 7), 1)
    imag = cv2.Canny(imag, 20, 70, apertureSize=apret)

    return imag, img
