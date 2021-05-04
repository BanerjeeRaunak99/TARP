
import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
c = 0
upper_left = (0, 0)
bottom_right = (300, 300)
 
while True:
    # Getting out image by webcam 
    _, image = cap.read()
    image = cv2.flip(image,1)
    r = cv2.rectangle(image, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = image[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    sketcher_rect = rect_img
    hsvim = cv2.cvtColor(sketcher_rect, cv2.COLOR_BGR2HSV)
    lower = np.array([0,20,70], dtype = "uint8")
    upper = np.array([20,255,255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(image, [contours], -1, (255,255,0), 2)
    
    # cv2.imshow("contours", image)
    hull = cv2.convexHull(contours)
    

    cv2.drawContours(image, [hull], -1, (0, 255, 255), 2)
    # cv2.imshow("hull", image)  
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)
    
    cnt = 0
    for i in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57 #      cosine theorem
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
        d=(2*ar)/a
        if angle <= 90 and d>30:  # angle less than 90 degree, treat as fingers
            cnt += 1
            cv2.circle(image, far, 4, [0, 0, 255], -1)
        
    cnt+=1
    cv2.putText(image, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
    cv2.imshow('final_result',image)
    # Show the image
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
