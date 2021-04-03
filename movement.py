from imutils import face_utils
import dlib
import cv2
import pyautogui
 

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
font = cv2.FONT_HERSHEY_SIMPLEX
print(pyautogui.size())
cap = cv2.VideoCapture(0)
c = 0
 
while True:
    # Getting out image by webcam 
    _, image = cap.read()
    image = cv2.flip(image,1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    x_mean=0
    y_mean = 0
   
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        positiontrack = [shape[39],shape[27],shape[42],shape[29]]
        # print(positiontrack)
        for (x,y) in positiontrack:
            x_mean += x
            y_mean += y 
        # print(x_mean,y_mean)
        x_m, y_m = pyautogui.position()
        print(pyautogui.position())
        if(x_mean<1100):
            pyautogui.moveTo(x_m-10,y_m)
        elif(x_mean>1300):
            pyautogui.moveTo(x_m+10,y_m)
        # if(y_mean>1100):
        #     pyautogui.moveTo(x_m,y_m-10)
        # elif(y_mean<950):
        #     pyautogui.moveTo(x_m,y_m+10)

        cv2.circle(image, (x_mean//4, y_mean//4), 10, (0, 255, 0), -1)
        
            
    
    # Show the image
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

