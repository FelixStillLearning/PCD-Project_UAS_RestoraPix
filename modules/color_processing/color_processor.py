import cv2
import numpy as np

def color_picker():
    """
    A function to detect colors and display the HSV values using trackbars.
    Press ESC to exit.
    """
    def nothing(x):
        pass
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Trackbar')
    
    if not cam.isOpened():
        print("Error: Camera not found.")
        return

    cv2.createTrackbar('L-H', 'Trackbar', 0, 179, nothing)
    cv2.createTrackbar('L-S', 'Trackbar', 0, 255, nothing)
    cv2.createTrackbar('L-V', 'Trackbar', 0, 255, nothing)
    cv2.createTrackbar('U-H', 'Trackbar', 179, 179, nothing)
    cv2.createTrackbar('U-S', 'Trackbar', 255, 255, nothing)
    cv2.createTrackbar('U-V', 'Trackbar', 255, 255, nothing)

    while True:
        _, frame = cam.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos('L-H', 'Trackbar')
        l_s = cv2.getTrackbarPos('L-S', 'Trackbar')
        l_v = cv2.getTrackbarPos('L-V', 'Trackbar')
        u_h = cv2.getTrackbarPos('U-H', 'Trackbar')
        u_s = cv2.getTrackbarPos('U-S', 'Trackbar')
        u_v = cv2.getTrackbarPos('U-V', 'Trackbar')

        lower_color = np.array([l_h, l_s, l_v])
        upper_color = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower_color, upper_color)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Display HSV values
        cv2.putText(frame, f"Lower HSV: [{l_h}, {l_s}, {l_v}]", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Upper HSV: [{u_h}, {u_s}, {u_v}]", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cam.release()
    cv2.destroyAllWindows()

def color_tracking():
    """
    A function to track a specific color range using predefined HSV values.
    Press ESC to exit.
    """
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Camera not found.")
        return
        
    while True:
        _, frame = cam.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Default color range (blue-green)
        lower_color = np.array([66, 98, 100])
        upper_color = np.array([156, 232, 255])
        
        mask = cv2.inRange(hsv, lower_color, upper_color)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Display the color range being tracked
        cv2.putText(frame, f"Tracking HSV: [{66}-{156}, {98}-{232}, {100}-{255}]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
    
    cam.release()
    cv2.destroyAllWindows()
