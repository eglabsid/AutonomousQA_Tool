import cv2
import dlib

import numpy as np
import mss
import mss.tools

# Load the template image
template = cv2.imread('target/b1.jpg', cv2.IMREAD_GRAYSCALE)
template_w, template_h = template.shape[::-1]

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the tracker
tracker = dlib.correlation_tracker()

# Flag to indicate if we are currently tracking an object
tracking = False

with mss.mss() as sct:
    monitor = sct.monitors[1]                
    while True:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if tracking:
            # Update the tracker and get the position of the object
            tracker.update(img)
            pos = tracker.get_position()

            # Draw a rectangle around the tracked object
            cv2.rectangle(img, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 2)
        else:
            # Use template matching to find the object in the frame
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Define the bounding box for the detected object
            top_left = max_loc
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

            # Check if the match is good enough
            if max_val > 0.8:  # You can adjust this threshold
                # Start the tracker with the detected object
                tracker.start_track(img, dlib.rectangle(top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
                tracking = True

        # Display the frame
        cv2.imshow('Frame', img)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
