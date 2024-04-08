import cv2
import numpy as np

# Attach camera indexed as 0
camera = cv2.VideoCapture(0)

# Setting frame width and frame height as 640 x 480
camera.set(3, 640)
camera.set(4, 480)

# Loading the mountain image
mountain = cv2.imread('mount_everest.jpg')

# Resizing the mountain image as 640 x 480
mountain = cv2.resize(mountain, (640, 480))

while True:
    # Read a frame from the attached camera
    status, frame = camera.read()

    # If we got the frame successfully
    if status:
        # Flip it
        frame = cv2.flip(frame, 1)

        # Converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Thresholding the frame to get binary image
        _, thresh = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

        # Inverting the binary mask
        thresh_inv = cv2.bitwise_not(thresh)

        # Bitwise and operation to extract foreground / person
        foreground = cv2.bitwise_and(frame, frame, mask=thresh_inv)

        # Combine foreground and mountain image
        result = cv2.addWeighted(foreground, 1, mountain, 0.5, 0)

        # Show the result
        cv2.imshow('Result', result)

        # Wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:  # Press spacebar to exit
            break

# Release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
