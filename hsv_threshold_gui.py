# Preview: https://raw.githubusercontent.com/FedericoPonzi/LegoLab/master/media/hsv-colour.png
import time
import cv2
import sys

import numpy as np

def nothing(x):
    pass

useCamera=False

# Check if filename is passed
if (len(sys.argv) <= 1) :
    print ("'Usage: python hsvThresholder.py <ImageFilePath>' to ignore camera and use a local image.")
    useCamera = True

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,255,nothing) # Hue is from 0-179 for Opencv, not 255
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,255,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 255)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Output Image to display
if useCamera:
    # cap = cv2.VideoCapture(0)
    from relod.envs.create2_visual_reacher.depstech_camera_communicator import CameraCommunicator
    comm = CameraCommunicator(res=(160, 120))
    comm.start()
    waitTime = 330
else:
    img = cv2.imread(sys.argv[1])
    output = img
    waitTime = 33

while(1):

    if useCamera:
        # Capture frame-by-frame
        # ret, img = cap.read()
        img, _, _ = comm.get_image()
        # print(img.shape(), type(img))
        img = img[0].reshape(120, 160, 3)

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(blackAndWhiteImage, pts=contours, color=(255, 255, 255))
    target_size = np.sum(blackAndWhiteImage/255.) / blackAndWhiteImage.size
    #print('target size:', target_size)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display output image
    cv2.imshow('image', output)
    print(np.sum(blackAndWhiteImage/255.)/float(blackAndWhiteImage.size), np.sum(blackAndWhiteImage/255.),
          float(blackAndWhiteImage.size), target_size)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

# Release resources
if useCamera:
    # cap.release()
    pass
cv2.destroyAllWindows()
