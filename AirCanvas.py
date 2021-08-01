import cv2
import os
import numpy as np
import HandTrackingModule as htm


folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.5)

while True:

    # 1. Import images
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList)

        # Tip of Index and Middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Check which fingers are up
    # 4. Selection Mode : 2 fingers are up
    # 5. Drawing Mode : Index finger is up

    # Setting the overlay
    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)
