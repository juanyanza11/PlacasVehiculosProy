import cv2

cap = cv2.VideoCapture("vid2.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (640,480))
roiSelected = cv2.selectROI(frame)

cv2.destroyAllWindows()


while (cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    roi_cropped = frame[int(roiSelected[1]):int(roiSelected[1]+roiSelected[3]), int(roiSelected[0]):int(roiSelected[0]+roiSelected[2])]

    #cv2.imshow('webCam',frame)
    cv2.imshow('webCamROI',roi_cropped)
    if (cv2.waitKey(1) == ord('s')):
        break

cap.release()
cv2.destroyAllWindows()