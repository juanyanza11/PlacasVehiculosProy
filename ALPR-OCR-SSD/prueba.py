import cv2
import os
import time
RTSP_URL = 'rtsp://movilidad:movi_4567@200.0.29.199:554/Streaming/Channels/102'
 
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
 
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
 
if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)
 
fps_start_time = 0

while True:
    _, frame = cap.read()
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/time_diff
    fps_start_time = fps_end_time
    fps_txt = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_txt, (5, 30), cv2.FONT_ITALIC, 1, (255 , 255, 255), 1)
    cv2.imshow('RTSP stream', frame)
 
    if cv2.waitKey(1) == 27:
        break
 
cap.release()
cv2.destroyAllWindows()