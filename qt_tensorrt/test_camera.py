import numpy as np
import cv2
from libs.fps import FPS

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

window_name = 'frame'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

fps = FPS().start()
while(True):
    if not cap.isOpened():
        print("camera has not been connected")
        break
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(frame.shape)
    frame = cv2.UMat(frame)

    cv2.imshow(window_name,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
