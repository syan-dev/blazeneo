import os
import cv2
import numpy as np

from libs.utilities import scan_dir, create_dir
from libs.common import load_engine, allocate_buffers, TRT_LOGGER, do_inference
from libs.fps import FPS
import time

engine_file_path = '/home/vdsense/Documents/data/models/unet_b4_320x320_040520_120520_140520_180520_240520_260520_280520_040620_fp16_b1_w1.trt'
video_dir = '/media/vdsense/LOC/video_polyp'
saved_dir = '/media/vdsense/LOC/tmp/pred_video_polyp_fp32'

# create_dir(saved_dir)

# video_fps = scan_dir(video_dir)
# print(video_fps)

import pycuda.driver as cuda
import pycuda.autoinit

engine = load_engine(engine_file_path)
inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

cv2.namedWindow('image', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

video_fps = [0]
for video_path in video_fps:
    print(video_path)
    # fn = video_path.split('/')[-1]
    # print(">>>>> process video: ", fn)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    # out = cv2.VideoWriter(os.path.join(saved_dir, fn), cv2.VideoWriter_fourcc(*'VP80'), 60, (320*2, 320))

    fps = FPS().start()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        
        h, w, c = frame.shape
        # image = frame[:, :int(w*0.7), :]

        image = cv2.resize(frame, (320, 320))
        cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cvt_image = cvt_image.astype(np.float32)
        cvt_image = cvt_image/127.5-1

        # Both inputs and outputs are numpy array
        np.copyto(inputs[0].host, cvt_image.ravel())
        # inputs[0].host = cvt_image
        [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        # convert mask from 1 channel to 3 channel
        output[output<0.5] = 0
        mask = (output)*255
        mask = np.reshape(mask, (320, 320))
        mask = np.uint8(mask)
        # pr_mask = np.stack((pr_mask,)*3, axis=-1)

        # tmp = np.zeros([320, 320*2, 3],dtype=np.uint8)
        # tmp[:, :320, :] = image
        # tmp[:, 320:, :] = pr_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
        erode_mask = cv2.erode(mask, kernel, iterations=2)
        dilate_mask = cv2.dilate(erode_mask, kernel, iterations=1)
        dilate_mask[erode_mask>0] = 0

        mask = cv2.resize(dilate_mask, (w, h))
        frame[mask>0] = [0, 255, 0]

        # cv2.imshow('test', tmp)
        cv2.imshow('image', frame)
        # cv2.imshow('mask', dilate_mask)
        fps.update()
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        # out.write(tmp)

    fps.stop()

    cap.release()
    # out.release()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

