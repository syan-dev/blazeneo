import os
import cv2
import numpy as np

from libs.common import load_engine, allocate_buffers, TRT_LOGGER, do_inference

import pycuda.driver as cuda
import pycuda.autoinit

engine_file_path = '/home/vdsense/Documents/data/models/unet_b4_320x320_040520_120520_140520_180520_240520_260520_280520_040620_int8_b1_w1.trt'
engine = load_engine(engine_file_path)
inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

image = cv2.imread('/home/vdsense/Documents/data/images/medium_part/BLI/BLI_191102_BN029_029.JPG')
image = cv2.resize(image, (320, 320))
cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cvt_image = cvt_image.astype(np.float32)
cvt_image = cvt_image/127.5-1

# Both inputs and outputs are numpy array
np.copyto(inputs[0].host, cvt_image.ravel())
# inputs[0].host = cvt_image
[output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)

pr_mask = (output)*255
# convert mask from 1 channel to 3 channel
pr_mask = np.uint8(pr_mask)
pr_mask = np.reshape(pr_mask, (320, 320))
pr_mask = np.stack((pr_mask,)*3, axis=-1)
# print(pr_mask.shape)

cv2.imshow('Image', image)
cv2.imshow('mask', pr_mask)
cv2.waitKey(0)

