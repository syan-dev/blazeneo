import numpy as np 
import cv2
import sys
import os
import time
import json
from queue import Queue
from threading import Thread, Lock

from .common import load_engine, allocate_buffers, TRT_LOGGER, do_inference
from .fps import FPS



class Config():
    def __init__(self, config_file):
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self.config = json.load(f)
        
        self.source = self.config["source"]
        self.engine_file_path = self.config["engine_file_path"]
        self.width = self.config["width"]
        self.height = self.config["height"]
        self.n_classes = self.config["n_classes"]
        self.threshold = self.config["threshold"]
        self.preprocess_mode = self.config["preprocess_mode"]
        self.postprocess_mode = self.config["postprocess_mode"]
        self.debug = self.config["debug"]
        self.x_start = self.config["crop_coordinates"]["x_start"]
        self.y_start = self.config["crop_coordinates"]["y_start"]
        self.x_end = self.config["crop_coordinates"]["x_end"]
        self.y_end = self.config["crop_coordinates"]["y_end"]

        self.camera_width = self.config["camera_width"]
        self.camera_height = self.config["camera_height"]
        self.camera_fps = self.config["camera_fps"]

        self.x_start = max(min(self.x_start, self.camera_width), 0)
        self.y_start = max(min(self.y_start, self.camera_height), 0)
        self.x_end = max(min(self.x_end, self.camera_width), 0)
        self.y_end = max(min(self.y_end, self.camera_height), 0)
        
        self.input_buffer_size = self.config["input_buffer_size"]
        self.input_sleep_time = self.config["input_sleep_time"]
        self.output_buffer_size = self.config["output_buffer_size"]

        self.lock_stop = Lock()
        self.stopped = False

    def is_stop(self):
        with self.lock_stop:
            return self.stopped

    def stop(self):
        with self.lock_stop:
            self.stopped =True


class Streamming:
    """
    Class that continously get frames from stream with a dedicated thread. 
    """
    def __init__(self, input_queue, config, name="stream video thread"):
        self.input_queue = input_queue

        self.thread = Thread(target=self.update, name=name, args=())
        self.thread.daemon = True

        self.cfg = config
        self.debug = self.cfg.debug


    def update(self):
        capture = cv2.VideoCapture(self.cfg.source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        capture.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)

        while not self.cfg.is_stop():
            start = time.time()
            
            if self.input_queue.full(): 
                self.input_queue.get()

            if not capture.isOpened():
                self.log("Camera has been disconnected")
                self.stop()
                return

            (ret, frame) = capture.read()

            if not ret:
                self.log("Camera has been disconnected")
                self.stop()
                return

            self.input_queue.put(frame, block=False)

            # time.sleep(self.cfg.input_sleep_time)
            end = time.time()
            self.log("Streamming time: {:.4f} s".format(end - start))

        capture.release()


    def log(self, str):
        if self.debug:
            print(str)


    def start(self):
        self.thread.start()


    def stop(self):
        self.cfg.stop()


class Inferencing:
    """
    Create TensorRT model and infer image in seperate thread
    """
    def __init__(self, input_queue, output_queue, preprocessing, postprocessing, config, name="inferencing thread"):
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.thread = Thread(target=self.update, name=name, args=())
        self.thread.daemon = True

        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

        self.cfg = config
        self.debug = self.cfg.debug
        self.name = name


    def update(self):
        import pycuda.driver as cuda

        # Initialize CUDA
        cuda.init()

        from pycuda.tools import make_default_context
        ctx = make_default_context()

        engine = load_engine(self.cfg.engine_file_path)
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        context = engine.create_execution_context()

        while not self.cfg.is_stop():
            start = time.time()

            image = self.input_queue.get(block=True)
            
            crop_image = image[self.cfg.y_start:self.cfg.y_end, self.cfg.x_start:self.cfg.x_end]
            input = self.preprocessing(crop_image)

            # Both inputs and outputs are numpy array
            np.copyto(inputs[0].host, input.ravel())
            [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)

            # output = np.zeros(image.shape, dtype=np.float32)
            
            self.postprocessing(crop_image, output)

            self.output_queue.put(image, block=True)

            end = time.time()
            self.log("Inference time of {}: {:.4f} s".format(self.name, end - start))

        ctx.pop()
        del ctx


    def log(self, str):
        if self.debug:
            print(str)


    def start(self):
        self.thread.start()


    def stop(self):
        self.cfg.stop()



class Displaying:
    """
    Class that continously get frames with output from output queue with a dedicated thread.
    """
    def __init__(self, output_queue, config, name="display video thread"):
        self.output_queue = output_queue

        self.window_name = name

        self.thread = Thread(target=self.update, name=name, args=())
        self.thread.daemon = True

        self.cfg = config
        self.debug = self.cfg.debug


    def update(self):
        self.__setupWindow()
        start_fps = True

        while not self.cfg.is_stop():
            start = time.time()
    
            if self.output_queue.empty():
                continue

            image = self.output_queue.get(block=False)
            
            self.__draw_crop_box(image)

            cv2.imshow(self.window_name, image)
            key = cv2.waitKey(1)

            if start_fps:
                fps = FPS().start()
                start_fps = False
            fps.update()
                
            if key == ord('q') or key == 27:
                self.stop()

            end = time.time()
            self.log("Display time: {:.4f} s".format(end - start))

        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


    def log(self, str):
        if self.debug:
            print(str)


    def start(self):
        self.thread.start()


    def stop(self):
        cv2.destroyAllWindows()
        self.cfg.stop()


    def __setupWindow(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    def __draw_crop_box(self, image):
        cv2.rectangle(image, (self.cfg.x_start, self.cfg.y_start), (self.cfg.x_end, self.cfg.y_end), (0, 0, 192), 2)