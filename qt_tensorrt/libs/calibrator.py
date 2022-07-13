import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
import tensorrt as trt
import os
import cv2


class PythonEntropyCalibrator(trt.tensorrt.IInt8EntropyCalibrator):
    def __init__(self, input_layers, stream):
        trt.tensorrt.IInt8EntropyCalibrator.__init__(self)
        self.input_layers = input_layers
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = "cache_file.bin"
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, bindings, names):
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i

        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self):
        return None
        # if os.path.exists(self.cache_file):
        #     with open(self.cache_file, "rb") as f:
        #         cache = f.read()
        #         return cache

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)



class ImageBatchStream():
    def __init__(self, batch_size, input_shape, calibration_files, preprocess_fn):
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // self.batch_size) + \
                           (1 if (len(calibration_files) % self.batch_size)
                            else 0)
        self.files = calibration_files
        black_image = np.zeros(input_shape, dtype=np.float32)
        self.calibration_data = np.stack((black_image,)*self.batch_size, axis=0)
        self.batch = 0
        self.preprocess_fn = preprocess_fn

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch:
                                         self.batch_size * (self.batch + 1)]
            for fp in files_for_batch:
                print("[ImageBatchStream] Processing ", fp)
                img = cv2.imread(fp)
                img = self.preprocess_fn(img)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])