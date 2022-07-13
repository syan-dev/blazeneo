import os
import sys
import time
import cv2
import numpy as np
from queue import Queue

from libs.pipeline import Streamming, Inferencing, Displaying, Config
from libs.utilities import Preprocess, Postprocess

def running(cfg):
    while True:
        if cfg.is_stop():
            break


def main():
    config_file = "app_config.json"
    cfg = Config(config_file)

    preprocess = Preprocess(cfg.width, cfg.height, cfg.preprocess_mode)
    preprocessing = preprocess.run
    postprocess = Postprocess(cfg.width, cfg.height, cfg.n_classes, cfg.threshold, cfg.postprocess_mode)
    postprocessing = postprocess.run

    input_queue = Queue(maxsize=cfg.input_buffer_size)
    output_queue = Queue(maxsize=cfg.output_buffer_size)
    
    streamming = Streamming(input_queue, cfg)
    streamming.start()
    
    displaying = Displaying(output_queue, cfg)
    displaying.start()

    print("Warmming up ...")
    time.sleep(5)

    inferencing_1 = Inferencing(input_queue, output_queue, preprocessing, postprocessing, cfg, name="inferencing thread 1")
    inferencing_1.start()

    inferencing_2 = Inferencing(input_queue, output_queue, preprocessing, postprocessing, cfg, name="inferencing thread 2")
    inferencing_2.start()

    # inferencing_3 = Inferencing(input_queue, output_queue, preprocessing, postprocessing, cfg, name="inferencing thread 3")
    # inferencing_3.start()

    # inferencing_4 = Inferencing(input_queue, output_queue, preprocessing, postprocessing, cfg, name="inferencing thread 4")
    # inferencing_4.start()

    print("Running ...")
    running(cfg)

    time.sleep(5)


if __name__ == "__main__":
    main()