import numpy as np
import random
from libs.common import build_engine_from_onnx, build_engine_from_uff, save_engine
from libs.utilities import scan_dir, Preprocess
from libs.calibrator import ImageBatchStream, PythonEntropyCalibrator


def convert_to_trt(ori_file, trt_file, max_batch_size, max_workspace_size, input_name, output_name, width, height, builder_mode, calibration_batch_size, calibration_dir, preprocess_mode):
    """
    Convert model from uff, onnx format to trt format
    Params:
        builder_mode: "int8" or "fp16"
        preprocess_mode: used for preprocessing image for calibration, all mode are defined in class Preprocess
    """
    if builder_mode == "int8":
        calibration_files = scan_dir(calibration_dir)
        random.shuffle(calibration_files)
        preprocess = Preprocess(width, height, preprocess_mode)
        input_shape = preprocess.get_input_shape()
        preprocess_fn = preprocess.run
        batchstream = ImageBatchStream(calibration_batch_size, input_shape, calibration_files, preprocess_fn)
        int8_calibrator = PythonEntropyCalibrator([input_name], batchstream)
    else:
        int8_calibrator = None 

    file_type = ori_file.split(".")[-1]
    if file_type == "onnx":
        print("building engine from onnx file at {}".format(ori_file))
        engine = build_engine_from_onnx(ori_file, max_batch_size, max_workspace_size, output_name, builder_mode, int8_calibrator)
        print("saving engine at {}".format(trt_file))
        save_engine(engine, trt_file)
    elif file_type == "uff":
        input_shape = (3, height, width)
        print("building engine from uff file at {}".format(ori_file))
        engine = build_engine_from_uff(ori_file, max_batch_size, max_workspace_size, input_name, input_shape, output_name, builder_mode, int8_calibrator)
        print("saving engine at {}".format(trt_file))
        save_engine(engine, trt_file)


def main():
    convert_to_trt(
        ori_file="/home/vdsense/Documents/data/models/unet_b4_320x320_040520_120520_140520_180520_240520_260520_280520_040620.onnx",
        trt_file="/home/vdsense/Documents/data/models/unet_b4_320x320_040520_120520_140520_180520_240520_260520_280520_040620_fp32_b1_w1.trt",
        max_batch_size = 1,
        max_workspace_size = 1,
        builder_mode="fp32",    
        input_name="input_1",   # uff
        output_name="subtract/sub", # uff
        width=320,
        height=320,
        calibration_batch_size=1,
        calibration_dir="/home/vdsense/Documents/data/images/dot7_dot8", 
        preprocess_mode=2 
    )  


if __name__ == "__main__":
    main()