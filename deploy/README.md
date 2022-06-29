# Deploy the model on Jetson Xavier

## Step 1: Convert model from onnx to tensorrt format

There are 3 precision modes to convert a model: fp32, fp16, or int8. With int8, we need to calibrate model during the converting. For example

Convert model to fp16:

```
python3 build_engine.py \
    --onnx <path to onnx model> \
    --engine <path to save tensorrt model> \
    --precision fp16 \
    --verbose
```

Convert model to int8:

```
python3 build_engine.py \
    --onnx <path to onnx model> \
    --engine <path to save tensorrt model> \
    --verbose \
    --calib_input <path to directory contains calibrating images>
    --calib_batch_size 1 \ 
    --calib_preprocessor v2
```

The ``calib_preprocessor`` specify the way to normalize input image. ``v1`` normalizes in ImageNet format. ``v2`` will normalizes image pixel from 0 to 1.

## Step 2: Infer model

For inference, use following command:

```
python3 infer.py \
    --engine <path to tensorrt model> \
    --input <path to directory contains input images>
    --preprocessor v2 \
    --postprocessor v2 \
    --ouput <path to directory where save the output images>
```