import os
import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import tensorrt as trt


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    # binding_to_type = {"data": np.int32, "sigmoid/Sigmoid": np.float32}
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # dtype = binding_to_type[str(binding)]
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def GiB(val):
    return val * 1 << 30


def build_engine_from_uff(model_file_path, max_batch_size, max_workspace_size, input_name, input_shape, output_name, mode, int8_calibrator):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size (maxBatchSize is the size for which the engine will be tuned. At execution time, smaller batches may be used, but not larger).
        builder.max_batch_size = max_batch_size
        # max_workspace_size int, The maximum GPU temporary memory which the ICudaEngine can use at execution time 
        builder.max_workspace_size = GiB(max_workspace_size)
        if mode == "fp16":
            builder.fp16_mode = True
        elif mode == "int8":
            builder.int8_mode=True
            builder.int8_calibrator=int8_calibrator
        # Parse the Uff Network
        parser.register_input(input_name, input_shape)
        parser.register_output(output_name)
        parser.parse(model_file_path, network)
        # Build and return an engine.
        engine = builder.build_cuda_engine(network)
        return engine


def build_engine_from_onnx(model_file_path, max_batch_size, max_workspace_size, output_name, mode, int8_calibrator):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size (maxBatchSize is the size for which the engine will be tuned. At execution time, smaller batches may be used, but not larger).
        builder.max_batch_size = max_batch_size
        # max_workspace_size int (GB), The maximum GPU temporary memory which the ICudaEngine can use at execution time 
        builder.max_workspace_size = GiB(max_workspace_size)
        if mode == "fp16":
            builder.fp16_mode = True
        elif mode == "int8":
            builder.int8_mode=True
            builder.int8_calibrator=int8_calibrator
        # Parse the ONNX Network
        with open(model_file_path, "rb") as model:
            parser.parse(model.read())
        # Build and return an engine
        engine = builder.build_cuda_engine(network)
        return engine


def save_engine(engine, engine_dest_path):
    buf = engine.serialize()
    with open(engine_dest_path, "wb") as f:
        f.write(buf)


def load_engine(engine_path):
    trt_runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine