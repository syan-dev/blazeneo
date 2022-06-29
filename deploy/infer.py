import os
import sys
import argparse
import cv2

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from image_batcher import ImageBatcher

class TensorRTInfer:

    def __init__(self, engine_path):
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setpup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        
        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the networl. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]['shape'], self.outputs[0]['dtype']

    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        """
        # Prepare the output data
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(output, self.outputs[0]['allocation'])

        # print(output)
        return output


def main(args):
    trt_infer = TensorRTInfer(args.engine)
    # print(trt_infer.input_spec())
    # print(trt_infer.output_spec())
    batcher = ImageBatcher(args.input, *trt_infer.input_spec(), preprocessor=args.preprocessor)
    for batch, images in batcher.get_batch():
        batch_output = trt_infer.infer(batch)
        for i in range(len(images)):
            output = batch_output[i]

            if args.postprocessor == 'v1':
                output = np.argmax(output, axis=0)
                h, w = output.shape[:2]
                mask = np.zeros((h, w, 3), dtype=np.uint8)
                mask[output == 0] = (0, 0, 255)
                mask[output == 1] = (0, 255, 0)
                mask[output == 2] = (0, 0, 0)
            elif args.postprocessor == 'v2':
                mask = np.argmax(output, axis=0)
                h, w = mask.shape[:2]
                neo = (mask == 1) * (np.max(output, axis=0) >= 0.5)
                non = (mask == 0) * (np.max(output, axis=0) >= 0.5)

                mask = np.zeros((h, w, 3), dtype=np.uint8)
                mask[neo>0] = (0, 0, 255)
                mask[non>0] = (0, 255, 0)
            else:
                print("Postprocessing method {} not supported".format(args.postprocessor))

            image_path = images[i]
            saved_path, _ = os.path.splitext(os.path.basename(image_path)) 
            saved_path = os.path.join(args.output, saved_path) + '.png'
            cv2.imwrite(saved_path, mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', help='The TensorRT engine to infer with')
    parser.add_argument('-i', '--input', help='The input to infer, either a single image path, or a directory of images')
    parser.add_argument('--preprocessor', default='v1', choices=['v1', 'v2'], help='Select the image preprocessor to use, either v1, or v2')
    parser.add_argument('--postprocessor', default='v1', choices=['v1', 'v2'], help='Select the image preprocessor to use, either v1, or v2')
    parser.add_argument('-o', '--output', help='Path to the saved directory')

    args = parser.parse_args()
    if not all([args.engine, args.input]):
        parser.print_help()
        print("\nThese arguments are required: --engine and --input")
        sys.exit(1)
    main(args)