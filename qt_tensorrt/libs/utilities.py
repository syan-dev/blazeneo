import os
import cv2
import numpy as np

def scan_dir(dir_paths):
    """
    Scan the original image directories, and save the path with key value.
    """
    paths = []
    for root, dirs, files in os.walk(dir_paths, topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))
    
    return paths

def create_dir(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")



class Preprocess:
    def __init__(self, width, height, mode):
        self.width = width
        self.height = height
        self.mode = mode 

    def get_input_shape(self):
        x = np.zeros((self.height, self.width, 3)).astype(np.uint8)
        y = self.run(x)
        return y.shape
    
    def run(self, image):
        if self.mode == 0:
            # rgb + hwc
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
        elif self.mode == 1:
            # rgb + chw  
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32)
        elif self.mode == 2:
            # rgb + hwc + normalize [-1, 1]
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            image = image/127.5-1
        elif self.mode == 3:
            # rgb + chw + normalize [-1, 1]
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32)
            image = image/127.5-1
        else:
            # bgr + hwc
            image = cv2.resize(image, (self.width, self.height))
            image = image.astype(np.float32)

        return image


class Postprocess:
    def __init__(self, width, height, n_classes, threshold, mode):
        self.width = width
        self.height = height
        self.n_classes = n_classes
        self.threshold = threshold
        self.mode = mode
    
    def run(self, image, output):
        """
        From Mask result, draw contour on image directly
        """
        if self.mode == 0:
            h, w = image.shape[:2]
            output[output<self.threshold] = 0
            if np.sum(output) > 0:
                masks = (output.reshape(self.height, self.width, self.n_classes)*255).astype(np.uint8)
                # Need to reimplement for multiple classes
                mask = masks[..., 0]

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
                erode_mask = cv2.erode(mask, kernel, iterations=2)
                dilate_mask = cv2.dilate(erode_mask, kernel, iterations=1)

                dilate_mask[erode_mask>0] = 0

                dilate_mask = cv2.resize(dilate_mask, (w, h), cv2.INTER_LINEAR)

                image[dilate_mask>0, 0], image[dilate_mask>0, 1], image[dilate_mask>0, 2] = (0, 255, 0)
        
        # elif self.mode == 1:
        #     h, w = image.shape[:2]
        #     output[output<self.threshold] = 0
        #     masks = (output.reshape(self.height, self.width, self.n_classes)*255).astype(np.uint8)
        #     mask = (masks[..., 0] * 255).astype(np.uint8)
        #     mask = cv2.resize(mask, (w, h), cv2.INTER_LINEAR)
        #     ret, labels = cv2.connectedComponents(mask)
        #     if ret > 0:
        #         for i in range(1, ret):
        #             polyp = labels==i
        #             mask_i = labels[polyp]
        #             print(np.sum(mask_i))
        #             image[polyp] = 255

        return image