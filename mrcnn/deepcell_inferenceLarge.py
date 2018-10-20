if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import cv2
from scipy.ndimage.measurements import label
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn.deepcell_dataset import CellDataset
from mrcnn import utils

from mrcnn.deepcell_config import CellConfig

class InferenceConfig(CellConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    #You may increse this if you have a super small or blurry segmentation object
    RPN_NMS_THRESHOLD = 0.7



class InferenceConfigLarge(CellConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    #You may increse this if you have a super small or blurry segmentation object
    RPN_NMS_THRESHOLD = 0.7

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 7000

