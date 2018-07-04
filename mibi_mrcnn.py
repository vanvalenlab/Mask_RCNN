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
from mrcnn.deepcell_config import CellConfig
from mrcnn.deepcell_dataset import CellDataset
from mrcnn.deepcell_inference import InferenceConfig
from mrcnn import deepcell_traintest
from mrcnn.deepcell_traintest import train_model_withvalidation, train_model, test, random_colors, apply_mask, display_instances
from mrcnn import utils
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from imgaug import augmenters as iaa

TRAIN_DIR = '/Mask_RCNN/data/raw_train'
VAL_DIR = '/Mask_RCNN/data/raw_test'
MASK_DIR = '/Mask_RCNN/data/annotated'
MODEL_DIR = '/Mask_RCNN/models'
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = CellConfig()
##config.display()


#Create model
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

#initiate weights
init_with = "coco"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

#train model
train_model_withvalidation(model, TRAIN_DIR, VAL_DIR)


#train_model_withvalidation(model,dataset_dir,validation_dir,nepoch=40,config=CellConfig(),datasetclass=CellDataset)
