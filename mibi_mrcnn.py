import numpy as np
import os
import errno                #error symbols
import argparse             #command line input parsing
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



config = CellConfig()
##config.display()





def init_model():
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    #load COCO trained weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=['train', 'run', 'export'],
                        help='train or run models')
    parser.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                        help='force re-write of training data npz files')

    args = parser.parse_args()

    if args.command == 'train':
        #Create model
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
        init_model()

        #train model
        deepcell_traintest.train_model_withvalidation(model, TRAIN_DIR, VAL_DIR, nepoch=2)

    elif args.command == 'run':
        inference_config = InferenceConfig()

        #recreate model in inference mode
        model = modellib.MaskRCNN(mode='inference',
                                  config=inference_config,
                                  model_dir=MODEL_DIR)

        model_path = model.find_last()[1]
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)







#train_model_withvalidation(model,dataset_dir,validation_dir,nepoch=40,config=CellConfig(),datasetclass=CellDataset)
