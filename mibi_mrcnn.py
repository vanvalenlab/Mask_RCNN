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
import tifffile as tiff

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
OUTPUT_DIR = '/Mask_RCNN/output'
WEIGHTS_PATH = '/Mask_RCNN/models/cell20180710T0023/mask_rcnn_cell_0020.h5'


#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

COCO_MODEL_PATH = './mask_rcnn_coco.h5'

TEST_IMAGE_1 = 'crop14_dsDNA1.tif'
TEST_IMAGE_2 = 'crop14_dsDNA2.tif'

config = CellConfig()
#config.display()

import warnings
warnings.filterwarnings("ignore")



def init_model():
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    #load COCO trained weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

if __name__ == '__main__':
    #parse command line input to either train or run mrcnn
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
        deepcell_traintest.train_model_withvalidation(model, TRAIN_DIR, VAL_DIR, nepoch=20)

    elif args.command == 'run':
        #recreate model in inference mode
        inference_config = InferenceConfig()
        model = modellib.MaskRCNN(mode='inference',
                                  config=inference_config,
                                  model_dir=MODEL_DIR)


        test_image_path = os.path.join(VAL_DIR, TEST_IMAGE_1)
#        image = deepcell_traintest.autotest(test_image_path)
#        images = deepcell_traintest.test(model, VAL_DIR, model_path=WEIGHTS_PATH)

        model_path = WEIGHTS_PATH        
        assert model_path != "", "Provide path to trained weights(The .h5 files)"
        print("Loading weights from: ", model_path)
        model.load_weights(model_path, by_name=True)

        class_names = ['BG', 'Cell']
        frame=skimage.io.imread(test_image_path)
        frame=np.expand_dims(frame, axis=-1)
 
        results = model.detect([frame], verbose=1)
        r = results[0]
        print('Mask shape is: ', r['masks'].shape)

        output = deepcell_traintest.display_instances_mibi(frame, r['rois'], r['masks'], 
                                          r['class_ids'], class_names, r['scores'])
        print('output shape is:', output.shape)

        save_name = 'mrcnn_output.tif'
        output_save_path = os.path.join(OUTPUT_DIR, save_name)
        tiff.imsave(output_save_path, output)
        

'''
tiff.imsave(os.path.join(output_location, cnnout_name), feature)

        answers = deepcell_traintest.test(model, VAL_DIR, model_path=model_path)
        class_names = ['BG', 'Cell']

        colors = deepcell_traintest.random_colors(len(class_names))
        class_dict = {
            name: color for name, color in zip(class_names, colors)
        }

def test(model,testset_dir,model_path=DEFAULT_MODEL_PATH):
        model_path = model.find_last()[1]
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True) '''







#train_model_withvalidation(model,dataset_dir,validation_dir,nepoch=40,config=CellConfig(),datasetclass=CellDataset)
