# generate mrcnn data


import numpy as np
import skimage.io as sk
import skimage.external.tifffile as tiff
import os


import os
import errno
import argparse

import numpy as np
import skimage.io
import skimage.external.tifffile as tiff
import skimage.morphology
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import backend as K
from scipy import stats

from deepcell import get_image_sizes
from deepcell import make_training_data
from deepcell import bn_feature_net_31x31
from deepcell import dilated_bn_feature_net_31x31
from deepcell import bn_feature_net_61x61           #model_zoo
from deepcell import dilated_bn_feature_net_61x61


from deepcell import train_model_watershed
from deepcell import train_model_watershed_sample
from deepcell import bn_dense_feature_net
from deepcell import rate_scheduler
from deepcell import train_model_disc, train_model_conv, train_model_sample
from deepcell import run_models_on_directory
from deepcell import export_model
from deepcell import get_data

DATA_OUTPUT_MODE = 'sample'
BORDER_MODE = 'valid' if DATA_OUTPUT_MODE == 'sample' else 'same'
RESIZE = True
RESHAPE_SIZE = 2048
N_EPOCHS = 40
WINDOW_SIZE = (15,15)
BATCH_SIZE = 64
MAX_TRAIN = 1e8

#SET_RANGE = range(1, 41+1)
SET_RANGE = range(1, 5+1)

CROP_SIZE = 512
 

EDGE_THRESH = 0.25
INT_THRESH = 0.25
CELL_THRESH = 0.25
NUM_FINAL_EROSIONS = 1 

# Check for channels_first or channels_last
IS_CHANNELS_FIRST = K.image_data_format() == 'channels_first'
ROW_AXIS = 2 if IS_CHANNELS_FIRST else 1
COL_AXIS = 3 if IS_CHANNELS_FIRST else 2
CHANNEL_AXIS = 1 if IS_CHANNELS_FIRST else -1

# filepath constants
DATA_DIR = '/data/data'
MODEL_DIR = '/data/models'
NPZ_DIR = '/data/npz_data'
RESULTS_DIR = '/data/results'
EXPORT_DIR = '/data/exports'

PREFIX_SEG = 'tissues/mibi/samir'
PREFIX_CLASS = 'tissues/mibi/mibi_full'

PREFIX_SAVE = 'tissues/mibi/pipeline'

# output filepaths
MRCNN_DATA_DIR = './data'
MRCNN_ANNO_DIR = 'annotated'
MRCNN_TRAIN_DIR = 'raw_train'
MRCNN_TEST_DIR = 'raw_test'

NUM_FEATURES_IN_SEG = 2
NUM_FEATURES_OUT_SEG = 3
NUM_FEATURES_CLASS = 17

#5chan
#MODEL_FGBG = '2018-07-13_mibi_31x31_channels_last_sample__0.h5'
#CHANNELS_SEG = ['dsDNA', 'Ca', 'H3K27me3', 'H3K9ac', 'Ta']

MODEL_FGBG = '2018-08-20_mibi_31x31_8chanCFHHNPTd__channels_last_sample__0.h5'
CHANNELS_SEG = ['Ca.', 'Fe.', 'H3K27me3', 'H3K9ac', 'Na.', 'P.', 'Ta.', 'dsDNA.']

DTYPE = 'uint16'
MASK_DTYPE = 'uint8'
WIN_SIZE = (15, 15)
NUM_CHAN = 3
PAD = 15

def dilate(array, mask, num_dilations):
    copy = np.copy(array)
    for x in range(0, num_dilations):
        dilated = skimage.morphology.dilation(copy)

        # if still within the mask range AND one cell not eating another, dilate
        #copy = np.where( ((mask!=0) & (dilated!=copy & copy==0)), dilated, copy)
        copy = np.where( (mask!=0) & (dilated!=copy) & (copy==0), dilated, copy)
    return copy

def dilate_nomask(array, num_dilations):
    copy = np.copy(array)
    for x in range(0, num_dilations):
        dilated = skimage.morphology.dilation(copy)

        # if one cell not eating another, dilate
        #copy = np.where( ((mask!=0) & (dilated!=copy & copy==0)), dilated, copy)
        copy = np.where( (dilated!=copy) & (copy==0), dilated, copy)
    return copy


def erode(array, num_erosions):
    original = np.copy(array)

    for x in range(0, num_erosions):
        eroded = skimage.morphology.erosion(np.copy(original))
        original[original != eroded] = 0

    return original


def concat_channels(dsDNA, f0, f1):

    f0 = np.squeeze(f0)

    print('shape of dsDNA is:', dsDNA.shape, ' f0 is:', f0.shape, ' f1 is:', f1.shape)
    print('type of dsDNA is:', dsDNA.dtype, ' f0 is:', f0.dtype)

    # make inputs compatible
#    dsDNA = dsDNA.astype(DTYPE)
    #dsDNA = dsDNA / dsDNA.max()


    f0 = (f0*100).astype(DTYPE)
    f1 = (f1*100).astype(DTYPE)

    # print image info
    print('dsDNA shape is:', dsDNA.shape, ', type is:', dsDNA.dtype, ' max is:', dsDNA.max())
    print('feature shape is:', f0.shape, ', type is:', f0.dtype, ' max is:', f0.max())

    # pad images back to 2048 x 2048
    dsDNA = np.pad(dsDNA, WIN_SIZE, 'constant')
    f0 = np.pad(f0, WIN_SIZE, 'constant')
    f1 = np.pad(f1, WIN_SIZE, 'constant')

    # make empty output image
    output = np.zeros((dsDNA.shape[0], dsDNA.shape[1], NUM_CHAN), dtype=DTYPE)

    print('padded feature shape is:', f0.shape)
    print('output shape is:', output.shape)
    print('')

    # insert each layer into output
    output[:,:,0] = dsDNA[:,:]
    output[:,:,1] = f0[:,:]
    output[:,:,2] = f1[:,:]

    return output

def run_segmentation(set):

    print('Segmenting set', set)

    raw_dir = 'raw'
    data_location = os.path.join(DATA_DIR, PREFIX_CLASS, set, raw_dir)
    output_location = os.path.join(RESULTS_DIR, PREFIX_SEG)
    image_size_x, image_size_y = get_image_sizes(data_location, CHANNELS_SEG)

    weights = os.path.join(MODEL_DIR, PREFIX_SEG, MODEL_FGBG)

    n_features = 3
    window_size = (30, 30)

    if DATA_OUTPUT_MODE == 'sample':
        model_fn = dilated_bn_feature_net_31x31                                 #changed to 21x21
    elif DATA_OUTPUT_MODE == 'conv':
        model_fn = bn_dense_feature_net
    else:
        raise ValueError('{} is not a valid training mode for 2D images (yet).'.format(
            DATA_OUTPUT_MODE))

    predictions = run_models_on_directory(
        data_location=data_location,
        channel_names=CHANNELS_SEG,
        output_location=output_location,
        n_features=n_features,
        model_fn=model_fn,
        list_of_weights=[weights],
        image_size_x=image_size_x,
        image_size_y=image_size_y,
        win_x=WINDOW_SIZE[0],
        win_y=WINDOW_SIZE[1],
        split=False)

    #0.25 0.25 works good
    edge_thresh = EDGE_THRESH
    interior_thresh = INT_THRESH
    cell_thresh = CELL_THRESH

    print('shape of predictions is:', predictions.shape)

    edge = np.copy(predictions[:,:,:,0])
    edge[edge < edge_thresh] = 0
    edge[edge > edge_thresh] = 1

    interior = np.copy(predictions[:, :, :, 1])
    interior[interior > interior_thresh] = 1
    interior[interior < interior_thresh] = 0

    cell_notcell = 1 - np.copy(predictions[:, :, :, 2])
    cell_notcell[cell_notcell > cell_thresh] = 1
    cell_notcell[cell_notcell < cell_thresh] = 0

    # define foreground as the interior bounded by edge
    fg_thresh = np.logical_and(interior==1, edge==0)

    # remove small objects from the foreground segmentation
    fg_thresh = skimage.morphology.remove_small_objects(fg_thresh, min_size=50, connectivity=1)

    #fg_thresh = skimage.morphology.binary_erosion(fg_thresh)
    #fg_thresh = skimage.morphology.binary_dilation(fg_thresh)

    fg_thresh = np.expand_dims(fg_thresh, axis=CHANNEL_AXIS)

    watershed_segmentation = skimage.measure.label(  np.squeeze(fg_thresh), connectivity=2)


    # dilate gradually into the mask area
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)
    watershed_segmentation = erode(watershed_segmentation, 1)
    watershed_segmentation = dilate(watershed_segmentation, interior, 2)


    watershed_segmentation = dilate_nomask(watershed_segmentation, 1)
    watershed_segmentation = erode(watershed_segmentation, 2)
    watershed_segmentation = dilate_nomask(watershed_segmentation, 2)
    watershed_segmentation = erode(watershed_segmentation, NUM_FINAL_EROSIONS)

    index = 0

    output_location = os.path.join(RESULTS_DIR, PREFIX_SAVE)
    print('saving to: ', output_location)

    dsDNA = tiff.imread(os.path.join(data_location, 'dsDNA.tif'))
    dsDNA = dsDNA[15:-15, 15:-15]

    watershed_segmentation = watershed_segmentation.astype('float32')

    cell_edge = predictions[index, :, :, 0]

    tiff.imsave(os.path.join(output_location, 'shallowWatershed_instance_seg.tif'), watershed_segmentation)

    return watershed_segmentation, dsDNA, cell_notcell, cell_edge

def crop_im_and_save(img, save_file_path, im_size, crop_size, im_name, mask):

    print('shape of image is:', img.shape)

    print('crop size is:', crop_size)

    crop_counter = 0
    for x in range(0, im_size, crop_size):
        for y in range(0, im_size, crop_size):

            crop = np.zeros((crop_size, crop_size))
            crop = img[x:x+crop_size, y:y+crop_size]
            crop_num = str(crop_counter)

            
            out_file_path = save_file_path + '/' + mask + 'crop' + crop_num + '_' + im_name
            tiff.imsave( out_file_path, crop)
            crop_counter += 1

def process_set(set_num):

    set_dir = 'set' + str(set_num)
    mask, raw, interior, edge = run_segmentation(set_dir)

#    if set_num == 1 or set_num == 2:
#        mask = skimage.io.imread( os.path.join('/data/data/tissues/mibi/samir', set_dir, 'annotated/feature_1.tif') )
#        mask = mask[15:-15, 15:-15]

        # remove small objects from the mask


#        mask = skimage.measure.label( mask, connectivity=2)

#        mask = skimage.morphology.remove_small_objects(mask, min_size=40, connectivity=1)
 #   else:
 #       mask[mask > 0] = 1


    # concatenate raw, interior, and edge into one 3 channel image
    output = concat_channels(raw, interior, edge)

    # set mask to uint8
    mask = mask.astype('uint8')

    output_name = 'dsDNA' + str(set_num) + '.tif'
    mask_name = 'mask_dsDNA' + str(set_num) + '.tif'

    print(raw.sum())
    print(mask.sum())

    
    print('cropping input')
    crop_im_and_save(output, os.path.join(MRCNN_DATA_DIR, MRCNN_TRAIN_DIR), 2048, CROP_SIZE, output_name, mask='')

    mask = np.pad( np.squeeze(mask), WIN_SIZE, 'constant')
    print('cropping mask')
    crop_im_and_save(mask, os.path.join(MRCNN_DATA_DIR, MRCNN_ANNO_DIR), 2048, CROP_SIZE, output_name, mask='mask_')

if __name__ == '__main__':

    # make storage directories
    if not os.path.exists(MRCNN_DATA_DIR):
        os.mkdir(MRCNN_DATA_DIR)
        os.mkdir(os.path.join(MRCNN_DATA_DIR, MRCNN_ANNO_DIR))
        os.mkdir(os.path.join(MRCNN_DATA_DIR, MRCNN_TRAIN_DIR))
        os.mkdir(os.path.join(MRCNN_DATA_DIR, MRCNN_TEST_DIR))

    # generate training data for all sets
    for set_num in SET_RANGE:
        if set_num == 99:
            continue
        else:
            process_set(set_num)
