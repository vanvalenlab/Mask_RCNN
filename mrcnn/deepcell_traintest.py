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
from mrcnn import utils
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from imgaug import augmenters as iaa
DEFAULT_MODEL_PATH=os.path.join(ROOT_DIR,"mask_rcnn_nucleus_0040.h5")

def train_model_withvalidation(model,dataset_dir,validation_dir,nepoch=40,config=CellConfig(),datasetclass=CellDataset):
    """Train the model."""
    # Training dataset.
    dataset_train = CellDataset()
    dataset_train.load_trainingcells(dataset_dir)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellDataset()
    dataset_val.load_validationcells(dataset_dir)
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=nepoch//2,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=nepoch,
                augmentation=augmentation,
                layers='all')

def train_model(model,dataset_dir,nepoch=40,train_val_ratio=0.7,config=CellConfig(),datasetclass=CellDataset):
    dataset_train = datasetclass()
    #Finding out the most the images which would go as training.
    train_ids=dataset_train.train_validate_split(dataset_dir,train_val_ratio)
    dataset_train.train_validate_loadtrain(dataset_dir,train_ids)
    dataset_train.prepare()

    #Validation dataset
    dataset_val=datasetclass()
    dataset_val.train_validate_loadval(dataset_dir,train_ids)
    dataset_val.prepare()


    #Image augmentation
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=nepoch//2,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=nepoch,
                augmentation=augmentation,
                layers='all')


def test(model,testset_dir,model_path=DEFAULT_MODEL_PATH):
    assert model_path != "", "Provide path to trained weights(The .h5 files)"
    print("Loading weights from: ", model_path)
    model.load_weights(model_path, by_name=True)
    test_ids=next(os.walk(testset_dir))[2]
    answers=[]
    for i in range(len(test_ids)):
        frame=cv2.imread(os.path.join(testset_dir,test_ids[i]))
        results = model.detect([frame], verbose=0)
        r = results[0]
        answers.append(r)
    return answers


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

class_names = ['BG', 'Cell']

colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def apply_mask(image, mask, color, alpha=1):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)

    return image

def masked_image(model,image_path):
    frame=cv2.imread(image_path)
    results = model.detect([frame], verbose=0)
    r = results[0]
    frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return frame

def autotrain(dataset_dir,nepoch,model_dir=".",resume=False):
    print("Sit back, Relax! Your training would be taken care of!")
    print("The .h5(model) file would go to "+str(os.path.abspath(model_dir)))
    model = modellib.MaskRCNN(mode="training", config=CellConfig(),model_dir=model_dir)
    if resume == False:
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    else:
        model.load_weights(model.find_last()[1], by_name=True)
    dataset_train = CellDataset()
    #Finding out the most the images which would go as training.
    train_ids=dataset_train.train_validate_split(dataset_dir,0.7)
    dataset_train.train_validate_loadtrain(dataset_dir,train_ids)
    dataset_train.prepare()

    #Validation dataset
    dataset_val=CellDataset()
    dataset_val.train_validate_loadval(dataset_dir,train_ids)
    dataset_val.prepare()


    #Image augmentation
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=nepoch//2,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=nepoch,
                augmentation=augmentation,
                layers='all')


def autotest(testimage_path,model_path="."):
    model = modellib.MaskRCNN(mode="inference",
                          config=InferenceConfig(),
                          model_dir=model_path)
    model_path = model.find_last()[1]
    model.load_weights(model_path, by_name=True)
    image1=cv2.imread(testimage_path)
    results = model.detect([image1], verbose=0)
    r = results[0]
    frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return frame


def autotest_mymodel(testimage_path,model_path):
    model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(),model_dir=model_path)
    model.load_weights(model_path, by_name=True)
    image1=cv2.imread(testimage_path)
    results = model.detect([image1], verbose=0)
    r = results[0]
    frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return frame
