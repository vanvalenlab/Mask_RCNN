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
#from scipy.ndimage.measurements import label
from skimage.measure import label
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class CellDataset(utils.Dataset):

    def load_trainingcells(self, dataset_dir):
        """
        dataset_dir: The directory of training images.
        """
        # Add classes. We have one class.
        # Naming the dataset cell, and the class cell
        self.add_class("cell", 1, "cell")
        image_ids = next(os.walk(dataset_dir))[2]
        #To suffle the images just to make sure that any bais due to
        #generation does not kick in.
        image_ids = list(set(image_ids))

        # Add images
        for image_id in image_ids:

            self.add_image(
                "cell",
                image_id=image_id,
                path=os.path.join(dataset_dir,image_id))

    def load_validationcells(self,dataset_dir):
        #dataset_dir: The directory of validation images.
        self.add_class("cell",1,"cell")
        image_ids=next(os.walk(dataset_dir))[2]
        image_ids=list(set(image_ids))
        for image_id in image_ids:
            self.add_image("cell",image_id=image_id,path=os.path.join(dataset_dir,image_id))

    def train_validate_split(self,dataset_dir,ratio=0.7):
        #If the validation set needs to be a part of the training data
        #We need to randomly select out of it ,by the ratio=training/validation
        image_ids=next(os.walk(dataset_dir))[2]
        splitratio=int((1-ratio)*len(image_ids))
        train_ids=np.random.choice(image_ids,splitratio)
        for i in range(len(train_ids)):
            train_ids[i]=str(os.path.join(dataset_dir,train_ids[i]))
        #returns the array of images to train
        return train_ids

    def train_validate_loadtrain(self,dataset_dir,train_ids):
        self.add_class("cell", 1, "cell")
        for image_id in train_ids:
            self.add_image("cell",image_id=image_id,path=image_id)

    def train_validate_loadval(self,dataset_dir,train_ids):
        self.add_class("cell",1,"cell")
        image_ids=next(os.walk(dataset_dir))[2]
        val_ids=list(set(image_ids)-set(train_ids))
        for image_id in val_ids:
            self.add_image("cell",image_id=image_id,path=image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        image_name=info['path'].split("/")[-1]
        # Get mask directory from image path
        #mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        mask_dir = '/Mask_RCNN/data/annotated'

        #Get the exact full mask for that image
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(str(image_name)):
#                full_mask = cv2.imread(os.path.join(mask_dir, f),0)
                full_mask = skimage.io.imread(os.path.join(mask_dir,f))
#needtoadddimension

        #extract indivial masks of cells from the full mask
        
        if (np.max(np.unique(full_mask)) > 3):
          
            lb = full_mask
            lb = lb.astype('int32')  
        #if masks are not uniquely annotated, label them as such
        else:
            print('labeling')
            lb = label(full_mask)


        msks = []
        #for key in range(1,lb[1]+1):
#        for key in range(1, np.max(np.unique(lb))+1):
        for key in np.unique(lb):
            if key == 0:
                continue

            #x = lb[1]+1
            #x = np.max(np.unique(lb))

            newim=np.zeros(full_mask.shape)
            for i in range(full_mask.shape[0]):
                for j in range(full_mask.shape[1]):
                    if lb[i][j]==key:
                        newim[i][j]=1
            msks.append(newim)
        #msks=np.astype(np.bool)

        # Combine these masks of indiviual cells
        if np.max(np.unique(lb))>2:
            mask = np.stack(msks, axis=-1)

        else:
            mask = []

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
